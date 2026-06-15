#!/usr/bin/env python3
"""DeepSeek-style static-shape Gemma flatten-MatMul experiment.

For each fixed shape this script can compile baseline and flatten_matmul VMFBs,
benchmark both variants, keep per-shape JSON results, and write one integrated
summary under ./results.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
STANDARD_SCRIPT = ROOT / "run_gemma_standard_iree.py"
FLATTEN_SCRIPT = ROOT / "run_gemma_flatten_compare.py"

DEFAULT_SHAPES = ["b1_s16", "b1_s32", "b2_s16", "b2_s32", "b4_s16", "b4_s32"]


def parse_shape(shape: str) -> tuple[int, int]:
    try:
        batch, seq = shape.lower()[1:].split("_s", 1)
        return int(batch), int(seq)
    except Exception as exc:
        raise argparse.ArgumentTypeError(f"invalid shape '{shape}', expected b4_s32") from exc


def shape_id(batch: int, seq: int) -> str:
    return f"b{batch}_s{seq}"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapes", nargs="+", default=DEFAULT_SHAPES)
    parser.add_argument(
        "--action",
        choices=["plan", "compile", "benchmark", "all", "summarize"],
        default="plan",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=["baseline", "flatten_matmul"],
        default=["baseline", "flatten_matmul"],
    )
    parser.add_argument("--model-path", type=Path, default=ROOT / "googlegemma-4-E4B-it")
    parser.add_argument(
        "--model-label",
        default="gemma_e4b_it",
        help="Prefix used in experiment names, e.g. gemma_e2b_it.",
    )
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--export-device", default="cuda")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cuda-target", default="sm_86")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--min-time", default="1s")
    parser.add_argument("--timeout-seconds", type=int, default=43200)
    parser.add_argument("--force-export", action="store_true")
    parser.add_argument("--force-import", action="store_true")
    parser.add_argument("--force-compile", action="store_true")
    parser.add_argument("--force-benchmark", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--stop-on-benchmark-failure", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results" / "gemma_static_flatten_summary.json",
    )
    return parser.parse_args()


def require_tool(name: str) -> str:
    tool = shutil.which(name)
    if not tool:
        raise RuntimeError(f"Missing required tool: {name}")
    return tool


def experiment_name(model_label: str, batch: int, seq: int, dtype: str) -> str:
    dtype_label = {"float16": "fp16", "bfloat16": "bf16", "float32": "fp32"}[dtype]
    return f"{model_label}_b{batch}_s{seq}_{dtype_label}_cuda_standard"


def paths_for(shape: str, args):
    batch, seq = parse_shape(shape)
    exp_name = experiment_name(args.model_label, batch, seq, args.dtype)
    exp = ROOT / "experiments" / exp_name
    return {
        "shape": shape_id(batch, seq),
        "batch": batch,
        "seq": seq,
        "token_slots": batch * seq,
        "experiment": exp_name,
        "exp": exp,
        "baseline_build": exp / "standard_cuda",
        "flatten_build": exp / "standard_cuda_flatten_matmul",
        "baseline_vmfb": exp / "standard_cuda" / "gemma_cuda.vmfb",
        "baseline_params": exp / "standard_cuda" / "gemma_params.irpa",
        "flatten_vmfb": exp / "standard_cuda_flatten_matmul" / "gemma_flatten_cuda.vmfb",
        "flatten_params": exp / "standard_cuda_flatten_matmul" / "gemma_flatten_params.irpa",
        "baseline_benchmark": exp / "benchmark_baseline.googlebench.json",
        "flatten_benchmark": exp / "benchmark_flatten_matmul.googlebench.json",
        "baseline_log": exp / "logs" / "benchmark_baseline.log",
        "flatten_log": exp / "logs" / "benchmark_flatten_matmul.log",
        "compare": exp / "benchmark_flatten_matmul_compare.json",
        "rewrite_report": exp / "flatten_matmul_rewrite_report.json",
    }


def print_command(cmd: list[object]):
    print("Command:")
    print("  " + " \\\n  ".join(shlex.quote(str(part)) for part in cmd))


def run(cmd: list[object], log_path: Path, args, env=None, check: bool = True) -> int:
    print_command(cmd)
    print(f"Log: {log_path}")
    if args.dry_run or args.action == "plan":
        return 0
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = datetime.now()
    result = subprocess.run(
        [str(part) for part in cmd],
        capture_output=True,
        text=True,
        timeout=args.timeout_seconds,
        env=env,
    )
    log_path.write_text(
        "\n".join(
            [
                f"Started: {started.isoformat()}",
                f"Finished: {datetime.now().isoformat()}",
                f"Return code: {result.returncode}",
                "",
                "Command:",
                " ".join(shlex.quote(str(part)) for part in cmd),
                "",
                "STDOUT:",
                result.stdout or "",
                "",
                "STDERR:",
                result.stderr or "",
            ]
        ),
        encoding="utf-8",
    )
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    if result.returncode != 0 and check:
        raise RuntimeError(f"command failed with code {result.returncode}; see {log_path}")
    return result.returncode


def summary_succeeded(path: Path) -> bool:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(data.get("benchmarks"))


def benchmark_failure(log_path: Path) -> dict | None:
    if not log_path.exists():
        return None
    text = log_path.read_text(encoding="utf-8", errors="replace")
    returncode = None
    for line in text.splitlines():
        if line.startswith("Return code:"):
            try:
                returncode = int(line.split(":", 1)[1].strip())
            except ValueError:
                pass
            break
    reason = None
    for line in text.splitlines():
        if (
            "RESOURCE_EXHAUSTED" in line
            or "CUDA_ERROR_OUT_OF_MEMORY" in line
            or "ABORTED" in line
            or "FAILED_PRECONDITION" in line
        ):
            reason = line.strip()
            break
    if reason is None:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        reason = lines[-1] if lines else "benchmark command failed"
    return {"log": str(log_path), "returncode": returncode, "reason": reason}


def compile_variant(ps: dict, variant: str, args):
    if variant == "baseline":
        vmfb = ps["baseline_vmfb"]
        params = ps["baseline_params"]
        if vmfb.exists() and params.exists() and not args.force_compile and not args.force_import:
            print(f"Reusing baseline VMFB: {vmfb}")
            return
        cmd = [
            "python3",
            STANDARD_SCRIPT,
            "--action",
            "compile",
            "--model-path",
            args.model_path,
            "--experiment-name",
            ps["experiment"],
            "--batch",
            ps["batch"],
            "--seq",
            ps["seq"],
            "--dtype",
            args.dtype,
            "--export-device",
            args.export_device,
            "--cuda-target",
            args.cuda_target,
        ]
        if args.force_export:
            cmd.append("--force-export")
        if args.force_import:
            cmd.append("--force-import")
        if args.force_compile:
            cmd.append("--force-compile")
        if args.trust_remote_code:
            cmd.append("--trust-remote-code")
        run(cmd, ps["exp"] / "logs" / "compile_baseline.log", args)
        return

    vmfb = ps["flatten_vmfb"]
    params = ps["flatten_params"]
    if vmfb.exists() and params.exists() and not args.force_compile and not args.force_import:
        print(f"Reusing flatten VMFB: {vmfb}")
        return
    cmd = [
        "python3",
        FLATTEN_SCRIPT,
        "--action",
        "compile",
        "--model-path",
        args.model_path,
        "--experiment-name",
        ps["experiment"],
        "--batch",
        ps["batch"],
        "--seq",
        ps["seq"],
        "--dtype",
        args.dtype,
        "--cuda-target",
        args.cuda_target,
    ]
    if args.force_import:
        cmd.append("--force-import")
    if args.force_compile:
        cmd.append("--force-compile")
    if args.trust_remote_code:
        cmd.append("--trust-remote-code")
    run(cmd, ps["exp"] / "logs" / "compile_flatten_matmul.log", args)


def benchmark_variant(ps: dict, variant: str, args):
    if variant == "baseline":
        vmfb = ps["baseline_vmfb"]
        params = ps["baseline_params"]
        out = ps["baseline_benchmark"]
        log = ps["baseline_log"]
    else:
        vmfb = ps["flatten_vmfb"]
        params = ps["flatten_params"]
        out = ps["flatten_benchmark"]
        log = ps["flatten_log"]

    if out.exists() and summary_succeeded(out) and not args.force_benchmark:
        print(f"Reusing benchmark: {out}")
        return True
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": args.gpu} if args.gpu is not None else None
    returncode = run(
        [
            require_tool("iree-benchmark-module"),
            "--cuda_async_allocations=false",
            f"--module={vmfb}",
            f"--parameters=model={params}",
            f"--device={args.device}",
            "--function=main_graph",
            f"--input={ps['batch']}x{ps['seq']}xi64=0",
            f"--input={ps['batch']}x{ps['seq']}xi64=1",
            f"--benchmark_repetitions={args.repetitions}",
            f"--benchmark_min_time={args.min_time}",
            "--benchmark_format=console",
            f"--benchmark_out={out}",
            "--benchmark_out_format=json",
        ],
        log,
        args,
        env=env,
        check=args.stop_on_benchmark_failure,
    )
    if returncode != 0:
        print(f"Benchmark failed for {ps['shape']} {variant}; continuing.")
        return False
    return summary_succeeded(out)


def load_googlebench(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    benches = data.get("benchmarks", [])
    mean = next((item for item in benches if item.get("name", "").endswith("_mean")), None)
    median = next((item for item in benches if item.get("name", "").endswith("_median")), None)
    if mean is None and benches:
        mean = benches[0]
    if mean is None:
        raise ValueError(f"benchmark row not found in {path}")
    def to_ms(entry):
        if entry is None:
            return None
        real_time = entry.get("real_time")
        time_unit = entry.get("time_unit", "ms")
        if real_time is None:
            return None
        if time_unit == "ns":
            return real_time / 1_000_000.0
        if time_unit == "us":
            return real_time / 1_000.0
        if time_unit == "s":
            return real_time * 1000.0
        return real_time

    real_time = mean.get("real_time")
    time_unit = mean.get("time_unit", "ms")
    if time_unit == "ns":
        real_time_ms = real_time / 1_000_000.0
    elif time_unit == "us":
        real_time_ms = real_time / 1_000.0
    elif time_unit == "s":
        real_time_ms = real_time * 1000.0
    else:
        real_time_ms = real_time
    return {
        "summary": str(path),
        "real_time_ms_mean": real_time_ms,
        "real_time_ms_median": to_ms(median),
        "batch_iterations_per_second": mean.get("items_per_second"),
        "entry": mean.get("name"),
    }


def compare_if_possible(ps: dict):
    if not summary_succeeded(ps["baseline_benchmark"]) or not summary_succeeded(ps["flatten_benchmark"]):
        return None
    baseline = load_googlebench(ps["baseline_benchmark"])
    flat = load_googlebench(ps["flatten_benchmark"])
    for result in (baseline, flat):
        ips = result.get("batch_iterations_per_second") or 0.0
        result["sequences_per_second"] = ips * ps["batch"]
        result["token_slots_per_second"] = ips * ps["token_slots"]
    baseline_ms = baseline["real_time_ms_mean"]
    flat_ms = flat["real_time_ms_mean"]
    compare = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "batch": ps["batch"],
            "seq": ps["seq"],
            "shape": ps["shape"],
            "token_slots": ps["token_slots"],
        },
        "baseline": baseline,
        "flatten_matmul": flat,
        "speedup": baseline_ms / flat_ms if flat_ms else None,
        "latency_reduction_percent": (1.0 - flat_ms / baseline_ms) * 100.0 if baseline_ms else None,
    }
    performance = {
        "baseline": {
            "summary_file": str(ps["baseline_benchmark"].relative_to(ROOT)),
            "latency_ms_mean": baseline["real_time_ms_mean"],
            "latency_ms_median": baseline["real_time_ms_median"],
            "batch_iterations_per_second": baseline["batch_iterations_per_second"],
            "sequences_per_second": baseline["sequences_per_second"],
            "tokens_per_second": baseline["token_slots_per_second"],
        },
        "flatten_matmul": {
            "summary_file": str(ps["flatten_benchmark"].relative_to(ROOT)),
            "latency_ms_mean": flat["real_time_ms_mean"],
            "latency_ms_median": flat["real_time_ms_median"],
            "batch_iterations_per_second": flat["batch_iterations_per_second"],
            "sequences_per_second": flat["sequences_per_second"],
            "tokens_per_second": flat["token_slots_per_second"],
        },
        "speedup": compare["speedup"],
        "latency_reduction_percent": compare["latency_reduction_percent"],
    }
    compare["performance"] = performance
    ps["compare"].write_text(json.dumps(compare, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return compare


def summarize(shapes: list[str], args):
    rows = []
    for shape in shapes:
        ps = paths_for(shape, args)
        row = {
            "shape": ps["shape"],
            "batch": ps["batch"],
            "seq": ps["seq"],
            "token_slots": ps["token_slots"],
            "experiment": ps["experiment"],
            "baseline_vmfb_exists": ps["baseline_vmfb"].exists(),
            "flatten_vmfb_exists": ps["flatten_vmfb"].exists(),
        }
        if summary_succeeded(ps["baseline_benchmark"]):
            row["baseline"] = load_googlebench(ps["baseline_benchmark"])
        elif ps["baseline_log"].exists():
            row["baseline"] = {
                "status": "failed",
                **(benchmark_failure(ps["baseline_log"]) or {}),
            }
        else:
            row["baseline"] = {"status": "missing"}
        if summary_succeeded(ps["flatten_benchmark"]):
            row["flatten_matmul"] = load_googlebench(ps["flatten_benchmark"])
        elif ps["flatten_log"].exists():
            row["flatten_matmul"] = {
                "status": "failed",
                **(benchmark_failure(ps["flatten_log"]) or {}),
            }
        else:
            row["flatten_matmul"] = {"status": "missing"}
        compare = compare_if_possible(ps)
        if compare:
            row["compare"] = compare
        rows.append(row)

    summary = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "model": str(args.model_path),
            "model_label": args.model_label,
            "dtype": args.dtype,
            "shapes": shapes,
            "variants": args.variants,
        },
        "results": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved summary: {args.out}")


def run_shape(shape: str, args):
    ps = paths_for(shape, args)
    print(f"\n===== {ps['shape']} ({ps['token_slots']} token slots) =====")
    if args.action in ("plan", "compile", "all"):
        for variant in args.variants:
            compile_variant(ps, variant, args)
    if args.action in ("plan", "benchmark", "all"):
        for variant in args.variants:
            benchmark_variant(ps, variant, args)
    if args.action in ("benchmark", "all", "summarize"):
        compare_if_possible(ps)


def main():
    args = parse_args()
    args.model_path = args.model_path.resolve()
    shapes = [shape_id(*parse_shape(shape)) for shape in args.shapes]
    if args.action == "summarize":
        summarize(shapes, args)
        return
    for shape in shapes:
        run_shape(shape, args)
    if args.action in ("benchmark", "all"):
        summarize(shapes, args)


if __name__ == "__main__":
    main()
