#!/usr/bin/env python3
"""Export, rewrite, compile, and benchmark the uploaded Qwen3.5 model.

This runner keeps all generated artifacts in ./experiments and uses the
isolated Python dependencies installed in ./pydeps.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from statistics import median


ROOT = Path(__file__).resolve().parent
PYDEPS = ROOT / "pydeps"
DEEPSEEK_DIR = ROOT.parent / "deepseek"
REWRITE_SCRIPT = DEEPSEEK_DIR / "rewrite_onnx_flatten_matmul.py"
INLINE_SCRIPT = DEEPSEEK_DIR / "inline_onnx_dense_resources.py"

if str(PYDEPS) not in sys.path:
    sys.path.insert(0, str(PYDEPS))

import torch
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration


class LastTokenWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=False,
        )
        return outputs[0][:, -1, :].float()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=ROOT / "Qwen3.5-4B")
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq", type=int, default=16)
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument(
        "--export-device",
        default="cuda",
        help="Device used for PyTorch ONNX export, e.g. cuda or cuda:0.",
    )
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--backend", choices=["llvm-cpu", "cuda"], default="llvm-cpu")
    parser.add_argument("--device", default=None)
    parser.add_argument("--cuda-target", default="sm_86")
    parser.add_argument("--no-ptxas", action="store_true")
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=["baseline", "flatten_matmul"],
        default=["baseline", "flatten_matmul"],
    )
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--min-time", default="3x")
    parser.add_argument("--warmup-time", default="0.2")
    parser.add_argument("--timeout-seconds", type=int, default=43200)
    parser.add_argument(
        "--action",
        choices=["export", "rewrite", "compile", "benchmark", "all", "summarize"],
        default="all",
    )
    parser.add_argument("--force-export", action="store_true")
    parser.add_argument("--force-import", action="store_true")
    parser.add_argument("--force-compile", action="store_true")
    parser.add_argument("--force-benchmark", action="store_true")
    return parser.parse_args()


def dtype_from_name(name: str):
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def require_tool(name: str) -> str:
    tool = shutil.which(name)
    if not tool:
        raise RuntimeError(f"Missing required tool: {name}")
    return tool


def experiment_dir(args) -> Path:
    name = args.experiment_name or f"qwen35_4b_b{args.batch}_s{args.seq}_{args.backend}"
    return ROOT / "experiments" / name


def run(cmd: list[object], log_path: Path, timeout_seconds: int):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("Command:")
    print("  " + " \\\n  ".join(shlex.quote(str(part)) for part in cmd))
    print(f"Log: {log_path}")
    started = datetime.now()
    result = subprocess.run(
        [str(part) for part in cmd],
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
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
    if result.returncode != 0:
        raise RuntimeError(f"command failed with code {result.returncode}; see {log_path}")


def export_onnx(args, exp: Path) -> Path:
    out = exp / "baseline_last_token.onnx"
    if out.exists() and not args.force_export:
        print(f"Reusing ONNX: {out}")
        return out

    exp.mkdir(parents=True, exist_ok=True)
    print(f"Loading model from {args.model_path}")
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=dtype_from_name(args.dtype),
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    model.config.use_cache = False
    model.eval()
    export_device = torch.device(args.export_device)
    if export_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA export requested, but torch.cuda.is_available() is false.")
    wrapper = LastTokenWrapper(model).eval().to(export_device)

    input_ids = torch.zeros((args.batch, args.seq), dtype=torch.long, device=export_device)
    attention_mask = torch.ones((args.batch, args.seq), dtype=torch.long, device=export_device)
    print(f"Exporting fixed-shape ONNX: {out}")
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (input_ids, attention_mask),
            str(out),
            export_params=True,
            opset_version=args.opset,
            do_constant_folding=True,
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={},
        )

    metadata = {
        "model_path": str(args.model_path),
        "batch": args.batch,
        "seq": args.seq,
        "dtype": args.dtype,
        "opset": args.opset,
        "export_device": args.export_device,
        "exported": datetime.now().isoformat(),
    }
    (exp / "export_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return out


def rewrite_onnx(args, exp: Path, baseline: Path) -> tuple[Path, Path]:
    out = exp / "flatten_matmul_last_token.onnx"
    report = exp / "rewrite_report.json"
    if out.exists() and report.exists() and not args.force_export:
        print(f"Reusing flattened ONNX: {out}")
        return out, report
    run(
        [
            sys.executable,
            REWRITE_SCRIPT,
            "--input",
            baseline,
            "--output",
            out,
            "--batch",
            args.batch,
            "--seq",
            args.seq,
            "--check",
            "--report",
            report,
        ],
        exp / "logs" / "rewrite.log",
        args.timeout_seconds,
    )
    return out, report


def import_and_inline(args, exp: Path, onnx_path: Path, variant: str) -> tuple[Path, Path]:
    build = exp / f"build_{variant}_{args.backend}"
    external_mlir = build / f"{variant}_external.mlir"
    inlined_mlir = build / f"{variant}_external_inlined.mlir"
    params = build / f"{variant}_params.irpa"
    if external_mlir.exists() and params.exists() and not args.force_import:
        print(f"Reusing imported MLIR: {external_mlir}")
    else:
        run(
            [
                require_tool("iree-import-onnx"),
                onnx_path,
                "--large-model",
                "--externalize-params",
                "--num-elements-threshold",
                "2",
                "--param-gb-threshold",
                "2",
                "--save-params-to",
                params,
                "-o",
                external_mlir,
            ],
            build / "logs" / "import.log",
            args.timeout_seconds,
        )
    if inlined_mlir.exists() and not args.force_import:
        print(f"Reusing inlined MLIR: {inlined_mlir}")
    else:
        run(
            [sys.executable, INLINE_SCRIPT, external_mlir, "-o", inlined_mlir],
            build / "logs" / "inline.log",
            args.timeout_seconds,
        )
    return inlined_mlir, params


def compile_vmfb(args, exp: Path, inlined_mlir: Path, params: Path, variant: str) -> Path:
    out = exp / f"{variant}_{args.backend}.vmfb"
    if out.exists() and not args.force_compile:
        print(f"Reusing VMFB: {out}")
        return out

    flags = [
        "--iree-input-type=onnx",
        f"--iree-parameter-import={params}",
        "--iree-input-demote-i64-to-i32",
        "--iree-opt-strip-assertions",
    ]
    if args.backend == "llvm-cpu":
        flags.append("--iree-hal-target-backends=llvm-cpu")
    else:
        flags.extend(
            [
                "--iree-hal-target-backends=cuda",
                f"--iree-cuda-target={args.cuda_target}",
            ]
        )
        if not args.no_ptxas:
            flags.append("--iree-cuda-use-ptxas")

    run(
        [require_tool("iree-compile"), *flags, inlined_mlir, "-o", out],
        exp / "logs" / f"compile_{variant}_{args.backend}.log",
        args.timeout_seconds,
    )
    return out


def compile_variants(args, exp: Path, baseline: Path, flattened: Path) -> dict:
    result = {}
    for variant, onnx_path in [("baseline", baseline), ("flatten_matmul", flattened)]:
        if variant not in args.variants:
            continue
        inlined_mlir, params = import_and_inline(args, exp, onnx_path, variant)
        vmfb = compile_vmfb(args, exp, inlined_mlir, params, variant)
        result[variant] = {"onnx": str(onnx_path), "params": str(params), "vmfb": str(vmfb)}
    return result


def benchmark_variant(args, exp: Path, variant: str, vmfb: Path, params: Path) -> Path:
    out = exp / f"benchmark_{variant}_{args.backend}.json"
    raw = exp / f"benchmark_{variant}_{args.backend}.googlebench.json"
    if out.exists() and not args.force_benchmark:
        print(f"Reusing benchmark: {out}")
        return out
    device = args.device or ("cuda" if args.backend == "cuda" else "local-task")
    cmd = [
        require_tool("iree-benchmark-module"),
        f"--module={vmfb}",
        f"--parameters=model={params}",
        "--parameter_mode=file",
        f"--device={device}",
        "--function=main_graph",
        f"--input={args.batch}x{args.seq}xi64=0",
        f"--input={args.batch}x{args.seq}xi64=1",
        f"--benchmark_repetitions={args.repetitions}",
        f"--benchmark_min_time={args.min_time}",
        f"--benchmark_min_warmup_time={args.warmup_time}",
        "--benchmark_format=console",
        f"--benchmark_out={raw}",
        "--benchmark_out_format=json",
        "--benchmark_time_unit=ms",
    ]
    started = datetime.now()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout_seconds)
    summary = {
        "started": started.isoformat(),
        "finished": datetime.now().isoformat(),
        "returncode": result.returncode,
        "command": cmd,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "google_benchmark_json": str(raw),
    }
    if raw.exists():
        summary["google_benchmark"] = json.loads(raw.read_text(encoding="utf-8"))
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"benchmark failed with code {result.returncode}; see {out}")
    return out


def median_ms(path: Path) -> float | None:
    data = json.loads(path.read_text(encoding="utf-8"))
    values = [
        float(item["real_time"])
        for item in data.get("google_benchmark", {}).get("benchmarks", [])
        if item.get("run_type") == "iteration" and "real_time" in item
    ]
    return median(values) if values else None


def summarize(args, exp: Path) -> Path:
    report_path = exp / "rewrite_report.json"
    rewrite_report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    variants = {}
    for variant in ["baseline", "flatten_matmul"]:
        bench = exp / f"benchmark_{variant}_{args.backend}.json"
        variants[variant] = {
            "vmfb": str(exp / f"{variant}_{args.backend}.vmfb"),
            "benchmark": str(bench) if bench.exists() else None,
            "median_ms": median_ms(bench) if bench.exists() else None,
        }
    base = variants["baseline"]["median_ms"]
    flat = variants["flatten_matmul"]["median_ms"]
    summary = {
        "experiment": str(exp),
        "model_path": str(args.model_path),
        "shape": {"batch": args.batch, "seq": args.seq},
        "backend": args.backend,
        "device": args.device,
        "rewrite": {
            "matmul_total": rewrite_report.get("matmul_total"),
            "weight_matmul_total": rewrite_report.get("weight_matmul_total"),
            "rewritten_weight_matmul_nodes": rewrite_report.get("rewritten_weight_matmul_nodes"),
            "skipped_matmul_by_reason": rewrite_report.get("skipped_matmul_by_reason"),
        },
        "variants": variants,
    }
    if base and flat:
        summary["speedup"] = base / flat
        summary["latency_delta_pct"] = (flat - base) / base * 100.0
    out = exp / "summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved summary: {out}")
    if "speedup" in summary:
        print(f"baseline={base:.4f} ms flatten={flat:.4f} ms speedup={summary['speedup']:.3f}x")
    print(f"rewritten={summary['rewrite']['rewritten_weight_matmul_nodes']}")
    return out


def main():
    args = parse_args()
    args.model_path = args.model_path.resolve()
    exp = experiment_dir(args)
    baseline = exp / "baseline_last_token.onnx"
    flattened = exp / "flatten_matmul_last_token.onnx"

    if args.action in ("export", "all"):
        baseline = export_onnx(args, exp)
    if args.action in ("rewrite", "all"):
        baseline = baseline if baseline.exists() else export_onnx(args, exp)
        flattened, _ = rewrite_onnx(args, exp, baseline)
    if args.action in ("compile", "all"):
        baseline = baseline if baseline.exists() else export_onnx(args, exp)
        if not flattened.exists():
            flattened, _ = rewrite_onnx(args, exp, baseline)
        compile_variants(args, exp, baseline, flattened)
    if args.action in ("benchmark", "all"):
        for variant in ["baseline", "flatten_matmul"]:
            vmfb = exp / f"{variant}_{args.backend}.vmfb"
            params = exp / f"build_{variant}_{args.backend}" / f"{variant}_params.irpa"
            benchmark_variant(args, exp, variant, vmfb, params)
    if args.action in ("summarize", "all"):
        summarize(args, exp)


if __name__ == "__main__":
    main()
