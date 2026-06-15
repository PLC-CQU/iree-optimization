#!/usr/bin/env python3
"""Apply the flatten-MatMul ONNX rewrite to Gemma and compare with baseline."""

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


ROOT = Path(__file__).resolve().parent
SCRIPTS_ROOT = ROOT.parent
STANDARD_SCRIPT = ROOT / "run_gemma_standard_iree.py"
REWRITE_SCRIPT = SCRIPTS_ROOT / "deepseek" / "rewrite_onnx_flatten_matmul.py"
INLINE_SCRIPT = SCRIPTS_ROOT / "deepseek" / "inline_onnx_dense_resources.py"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=ROOT / "googlegemma-4-E4B-it")
    parser.add_argument("--experiment-name", default="gemma_e4b_it_b4_s32_fp16_cuda_standard")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seq", type=int, default=32)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--export-device", default="cuda")
    parser.add_argument("--cuda-target", default="sm_86")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--timeout-seconds", type=int, default=43200)
    parser.add_argument("--benchmark-repetitions", type=int, default=5)
    parser.add_argument("--benchmark-min-time", default="3s")
    parser.add_argument(
        "--action",
        choices=["rewrite", "import", "compile", "run", "benchmark", "all", "summarize"],
        default="all",
    )
    parser.add_argument("--force-rewrite", action="store_true")
    parser.add_argument("--force-import", action="store_true")
    parser.add_argument("--force-compile", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def require_tool(name: str) -> str:
    tool = shutil.which(name)
    if not tool:
        raise RuntimeError(f"Missing required tool: {name}")
    return tool


def exp_dir(args) -> Path:
    return ROOT / "experiments" / args.experiment_name


def run_cmd(cmd: list[object], log_path: Path, timeout_seconds: int):
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


def ensure_baseline_onnx(args, exp: Path) -> Path:
    onnx_path = exp / "gemma_last_token.onnx"
    if onnx_path.exists():
        return onnx_path
    cmd = [
        sys.executable,
        STANDARD_SCRIPT,
        "--action",
        "export",
        "--model-path",
        args.model_path,
        "--experiment-name",
        args.experiment_name,
        "--batch",
        args.batch,
        "--seq",
        args.seq,
        "--dtype",
        args.dtype,
        "--export-device",
        args.export_device,
    ]
    if args.trust_remote_code:
        cmd.append("--trust-remote-code")
    run_cmd(cmd, exp / "logs" / "baseline_export.log", args.timeout_seconds)
    return onnx_path


def rewrite_flatten(args, exp: Path) -> tuple[Path, Path]:
    baseline = ensure_baseline_onnx(args, exp)
    out = exp / "flatten_matmul_last_token.onnx"
    report = exp / "flatten_matmul_rewrite_report.json"
    if out.exists() and report.exists() and not args.force_rewrite:
        print(f"Reusing flattened ONNX: {out}")
        return out, report
    run_cmd(
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
        exp / "logs" / "flatten_rewrite.log",
        args.timeout_seconds,
    )
    return out, report


def import_flatten(args, exp: Path, onnx_path: Path) -> tuple[Path, Path]:
    build = exp / "standard_cuda_flatten_matmul"
    external_mlir = build / "gemma_flatten_external.mlir"
    inlined_mlir = build / "gemma_flatten_external_inlined.mlir"
    params = build / "gemma_flatten_params.irpa"

    if external_mlir.exists() and params.exists() and not args.force_import:
        print(f"Reusing imported MLIR: {external_mlir}")
    else:
        run_cmd(
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
        run_cmd(
            [sys.executable, INLINE_SCRIPT, external_mlir, "-o", inlined_mlir],
            build / "logs" / "inline.log",
            args.timeout_seconds,
        )
    return inlined_mlir, params


def compile_flatten(args, exp: Path, mlir_path: Path) -> Path:
    build = exp / "standard_cuda_flatten_matmul"
    vmfb = build / "gemma_flatten_cuda.vmfb"
    if vmfb.exists() and not args.force_compile:
        print(f"Reusing VMFB: {vmfb}")
        return vmfb
    run_cmd(
        [
            require_tool("iree-compile"),
            "--iree-input-type=onnx",
            "--iree-input-demote-i64-to-i32",
            "--iree-opt-strip-assertions",
            "--iree-hal-target-backends=cuda",
            f"--iree-cuda-target={args.cuda_target}",
            "--iree-codegen-llvmgpu-use-reduction-vector-distribution=false",
            mlir_path,
            "-o",
            vmfb,
        ],
        build / "logs" / "compile_cuda.log",
        args.timeout_seconds,
    )
    return vmfb


def run_variant(args, exp: Path, variant: str):
    if variant == "baseline":
        build = exp / "standard_cuda"
        vmfb = build / "gemma_cuda.vmfb"
        params = build / "gemma_params.irpa"
    else:
        build = exp / "standard_cuda_flatten_matmul"
        vmfb = build / "gemma_flatten_cuda.vmfb"
        params = build / "gemma_flatten_params.irpa"
    run_cmd(
        [
            require_tool("iree-run-module"),
            f"--module={vmfb}",
            f"--parameters=model={params}",
            f"--device={args.device}",
            "--function=main_graph",
            f"--input={args.batch}x{args.seq}xi64=0",
            f"--input={args.batch}x{args.seq}xi64=1",
        ],
        build / "logs" / "run_cuda.log",
        args.timeout_seconds,
    )


def benchmark_variant(args, exp: Path, variant: str):
    if variant == "baseline":
        build = exp / "standard_cuda"
        vmfb = build / "gemma_cuda.vmfb"
        params = build / "gemma_params.irpa"
    else:
        build = exp / "standard_cuda_flatten_matmul"
        vmfb = build / "gemma_flatten_cuda.vmfb"
        params = build / "gemma_flatten_params.irpa"
    run_cmd(
        [
            require_tool("iree-benchmark-module"),
            f"--module={vmfb}",
            f"--parameters=model={params}",
            f"--device={args.device}",
            "--function=main_graph",
            f"--input={args.batch}x{args.seq}xi64=0",
            f"--input={args.batch}x{args.seq}xi64=1",
            f"--benchmark_repetitions={args.benchmark_repetitions}",
            f"--benchmark_min_time={args.benchmark_min_time}",
        ],
        build / "logs" / "benchmark_cuda.log",
        args.timeout_seconds,
    )


def summarize(args, exp: Path):
    report_path = exp / "flatten_matmul_rewrite_report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        print("Rewrite summary:")
        print(json.dumps({
            "matmul_total": report.get("matmul_total"),
            "weight_matmul_total": report.get("weight_matmul_total"),
            "rewritten_weight_matmul_nodes": report.get("rewritten_weight_matmul_nodes"),
            "skipped_matmul_by_reason": report.get("skipped_matmul_by_reason"),
        }, indent=2, ensure_ascii=False))
    for variant, rel in [
        ("baseline", "standard_cuda/gemma_cuda.vmfb"),
        ("flatten_matmul", "standard_cuda_flatten_matmul/gemma_flatten_cuda.vmfb"),
    ]:
        vmfb = exp / rel
        print(f"{variant} vmfb: {vmfb} exists={vmfb.exists()} size={vmfb.stat().st_size if vmfb.exists() else 0}")


def main():
    args = parse_args()
    args.model_path = args.model_path.resolve()
    exp = exp_dir(args)

    if args.action in {"rewrite", "import", "compile", "run", "benchmark", "all"}:
        flattened_onnx, _ = rewrite_flatten(args, exp)
    if args.action in {"import", "compile", "run", "benchmark", "all"}:
        mlir_path, _params = import_flatten(args, exp, flattened_onnx)
    if args.action in {"compile", "run", "benchmark", "all"}:
        compile_flatten(args, exp, mlir_path)
    if args.action in {"run", "all"}:
        run_variant(args, exp, "baseline")
        run_variant(args, exp, "flatten_matmul")
    if args.action in {"benchmark", "all"}:
        benchmark_variant(args, exp, "baseline")
        benchmark_variant(args, exp, "flatten_matmul")
    if args.action in {"summarize", "all"}:
        summarize(args, exp)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
