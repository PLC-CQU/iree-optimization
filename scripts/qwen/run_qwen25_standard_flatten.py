#!/usr/bin/env python3
"""Export Qwen2.5 to ONNX, compile standard CUDA and flatten-MatMul variants."""

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
PYDEPS = ROOT / "pydeps"
REWRITE_SCRIPT = SCRIPTS_ROOT / "deepseek" / "rewrite_onnx_flatten_matmul.py"
INLINE_SCRIPT = SCRIPTS_ROOT / "deepseek" / "inline_onnx_dense_resources.py"

if PYDEPS.exists() and str(PYDEPS) not in sys.path:
    sys.path.insert(0, str(PYDEPS))

import torch
from transformers import AutoModelForCausalLM


class LastTokenWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        return outputs.logits[:, -1, :].float()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=ROOT / "Qwen2.5-3B")
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seq", type=int, default=32)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--export-device", default="cuda")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--cuda-target", default="sm_86")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--timeout-seconds", type=int, default=43200)
    parser.add_argument("--benchmark-repetitions", type=int, default=5)
    parser.add_argument("--benchmark-min-time", default="3s")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--force-export", action="store_true")
    parser.add_argument("--force-import", action="store_true")
    parser.add_argument("--force-compile", action="store_true")
    parser.add_argument("--force-rewrite", action="store_true")
    parser.add_argument(
        "--action",
        choices=["export", "rewrite", "import", "compile", "run", "benchmark", "all", "summarize"],
        default="all",
    )
    return parser.parse_args()


def dtype_from_name(name: str):
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def experiment_dir(args) -> Path:
    name = args.experiment_name or f"qwen25_3b_b{args.batch}_s{args.seq}_{args.dtype}_cuda_standard"
    return ROOT / "experiments" / name


def require_tool(name: str) -> str:
    tool = shutil.which(name)
    if not tool:
        raise RuntimeError(f"Missing required tool: {name}")
    return tool


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


def export_onnx(args, exp: Path) -> Path:
    out = exp / "qwen_last_token.onnx"
    if out.exists() and not args.force_export:
        print(f"Reusing ONNX: {out}")
        return out

    exp.mkdir(parents=True, exist_ok=True)
    print(f"Loading model from {args.model_path.resolve()}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype_from_name(args.dtype),
        low_cpu_mem_usage=True,
        trust_remote_code=args.trust_remote_code,
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
        "model_path": str(args.model_path.resolve()),
        "batch": args.batch,
        "seq": args.seq,
        "dtype": args.dtype,
        "opset": args.opset,
        "export_device": args.export_device,
        "exported": datetime.now().isoformat(),
    }
    (exp / "export_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return out


def rewrite_flatten(args, exp: Path, baseline: Path) -> tuple[Path, Path]:
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


def variant_paths(exp: Path, variant: str) -> dict[str, Path]:
    if variant == "baseline":
        build = exp / "standard_cuda"
        return {
            "build": build,
            "external_mlir": build / "qwen_external.mlir",
            "inlined_mlir": build / "qwen_external_inlined.mlir",
            "params": build / "qwen_params.irpa",
            "vmfb": build / "qwen_cuda.vmfb",
        }
    build = exp / "standard_cuda_flatten_matmul"
    return {
        "build": build,
        "external_mlir": build / "qwen_flatten_external.mlir",
        "inlined_mlir": build / "qwen_flatten_external_inlined.mlir",
        "params": build / "qwen_flatten_params.irpa",
        "vmfb": build / "qwen_flatten_cuda.vmfb",
    }


def import_onnx(args, exp: Path, onnx_path: Path, variant: str) -> tuple[Path, Path]:
    paths = variant_paths(exp, variant)
    if paths["external_mlir"].exists() and paths["params"].exists() and not args.force_import:
        print(f"Reusing imported MLIR: {paths['external_mlir']}")
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
                paths["params"],
                "-o",
                paths["external_mlir"],
            ],
            paths["build"] / "logs" / "import.log",
            args.timeout_seconds,
        )

    if paths["inlined_mlir"].exists() and not args.force_import:
        print(f"Reusing inlined MLIR: {paths['inlined_mlir']}")
    else:
        run_cmd(
            [sys.executable, INLINE_SCRIPT, paths["external_mlir"], "-o", paths["inlined_mlir"]],
            paths["build"] / "logs" / "inline.log",
            args.timeout_seconds,
        )
    return paths["inlined_mlir"], paths["params"]


def compile_vmfb(args, exp: Path, mlir_path: Path, variant: str) -> Path:
    paths = variant_paths(exp, variant)
    if paths["vmfb"].exists() and not args.force_compile:
        print(f"Reusing VMFB: {paths['vmfb']}")
        return paths["vmfb"]

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
            paths["vmfb"],
        ],
        paths["build"] / "logs" / "compile_cuda.log",
        args.timeout_seconds,
    )
    return paths["vmfb"]


def run_variant(args, exp: Path, variant: str):
    paths = variant_paths(exp, variant)
    run_cmd(
        [
            require_tool("iree-run-module"),
            f"--module={paths['vmfb']}",
            f"--parameters=model={paths['params']}",
            f"--device={args.device}",
            "--function=main_graph",
            f"--input={args.batch}x{args.seq}xi64=0",
            f"--input={args.batch}x{args.seq}xi64=1",
        ],
        paths["build"] / "logs" / "run_cuda.log",
        args.timeout_seconds,
    )


def benchmark_variant(args, exp: Path, variant: str):
    paths = variant_paths(exp, variant)
    run_cmd(
        [
            require_tool("iree-benchmark-module"),
            f"--module={paths['vmfb']}",
            f"--parameters=model={paths['params']}",
            f"--device={args.device}",
            "--function=main_graph",
            f"--input={args.batch}x{args.seq}xi64=0",
            f"--input={args.batch}x{args.seq}xi64=1",
            f"--benchmark_repetitions={args.benchmark_repetitions}",
            f"--benchmark_min_time={args.benchmark_min_time}",
        ],
        paths["build"] / "logs" / "benchmark_cuda.log",
        args.timeout_seconds,
    )


def summarize(args, exp: Path):
    report_path = exp / "flatten_matmul_rewrite_report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        print("Rewrite summary:")
        print(
            json.dumps(
                {
                    "matmul_total": report.get("matmul_total"),
                    "weight_matmul_total": report.get("weight_matmul_total"),
                    "rewritten_weight_matmul_nodes": report.get("rewritten_weight_matmul_nodes"),
                    "skipped_matmul_by_reason": report.get("skipped_matmul_by_reason"),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
    for variant in ["baseline", "flatten_matmul"]:
        paths = variant_paths(exp, variant)
        size = paths["vmfb"].stat().st_size if paths["vmfb"].exists() else 0
        print(f"{variant} vmfb: {paths['vmfb']} exists={paths['vmfb'].exists()} size={size}")


def main():
    args = parse_args()
    args.model_path = args.model_path.resolve()
    exp = experiment_dir(args)

    baseline = exp / "qwen_last_token.onnx"
    flattened = exp / "flatten_matmul_last_token.onnx"

    if args.action in {"export", "rewrite", "import", "compile", "run", "benchmark", "all"}:
        baseline = export_onnx(args, exp) if args.action == "export" or not baseline.exists() else baseline
    if args.action in {"rewrite", "import", "compile", "run", "benchmark", "all"}:
        flattened, _ = rewrite_flatten(args, exp, baseline)
    if args.action in {"import", "compile", "run", "benchmark", "all"}:
        for variant, onnx_path in [("baseline", baseline), ("flatten_matmul", flattened)]:
            mlir_path, _ = import_onnx(args, exp, onnx_path, variant)
            if args.action in {"compile", "run", "benchmark", "all"}:
                compile_vmfb(args, exp, mlir_path, variant)
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
