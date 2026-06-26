#!/usr/bin/env python3
"""Dynamic-shape Gemma experiment for testing IREE flatten-rank3-matmul.

This script exports one Gemma ONNX with dynamic batch/sequence dimensions,
imports it once to MLIR/external parameters, then compiles two VMFBs:

  * baseline_dynamic: compiler without the local flatten pass.
  * optimized_dynamic: current source build with the flatten pass in pipeline.

The default baseline compiler is the pip/venv IREE compiler because the local
source build already includes the new pass and currently has no disable flag.
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


ROOT = Path(__file__).resolve().parent
PROJECTS = ROOT.parents[1]
INLINE_SCRIPT = PROJECTS / "iree-optimization" / "scripts" / "deepseek" / "inline_onnx_dense_resources.py"
QWEN_PYDEPS = PROJECTS / "iree" / "Qwen" / "pydeps"

if QWEN_PYDEPS.exists() and str(QWEN_PYDEPS) not in sys.path:
    sys.path.insert(0, str(QWEN_PYDEPS))

import numpy as np
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


def parse_shape(text: str) -> tuple[int, int]:
    try:
        b, s = text.lower().removeprefix("b").split("_s", 1)
        return int(b), int(s)
    except Exception as exc:
        raise argparse.ArgumentTypeError(f"invalid shape '{text}', expected b1_s32") from exc


def dtype_from_name(name: str):
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def require_tool(path_or_name: str) -> str:
    path = Path(path_or_name)
    if path.exists():
        return str(path)
    tool = shutil.which(path_or_name)
    if not tool:
        raise RuntimeError(f"missing required tool: {path_or_name}")
    return tool


def run_cmd(
    cmd: list[object],
    log_path: Path,
    timeout_seconds: int,
    env: dict[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess:
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
    if check and result.returncode != 0:
        raise RuntimeError(f"command failed with code {result.returncode}; see {log_path}")
    return result


def load_model(args):
    kwargs = {
        "torch_dtype": dtype_from_name(args.dtype),
        "low_cpu_mem_usage": True,
        "trust_remote_code": args.trust_remote_code,
        "attn_implementation": "eager",
    }
    try:
        return AutoModelForCausalLM.from_pretrained(args.model_path, **kwargs)
    except ValueError:
        from transformers import AutoModelForImageTextToText

        return AutoModelForImageTextToText.from_pretrained(args.model_path, **kwargs)
    except TypeError:
        kwargs.pop("attn_implementation", None)
        return AutoModelForCausalLM.from_pretrained(args.model_path, **kwargs)


def export_dynamic_onnx(args, exp: Path) -> Path:
    out = exp / "gemma_dynamic_last_token.onnx"
    if out.exists() and not args.force_export:
        print(f"Reusing dynamic ONNX: {out}")
        return out

    exp.mkdir(parents=True, exist_ok=True)
    print(f"Loading model from {args.model_path}")
    model = load_model(args)
    model.config.use_cache = False
    model.eval()

    export_device = torch.device(args.export_device)
    if export_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA export requested, but torch.cuda.is_available() is false")
    wrapper = LastTokenWrapper(model).eval().to(export_device)

    input_ids = torch.zeros((args.export_batch, args.export_seq), dtype=torch.long, device=export_device)
    attention_mask = torch.ones((args.export_batch, args.export_seq), dtype=torch.long, device=export_device)

    print(f"Exporting dynamic B/S ONNX: {out}")
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
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq"},
                "attention_mask": {0: "batch", 1: "seq"},
                "logits": {0: "batch"},
            },
        )

    metadata = {
        "model_path": str(args.model_path),
        "dtype": args.dtype,
        "opset": args.opset,
        "export_batch": args.export_batch,
        "export_seq": args.export_seq,
        "dynamic_axes": ["batch", "seq"],
        "exported": datetime.now().isoformat(),
    }
    (exp / "export_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return out


def import_onnx(args, exp: Path, onnx_path: Path) -> tuple[Path, Path]:
    build = exp / "imported"
    external_mlir = build / "gemma_dynamic_external.mlir"
    inlined_mlir = build / "gemma_dynamic_external_inlined.mlir"
    params = build / "gemma_dynamic_params.irpa"

    if external_mlir.exists() and params.exists() and not args.force_import:
        print(f"Reusing imported MLIR: {external_mlir}")
    else:
        run_cmd(
            [
                require_tool(args.iree_import_onnx),
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


def compile_variant(args, exp: Path, mlir_path: Path, variant: str) -> Path:
    build = exp / variant
    vmfb = build / f"gemma_{variant}.vmfb"
    if vmfb.exists() and not args.force_compile:
        print(f"Reusing {variant} VMFB: {vmfb}")
        return vmfb

    compiler = args.baseline_iree_compile if variant == "baseline_dynamic" else args.optimized_iree_compile
    cmd = [
        require_tool(compiler),
        "--iree-input-type=onnx",
        "--iree-input-demote-i64-to-i32",
        "--iree-opt-strip-assertions",
        "--iree-hal-target-backends=cuda",
        f"--iree-cuda-target={args.cuda_target}",
        "--iree-codegen-llvmgpu-use-reduction-vector-distribution=false",
        mlir_path,
        "-o",
        vmfb,
    ]
    run_cmd(cmd, build / "logs" / "compile.log", args.timeout_seconds)
    return vmfb


def benchmark_variant(args, exp: Path, vmfb: Path, params: Path, variant: str, batch: int, seq: int) -> dict:
    out = exp / variant / f"benchmark_b{batch}_s{seq}.googlebench.json"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    result = run_cmd(
        [
            require_tool(args.iree_benchmark_module),
            f"--module={vmfb}",
            f"--parameters=model={params}",
            f"--device={args.device}",
            "--function=main_graph",
            f"--input={batch}x{seq}xi64=0",
            f"--input={batch}x{seq}xi64=1",
            f"--benchmark_repetitions={args.repetitions}",
            f"--benchmark_min_time={args.min_time}",
            f"--benchmark_min_warmup_time={args.warmup_time}",
            "--benchmark_time_unit=ms",
            f"--benchmark_out={out}",
            "--benchmark_out_format=json",
        ],
        exp / variant / "logs" / f"benchmark_b{batch}_s{seq}.log",
        args.benchmark_timeout_seconds,
        env=env,
        check=False,
    )
    return {
        "variant": variant,
        "shape": f"b{batch}_s{seq}",
        "returncode": result.returncode,
        "json": str(out),
        "mean_ms": read_mean_ms(out) if result.returncode == 0 else None,
    }


def read_mean_ms(path: Path) -> float | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    for bench in data.get("benchmarks", []):
        if bench.get("run_type") == "aggregate" and bench.get("aggregate_name") == "mean":
            return float(bench["real_time"])
    for bench in data.get("benchmarks", []):
        if bench.get("name", "").endswith("/real_time"):
            return float(bench["real_time"])
    return None


def run_output(args, exp: Path, vmfb: Path, params: Path, variant: str, batch: int, seq: int) -> Path:
    out = exp / variant / f"output_b{batch}_s{seq}.npy"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    run_cmd(
        [
            require_tool(args.iree_run_module),
            f"--module={vmfb}",
            f"--parameters=model={params}",
            f"--device={args.device}",
            "--function=main_graph",
            f"--input={batch}x{seq}xi64=0",
            f"--input={batch}x{seq}xi64=1",
            f"--output=@{out}",
        ],
        exp / variant / "logs" / f"run_b{batch}_s{seq}.log",
        args.benchmark_timeout_seconds,
        env=env,
    )
    return out


def compare_outputs(a_path: Path, b_path: Path) -> dict:
    a = np.load(a_path)
    b = np.load(b_path)
    diff = np.abs(a - b)
    return {
        "shape_a": list(a.shape),
        "shape_b": list(b.shape),
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "allclose_atol_0.06_rtol_0.01": bool(np.allclose(a, b, atol=0.06, rtol=0.01)),
    }


def summarize(exp: Path, benchmark_rows: list[dict], correctness_rows: list[dict]) -> Path:
    rows = {}
    for row in benchmark_rows:
        rows.setdefault(row["shape"], {})[row["variant"]] = row

    summary_rows = []
    for shape, variants in sorted(rows.items()):
        base = variants.get("baseline_dynamic", {}).get("mean_ms")
        opt = variants.get("optimized_dynamic", {}).get("mean_ms")
        speedup = base / opt if base and opt else None
        summary_rows.append(
            {
                "shape": shape,
                "baseline_dynamic_ms": base,
                "optimized_dynamic_ms": opt,
                "speedup": speedup,
                "baseline_returncode": variants.get("baseline_dynamic", {}).get("returncode"),
                "optimized_returncode": variants.get("optimized_dynamic", {}).get("returncode"),
            }
        )

    out = exp / "dynamic_shape_summary.json"
    out.write_text(
        json.dumps(
            {
                "generated": datetime.now().isoformat(),
                "summary": summary_rows,
                "benchmarks": benchmark_rows,
                "correctness": correctness_rows,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(json.dumps(json.loads(out.read_text(encoding="utf-8")), indent=2, ensure_ascii=False))
    return out


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=ROOT / "googlegemma-4-E2B-it")
    parser.add_argument("--experiment-name", default="gemma_e2b_dynamic_bs_fp16_cuda")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--export-device", default="cuda")
    parser.add_argument("--export-batch", type=int, default=1)
    parser.add_argument("--export-seq", type=int, default=32)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--cuda-target", default="sm_86")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--shapes", nargs="+", default=["b1_s32", "b1_s64", "b4_s32"])
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--min-time", default="1x")
    parser.add_argument("--warmup-time", default="0")
    parser.add_argument("--timeout-seconds", type=int, default=43200)
    parser.add_argument("--benchmark-timeout-seconds", type=int, default=300)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--force-export", action="store_true")
    parser.add_argument("--force-import", action="store_true")
    parser.add_argument("--force-compile", action="store_true")
    parser.add_argument(
        "--action",
        choices=["export", "import", "compile", "benchmark", "correctness", "all", "summarize"],
        default="all",
    )
    parser.add_argument("--iree-import-onnx", default=str(PROJECTS / ".venv" / "bin" / "iree-import-onnx"))
    parser.add_argument("--baseline-iree-compile", default=str(PROJECTS / ".venv" / "bin" / "iree-compile"))
    parser.add_argument("--optimized-iree-compile", default=str(PROJECTS / "iree-build" / "tools" / "iree-compile"))
    parser.add_argument("--iree-benchmark-module", default=str(PROJECTS / "iree-build" / "tools" / "iree-benchmark-module"))
    parser.add_argument("--iree-run-module", default=str(PROJECTS / "iree-build" / "tools" / "iree-run-module"))
    return parser.parse_args()


def main():
    args = parse_args()
    args.model_path = args.model_path.resolve()
    exp = ROOT / "dynamic_experiments" / args.experiment_name
    shapes = [parse_shape(shape) for shape in args.shapes]

    onnx_path = exp / "gemma_dynamic_last_token.onnx"
    mlir_path = exp / "imported" / "gemma_dynamic_external_inlined.mlir"
    params = exp / "imported" / "gemma_dynamic_params.irpa"
    baseline_vmfb = exp / "baseline_dynamic" / "gemma_baseline_dynamic.vmfb"
    optimized_vmfb = exp / "optimized_dynamic" / "gemma_optimized_dynamic.vmfb"

    if args.action in {"export", "import", "compile", "benchmark", "correctness", "all"}:
        onnx_path = export_dynamic_onnx(args, exp)
    if args.action in {"import", "compile", "benchmark", "correctness", "all"}:
        mlir_path, params = import_onnx(args, exp, onnx_path)
    if args.action in {"compile", "benchmark", "correctness", "all"}:
        baseline_vmfb = compile_variant(args, exp, mlir_path, "baseline_dynamic")
        optimized_vmfb = compile_variant(args, exp, mlir_path, "optimized_dynamic")

    benchmark_rows: list[dict] = []
    correctness_rows: list[dict] = []
    if args.action in {"benchmark", "all"}:
        for batch, seq in shapes:
            benchmark_rows.append(benchmark_variant(args, exp, baseline_vmfb, params, "baseline_dynamic", batch, seq))
            benchmark_rows.append(benchmark_variant(args, exp, optimized_vmfb, params, "optimized_dynamic", batch, seq))

    if args.action in {"correctness", "all"}:
        for batch, seq in shapes:
            base_out = run_output(args, exp, baseline_vmfb, params, "baseline_dynamic", batch, seq)
            opt_out = run_output(args, exp, optimized_vmfb, params, "optimized_dynamic", batch, seq)
            correctness_rows.append(
                {
                    "shape": f"b{batch}_s{seq}",
                    "baseline_dynamic_vs_optimized_dynamic": compare_outputs(base_out, opt_out),
                }
            )

    if args.action in {"benchmark", "correctness", "all", "summarize"}:
        if not benchmark_rows and (exp / "dynamic_shape_summary.json").exists():
            print((exp / "dynamic_shape_summary.json").read_text(encoding="utf-8"))
        else:
            summarize(exp, benchmark_rows, correctness_rows)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
