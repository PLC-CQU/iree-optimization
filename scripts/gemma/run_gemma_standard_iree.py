#!/usr/bin/env python3
"""Export and compile a Gemma model through the standard IREE ONNX path.

Pipeline:
  PyTorch/HuggingFace -> ONNX -> iree-import-onnx external params ->
  iree-compile CUDA -> iree-run-module.
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
SCRIPTS_ROOT = ROOT.parent
QWEN_PYDEPS = SCRIPTS_ROOT / "qwen" / "pydeps"
INLINE_SCRIPT = SCRIPTS_ROOT / "deepseek" / "inline_onnx_dense_resources.py"

if QWEN_PYDEPS.exists() and str(QWEN_PYDEPS) not in sys.path:
    sys.path.insert(0, str(QWEN_PYDEPS))

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
    parser.add_argument("--model-path", type=Path, default=ROOT / "googlegemma-4-E4B-it")
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seq", type=int, default=32)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--export-device", default="cuda")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--cuda-target", default="sm_86")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--timeout-seconds", type=int, default=43200)
    parser.add_argument(
        "--action",
        choices=["export", "import", "compile", "run", "all"],
        default="all",
    )
    parser.add_argument("--force-export", action="store_true")
    parser.add_argument("--force-import", action="store_true")
    parser.add_argument("--force-compile", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--compile-with-param-import",
        action="store_true",
        help="Pass --iree-parameter-import to iree-compile. Default keeps params external.",
    )
    parser.add_argument(
        "--enable-reduction-vector-distribution",
        action="store_true",
        help="Use IREE's reduction vector distribution path. Disabled by default for this Gemma export.",
    )
    return parser.parse_args()


def dtype_from_name(name: str):
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def require_tool(name: str) -> str:
    tool = shutil.which(name)
    if not tool:
        raise RuntimeError(f"Missing required tool: {name}")
    return tool


def experiment_dir(args) -> Path:
    name = args.experiment_name or f"gemma_e4b_it_b{args.batch}_s{args.seq}_{args.dtype}_cuda_standard"
    return ROOT / "experiments" / name


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


def export_onnx(args, exp: Path) -> Path:
    out = exp / "gemma_last_token.onnx"
    if out.exists() and not args.force_export:
        print(f"Reusing ONNX: {out}")
        return out

    exp.mkdir(parents=True, exist_ok=True)
    print(f"Loading model from {args.model_path.resolve()}")
    model = load_model(args)
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


def import_onnx(args, exp: Path, onnx_path: Path) -> tuple[Path, Path]:
    build = exp / "standard_cuda"
    external_mlir = build / "gemma_external.mlir"
    inlined_mlir = build / "gemma_external_inlined.mlir"
    params = build / "gemma_params.irpa"

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


def compile_vmfb(args, exp: Path, mlir_path: Path, params: Path) -> Path:
    build = exp / "standard_cuda"
    vmfb = build / "gemma_cuda.vmfb"
    if vmfb.exists() and not args.force_compile:
        print(f"Reusing VMFB: {vmfb}")
        return vmfb

    cmd = [
        require_tool("iree-compile"),
        "--iree-input-type=onnx",
        "--iree-input-demote-i64-to-i32",
        "--iree-opt-strip-assertions",
        "--iree-hal-target-backends=cuda",
        f"--iree-cuda-target={args.cuda_target}",
        mlir_path,
        "-o",
        vmfb,
    ]
    if args.compile_with_param_import:
        cmd.insert(2, f"--iree-parameter-import={params}")
    if not args.enable_reduction_vector_distribution:
        cmd.insert(-3, "--iree-codegen-llvmgpu-use-reduction-vector-distribution=false")

    run_cmd(
        cmd,
        build / "logs" / "compile_cuda.log",
        args.timeout_seconds,
    )
    return vmfb


def run_module(args, exp: Path, vmfb: Path, params: Path):
    build = exp / "standard_cuda"
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


def main():
    args = parse_args()
    args.model_path = args.model_path.resolve()
    exp = experiment_dir(args)

    onnx_path = exp / "gemma_last_token.onnx"
    mlir_path = exp / "standard_cuda" / "gemma_external_inlined.mlir"
    params = exp / "standard_cuda" / "gemma_params.irpa"
    vmfb = exp / "standard_cuda" / "gemma_cuda.vmfb"

    if args.action in {"export", "all"}:
        onnx_path = export_onnx(args, exp)
    if args.action in {"import", "all"}:
        onnx_path = export_onnx(args, exp) if not onnx_path.exists() else onnx_path
        mlir_path, params = import_onnx(args, exp, onnx_path)
    if args.action in {"compile", "all"}:
        onnx_path = export_onnx(args, exp) if not onnx_path.exists() else onnx_path
        if not mlir_path.exists() or not params.exists():
            mlir_path, params = import_onnx(args, exp, onnx_path)
        vmfb = compile_vmfb(args, exp, mlir_path, params)
    if args.action in {"run", "all"}:
        if not vmfb.exists():
            raise RuntimeError(f"VMFB does not exist yet: {vmfb}")
        if not params.exists():
            raise RuntimeError(f"Parameter archive does not exist yet: {params}")
        run_module(args, exp, vmfb, params)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
