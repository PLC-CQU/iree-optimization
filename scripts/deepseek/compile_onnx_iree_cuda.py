#!/usr/bin/env python3
"""Compile the local DeepSeek ONNX export with IREE CUDA.

This path uses:
  ONNX -> iree-import-onnx external-parameter MLIR -> IREE CUDA VMFB

The local IREE 3.11 compiler fails with sm_89 for this imported graph with
"missing GPU target in #hal.executable.target". sm_86 compiles successfully and
is the conservative default here.
"""

import argparse
import shlex
import shutil
import subprocess
from datetime import datetime
from pathlib import Path


_COMPILER_HELP = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, default=Path("fresh_from_model.onnx"))
    parser.add_argument("--build-dir", type=Path, default=Path("from_model_cuda_build"))
    parser.add_argument("--output", type=Path, default=Path("deepseek_r1_8b_onnx_iree_cuda_sm86.vmfb"))
    parser.add_argument("--cuda-target", default="sm_86")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seq", type=int, default=128)
    parser.add_argument(
        "--optimization-preset",
        choices=["baseline", "ptxas_o3", "fusion", "fusion_ptxas_o3", "balanced", "balanced_no_indirect", "ultimate"],
        default="baseline",
        help="baseline is the known-good compile; balanced/ultimate add CUDA flags from compile_ultimate_optimization.py.",
    )
    parser.add_argument("--prefetch-stages", type=int, default=2)
    parser.add_argument("--const-inline-bytes", type=int, default=4096)
    parser.add_argument("--ptxas-o3", action="store_true")
    parser.add_argument("--no-ptxas", action="store_true", help="Do not pass --iree-cuda-use-ptxas.")
    parser.add_argument(
        "--extra-compile-flag",
        action="append",
        default=[],
        help="Additional raw flag passed to iree-compile. May be repeated.",
    )
    parser.add_argument("--disable-indirect-command-buffers", action="store_true")
    parser.add_argument("--force-import", action="store_true")
    parser.add_argument("--force-compile", action="store_true")
    parser.add_argument(
        "--no-demote-i64-to-i32",
        action="store_true",
        help="Do not pass --iree-input-demote-i64-to-i32. Use this when the VMFB should keep the original ONNX input dtype.",
    )
    parser.add_argument("--timeout-seconds", type=int, default=21600)
    return parser.parse_args()


def require_tool(name: str) -> str:
    tool = shutil.which(name)
    if not tool:
        raise RuntimeError(f"Missing required tool: {name}")
    return tool


def compiler_supports(flag_name: str) -> bool:
    global _COMPILER_HELP
    if _COMPILER_HELP is None:
        compiler = require_tool("iree-compile")
        result = subprocess.run([compiler, "--help"], capture_output=True, text=True, timeout=30)
        _COMPILER_HELP = (result.stdout or "") + "\n" + (result.stderr or "")
    return flag_name in _COMPILER_HELP


def run_command(cmd, log_path: Path, timeout_seconds: int):
    print("Command:")
    print("  " + " \\\n  ".join(shlex.quote(str(part)) for part in cmd))
    print(f"Log: {log_path}")
    start = datetime.now()
    result = subprocess.run(
        [str(part) for part in cmd],
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "\n".join([
            f"Started: {start.isoformat()}",
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
        ]),
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with code {result.returncode}. See {log_path}")


def import_onnx(args, external_mlir: Path, params_irpa: Path):
    if external_mlir.exists() and params_irpa.exists() and not args.force_import:
        print(f"Reusing imported MLIR: {external_mlir}")
        print(f"Reusing params: {params_irpa}")
        return

    importer = require_tool("iree-import-onnx")
    args.build_dir.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            importer,
            args.onnx,
            "--large-model",
            "--externalize-params",
            "--param-gb-threshold",
            "2",
            "--save-params-to",
            params_irpa,
            "-o",
            external_mlir,
        ],
        args.build_dir / "import_onnx.log",
        args.timeout_seconds,
    )


def inline_resources(args, external_mlir: Path, inlined_mlir: Path):
    script = Path(__file__).with_name("inline_onnx_dense_resources.py")
    run_command(
        ["python3", script, external_mlir, "-o", inlined_mlir],
        args.build_dir / "inline_resources.log",
        args.timeout_seconds,
    )


def compile_vmfb(args, inlined_mlir: Path, params_irpa: Path):
    if args.output.exists() and not args.force_compile:
        print(f"VMFB already exists: {args.output}")
        return

    compiler = require_tool("iree-compile")
    flags = build_compile_flags(args, params_irpa)
    run_command(
        [compiler, *flags, inlined_mlir, "-o", args.output],
        args.build_dir / "compile_cuda.log",
        args.timeout_seconds,
    )


def build_compile_flags(args, params_irpa: Path):
    flags = [
        "--iree-input-type=onnx",
        "--iree-hal-target-backends=cuda",
        f"--iree-cuda-target={args.cuda_target}",
        f"--iree-parameter-import={params_irpa}",
        "--iree-opt-strip-assertions",
    ]
    if not args.no_ptxas:
        flags.insert(3, "--iree-cuda-use-ptxas")
    if not args.no_demote_i64_to_i32:
        flags.insert(-1, "--iree-input-demote-i64-to-i32")

    if args.optimization_preset in ("fusion", "fusion_ptxas_o3", "balanced", "balanced_no_indirect", "ultimate"):
        flags.extend([
            "--iree-global-opt-enable-attention-v-transpose",
            "--iree-global-opt-propagate-transposes",
            "--iree-dispatch-creation-enable-fuse-horizontal-contractions",
            "--iree-dispatch-creation-enable-early-trunc-fusion",
            "--iree-flow-enable-pad-handling",
            f"--iree-flow-inline-constants-max-byte-length={args.const_inline_bytes}",
            "--iree-opt-const-expr-hoisting",
        ])

    if args.optimization_preset in ("balanced", "balanced_no_indirect", "ultimate"):
        flags.extend([
            "--iree-codegen-llvmgpu-use-vector-distribution",
            "--iree-llvmgpu-enable-shared-memory-reuse",
            f"--iree-llvmgpu-prefetch-num-stages={args.prefetch_stages}",
        ])
        if args.optimization_preset != "balanced_no_indirect" and not args.disable_indirect_command_buffers:
            flags.append("--iree-hal-indirect-command-buffers")

    if args.optimization_preset in ("ptxas_o3", "fusion_ptxas_o3", "ultimate"):
        flags.append("--iree-cuda-use-ptxas-params=-O3")
    if args.optimization_preset == "ultimate":
        if compiler_supports("--iree-opt-strip-debug-ops"):
            flags.append("--iree-opt-strip-debug-ops")

    if args.ptxas_o3 and "--iree-cuda-use-ptxas-params=-O3" not in flags:
        flags.append("--iree-cuda-use-ptxas-params=-O3")

    flags.extend(args.extra_compile_flag)
    return flags


def main():
    args = parse_args()
    external_mlir = args.build_dir / "deepseek_r1_8b_external.mlir"
    inlined_mlir = args.build_dir / "deepseek_r1_8b_external_inlined.mlir"
    params_irpa = args.build_dir / "deepseek_r1_8b_params.irpa"

    import_onnx(args, external_mlir, params_irpa)
    inline_resources(args, external_mlir, inlined_mlir)
    compile_vmfb(args, inlined_mlir, params_irpa)

    print("Done.")
    print(f"VMFB: {args.output}")
    print(f"Params: {params_irpa}")
    print(f"Optimization preset: {args.optimization_preset}")
    input_dtype = "i64" if args.no_demote_i64_to_i32 else "i32"
    print("Run with:")
    print(
        "  iree-run-module "
        f"--module={args.output} "
        f"--parameters=model={params_irpa} "
        "--device=cuda "
        "--function=main_graph "
        f"--input={args.batch}x{args.seq}x{input_dtype}=0 "
        f"--input={args.batch}x{args.seq}x{input_dtype}=1"
    )
    if args.no_demote_i64_to_i32:
        print("Note: compiled without --iree-input-demote-i64-to-i32; ONNX int64 inputs stay int64 unless the ONNX graph was patched.")
    else:
        print("Note: compiled with --iree-input-demote-i64-to-i32; feed int32 inputs to the VMFB.")


if __name__ == "__main__":
    main()
