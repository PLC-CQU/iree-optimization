#!/usr/bin/env python3
"""Export, compile, and benchmark the fully dynamic-shape VMFB."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from pathlib import Path


DEFAULT_REQUESTS = [
    "b1_s16",
    "b1_s32",
    "b1_s48",
    "b1_s64",
    "b1_s96",
    "b1_s128",
    "b2_s48",
    "b3_s80",
    "b4_s64",
    "b5_s96",
    "b8_s128",
]


def parse_shape(shape: str) -> tuple[int, int]:
    if not shape.startswith("b") or "_s" not in shape:
        raise argparse.ArgumentTypeError(f"invalid shape '{shape}', expected b8_s128")
    batch, seq = shape[1:].split("_s", 1)
    return int(batch), int(seq)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", choices=["plan", "export", "rewrite", "compile", "benchmark", "all"], default="plan")
    parser.add_argument("--variant", choices=["baseline", "flatten_matmul"], default="flatten_matmul")
    parser.add_argument("--model-path", default="./model")
    parser.add_argument("--work-dir", type=Path, default=Path("dynamic_shape_b_s"))
    parser.add_argument("--sample-batch", type=int, default=1)
    parser.add_argument("--sample-seq", type=int, default=32)
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device used only for PyTorch ONNX export. Benchmarking still uses CUDA.",
    )
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--cuda-target", default="sm_86")
    parser.add_argument(
        "--input-dtype",
        choices=["i32", "i64"],
        default="i32",
        help="Runtime dtype passed to iree-benchmark-module. Default matches --iree-input-demote-i64-to-i32.",
    )
    parser.add_argument("--requests", nargs="+", default=DEFAULT_REQUESTS)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--min-time", default="10x")
    parser.add_argument("--warmup-time", default="1.0")
    parser.add_argument("--timeout-seconds", type=int, default=21600)
    parser.add_argument("--force-export", action="store_true")
    parser.add_argument("--force-import", action="store_true")
    parser.add_argument("--force-compile", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def paths(args):
    original = args.work_dir / "deepseek_dynamic_b_s_last.onnx"
    rewritten = args.work_dir / "deepseek_dynamic_b_s_last_flatten_matmul.onnx"
    onnx_path = rewritten if args.variant == "flatten_matmul" else original
    build_dir = args.work_dir / f"build_{args.variant}"
    vmfb = Path(f"deepseek_dynamic_b_s_last_{args.variant}.vmfb")
    return {
        "original_onnx": original,
        "rewritten_onnx": rewritten,
        "rewrite_report": args.work_dir / "rewrite_pass_dynamic_b_s.json",
        "onnx": onnx_path,
        "build_dir": build_dir,
        "params": build_dir / "deepseek_r1_8b_params.irpa",
        "vmfb": vmfb,
    }


def print_command(cmd: list[object]):
    print("Command:")
    print("  " + " \\\n  ".join(shlex.quote(str(part)) for part in cmd))


def run(cmd: list[object], args, env=None):
    print_command(cmd)
    if args.dry_run or args.action == "plan":
        return
    subprocess.run([str(part) for part in cmd], check=True, timeout=args.timeout_seconds, env=env)


def maybe_run_export(args, ps):
    if ps["original_onnx"].exists() and not args.force_export:
        print(f"Reusing dynamic ONNX: {ps['original_onnx']}")
        return
    run(export_command(args, ps), args)


def maybe_run_rewrite(args, ps):
    if args.variant != "flatten_matmul":
        return
    if ps["rewritten_onnx"].exists() and not args.force_export:
        print(f"Reusing dynamic flattened ONNX: {ps['rewritten_onnx']}")
        return
    run(rewrite_command(args, ps), args)


def export_command(args, ps):
    return [
        "python3",
        "export_dynamic_onnx.py",
        "--model-path",
        args.model_path,
        "--out",
        ps["original_onnx"],
        "--sample-batch",
        args.sample_batch,
        "--sample-seq",
        args.sample_seq,
        "--device",
        args.device,
        "--last-token-only",
    ]


def rewrite_command(args, ps):
    return [
        "python3",
        "rewrite_onnx_flatten_matmul.py",
        "--input",
        ps["original_onnx"],
        "--output",
        ps["rewritten_onnx"],
        "--dynamic-shape",
        "--check",
        "--report",
        ps["rewrite_report"],
        "--max-report-records",
        "5",
    ]


def compile_command(args, ps):
    cmd = [
        "python3",
        "compile_onnx_iree_cuda.py",
        "--onnx",
        ps["onnx"],
        "--build-dir",
        ps["build_dir"],
        "--output",
        ps["vmfb"],
        "--cuda-target",
        args.cuda_target,
        "--batch",
        args.sample_batch,
        "--seq",
        args.sample_seq,
        "--optimization-preset",
        "baseline",
    ]
    if args.force_import:
        cmd.append("--force-import")
    if args.force_compile:
        cmd.append("--force-compile")
    return cmd


def benchmark_commands(args, ps):
    commands = []
    for shape in args.requests:
        batch, seq = parse_shape(shape)
        commands.append(
            [
                "python3",
                "benchmark_b1_last_cuda.py",
                "--module",
                ps["vmfb"],
                "--params",
                ps["params"],
                "--gpu",
                args.gpu,
                "--batch",
                batch,
                "--seq",
                seq,
                "--input-dtype",
                args.input_dtype,
                "--repetitions",
                args.repetitions,
                "--min-time",
                args.min_time,
                "--warmup-time",
                args.warmup_time,
                "--out",
                args.work_dir / f"benchmark_dynamic_{shape}_{args.variant}.json",
            ]
        )
    return commands


def main():
    args = parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    ps = paths(args)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": args.gpu} if args.gpu is not None else None

    if args.action in ("plan", "export", "rewrite", "compile", "all"):
        maybe_run_export(args, ps)
    if args.action in ("plan", "rewrite", "compile", "all"):
        maybe_run_rewrite(args, ps)
    if args.action in ("plan", "compile", "all"):
        run(compile_command(args, ps), args)
    if args.action in ("plan", "benchmark", "all"):
        for cmd in benchmark_commands(args, ps):
            run(cmd, args, env=env)


if __name__ == "__main__":
    main()
