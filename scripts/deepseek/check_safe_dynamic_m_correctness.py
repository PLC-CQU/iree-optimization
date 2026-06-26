#!/usr/bin/env python3
"""Compare DeepSeek old_dynamic, safe_dynamic_m, optional candidate, and static outputs."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("/home/zhongjialin/projects/iree/deepseek-R1-Llama-8b"))
    p.add_argument("--iree-build", type=Path, default=Path("/home/zhongjialin/projects/iree-build"))
    p.add_argument("--gpu", default="2")
    p.add_argument("--batch", "-b", type=int, default=1)
    p.add_argument("--seq", "-s", type=int, default=32)
    p.add_argument("--atol", type=float, default=0.06)
    p.add_argument("--rtol", type=float, default=0.01)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--candidate-dynamic-vmfb", type=Path, default=None)
    return p.parse_args()


def run(cmd: list[str], gpu: str):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    print("  " + " \\\n  ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def compare(name: str, a: np.ndarray, b: np.ndarray, atol: float, rtol: float):
    a32 = a.astype(np.float32)
    b32 = b.astype(np.float32)
    diff = np.abs(a32 - b32)
    close = np.allclose(a32, b32, atol=atol, rtol=rtol)
    print(name)
    print(f"  shape_a: {a.shape}")
    print(f"  shape_b: {b.shape}")
    print(f"  max_abs: {float(diff.max()):.8f}")
    print(f"  mean_abs: {float(diff.mean()):.8f}")
    print(f"  allclose_atol_{atol}_rtol_{rtol}: {close}")
    if not close:
        raise SystemExit(1)


def main():
    args = parse_args()
    b, s = args.batch, args.seq
    root = args.root
    out_dir = args.out_dir or Path(f"/tmp/iree_safe_dynamic_m_correctness_b{b}_s{s}")
    out_dir.mkdir(parents=True, exist_ok=True)

    old_vmfb = root / "deepseek_dynamic_b_s_last_flatten_matmul_input_i32_with_demote.vmfb"
    safe_vmfb = root / "deepseek_dynamic_b_s_last_safe_dynamic_m_input_i32_with_demote.vmfb"
    dynamic_params = root / "dynamic_shape_b_s/build_flatten_matmul_input_i32_with_demote/deepseek_r1_8b_params.irpa"
    static_vmfb = root / f"deepseek_r1_8b_onnx_iree_cuda_b{b}_s{s}_last_nonpad_index_flatten_matmul_input_i32_with_demote.vmfb"
    static_params = root / f"flatten_shape_b{b}_s{s}/build_flatten_matmul_last_nonpad_index_input_i32_with_demote/deepseek_r1_8b_params.irpa"

    required = [old_vmfb, safe_vmfb, dynamic_params, static_vmfb, static_params]
    if args.candidate_dynamic_vmfb:
        required.append(args.candidate_dynamic_vmfb)
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)

    def run_dynamic(name: str, module: Path):
        output = out_dir / f"{name}.npy"
        print(f"\n## {name}")
        run([
            str(args.iree_build / "tools/iree-run-module"),
            f"--module={module}",
            f"--parameters=model={dynamic_params}",
            "--parameter_mode=file",
            "--device=cuda",
            "--function=main_graph",
            f"--input={b}x{s}xi32=0",
            f"--input={b}x{s}xi32=1",
            f"--output=@{output}",
        ], args.gpu)

    def run_static():
        output = out_dir / "static_exact.npy"
        print("\n## static_exact")
        run([
            str(args.iree_build / "tools/iree-run-module"),
            f"--module={static_vmfb}",
            f"--parameters=model={static_params}",
            "--parameter_mode=file",
            "--device=cuda",
            "--function=main_graph",
            f"--input={b}x{s}xi32=0",
            f"--input={b}x{s}xi32=1",
            f"--input={b}xi32={s - 1}",
            f"--output=@{output}",
        ], args.gpu)

    run_dynamic("old_dynamic", old_vmfb)
    run_dynamic("safe_dynamic_m", safe_vmfb)
    if args.candidate_dynamic_vmfb:
        run_dynamic("candidate_dynamic", args.candidate_dynamic_vmfb)
    run_static()

    old = np.load(out_dir / "old_dynamic.npy")
    safe = np.load(out_dir / "safe_dynamic_m.npy")
    static = np.load(out_dir / "static_exact.npy")
    print("\n## correctness summary")
    compare("old_dynamic vs safe_dynamic_m", old, safe, args.atol, args.rtol)
    if args.candidate_dynamic_vmfb:
        candidate = np.load(out_dir / "candidate_dynamic.npy")
        compare("safe_dynamic_m vs candidate_dynamic", safe, candidate, args.atol, args.rtol)
        compare("candidate_dynamic vs static_exact", candidate, static, args.atol, args.rtol)
    compare("safe_dynamic_m vs static_exact", safe, static, args.atol, args.rtol)
    print(f"output_dir: {out_dir}")


if __name__ == "__main__":
    main()
