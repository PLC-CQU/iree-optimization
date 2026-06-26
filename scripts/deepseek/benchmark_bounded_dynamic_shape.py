#!/usr/bin/env python3
"""Benchmark safe_dynamic_m against a bounded/candidate dynamic DeepSeek VMFB."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("/home/zhongjialin/projects/iree/deepseek-R1-Llama-8b"))
    p.add_argument("--iree-build", type=Path, default=Path("/home/zhongjialin/projects/iree-build"))
    p.add_argument("--gpu", default="2")
    p.add_argument("--batch", "-b", type=int, default=1)
    p.add_argument("--seq", "-s", type=int, default=32)
    p.add_argument("--repetitions", type=int, default=3)
    p.add_argument("--min-time", default="1x")
    p.add_argument("--warmup-time", default="0")
    p.add_argument("--timeout-seconds", type=int, default=120)
    p.add_argument("--candidate-name", default="bounded_dynamic_b8_s128")
    p.add_argument("--candidate-vmfb", type=Path, default=Path("/tmp/deepseek_dynamic_bs_bounded_b8_s128_simt_context.vmfb"))
    p.add_argument("--out-dir", type=Path, default=None)
    return p.parse_args()


def run_benchmark(args, name: str, module: Path, params: Path, last_token_indices: str | None = None):
    b, s = args.batch, args.seq
    out_dir = args.out_dir or Path(f"/tmp/iree_bounded_dynamic_benchmark_b{b}_s{s}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"{name}.googlebench.json"
    cmd = [
        str(args.iree_build / "tools/iree-benchmark-module"),
        f"--module={module}",
        f"--parameters=model={params}",
        "--parameter_mode=file",
        "--device=cuda",
        "--function=main_graph",
        f"--input={b}x{s}xi32=0",
        f"--input={b}x{s}xi32=1",
        f"--benchmark_repetitions={args.repetitions}",
        f"--benchmark_min_time={args.min_time}",
        f"--benchmark_min_warmup_time={args.warmup_time}",
        "--benchmark_time_unit=ms",
        f"--benchmark_out={out_json}",
        "--benchmark_out_format=json",
    ]
    if last_token_indices:
        cmd.append(f"--input={last_token_indices}")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    print(f"\n## {name}\nmodule: {module}\nparams: {params}\ntimeout: {args.timeout_seconds}s")
    print("  " + " \\\n  ".join(cmd))
    subprocess.run(cmd, check=True, env=env, timeout=args.timeout_seconds)
    return out_json


def mean_ms(path: Path) -> float:
    data = json.loads(path.read_text())
    vals = []
    for item in data.get("benchmarks", []):
        if item.get("run_name", "").endswith("_mean") or item.get("name", "").endswith("_mean"):
            return float(item["real_time"])
        if item.get("run_type") == "iteration" or item.get("name", "").endswith("/real_time"):
            vals.append(float(item["real_time"]))
    if not vals:
        raise RuntimeError(f"missing benchmark time in {path}")
    return sum(vals) / len(vals)


def main():
    args = parse_args()
    b, s = args.batch, args.seq
    root = args.root
    out_dir = args.out_dir or Path(f"/tmp/iree_bounded_dynamic_benchmark_b{b}_s{s}")
    args.out_dir = out_dir

    safe_vmfb = root / "deepseek_dynamic_b_s_last_safe_dynamic_m_input_i32_with_demote.vmfb"
    dynamic_params = root / "dynamic_shape_b_s/build_flatten_matmul_input_i32_with_demote/deepseek_r1_8b_params.irpa"
    static_vmfb = root / f"deepseek_r1_8b_onnx_iree_cuda_b{b}_s{s}_last_nonpad_index_flatten_matmul_input_i32_with_demote.vmfb"
    static_params = root / f"flatten_shape_b{b}_s{s}/build_flatten_matmul_last_nonpad_index_input_i32_with_demote/deepseek_r1_8b_params.irpa"

    for path in [safe_vmfb, dynamic_params, args.candidate_vmfb, static_vmfb, static_params]:
        if not path.exists():
            raise FileNotFoundError(path)

    safe_json = run_benchmark(args, "safe_dynamic_m", safe_vmfb, dynamic_params)
    cand_json = run_benchmark(args, args.candidate_name, args.candidate_vmfb, dynamic_params)
    static_json = run_benchmark(args, "static_exact", static_vmfb, static_params, f"{b}xi32={s - 1}")

    safe = mean_ms(safe_json)
    candidate = mean_ms(cand_json)
    static = mean_ms(static_json)
    print("\n## summary")
    print(f"safe_dynamic_m_ms: {safe:.3f}")
    print(f"{args.candidate_name}_ms: {candidate:.3f}")
    print(f"static_exact_ms: {static:.3f}")
    print(f"{args.candidate_name}_vs_safe_speedup: {safe / candidate:.3f}x")
    print(f"safe_vs_static_slowdown: {safe / static:.3f}x")
    print(f"safe_vs_static_gap: {(safe / static - 1.0) * 100.0:.2f}%")
    print(f"{args.candidate_name}_vs_static_slowdown: {candidate / static:.3f}x")
    print(f"{args.candidate_name}_vs_static_gap: {(candidate / static - 1.0) * 100.0:.2f}%")
    print(f"json_dir: {out_dir}")


if __name__ == "__main__":
    main()
