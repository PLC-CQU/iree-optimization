#!/usr/bin/env python3
"""Benchmark DeepSeek old_dynamic vs safe_dynamic_m vs optional static exact."""

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
    p.add_argument("--batch", "-b", type=int, default=4)
    p.add_argument("--seq", "-s", type=int, default=64)
    p.add_argument("--repetitions", type=int, default=3)
    p.add_argument("--min-time", default="1x")
    p.add_argument("--warmup-time", default="0.5")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--old-dynamic-vmfb", type=Path, default=None)
    p.add_argument("--safe-dynamic-m-vmfb", type=Path, default=None)
    p.add_argument("--params", type=Path, default=None)
    p.add_argument("--static-vmfb", type=Path, default=None)
    p.add_argument("--static-params", type=Path, default=None)
    return p.parse_args()


def run(cmd: list[str], gpu: str):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    print("  " + " \\\n  ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def mean_ms(path: Path) -> float | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    vals = []
    for bench in data.get("benchmarks", []):
        if bench.get("name", "").endswith("_mean"):
            return float(bench["real_time"])
        if bench.get("name", "").endswith("/real_time"):
            vals.append(float(bench["real_time"]))
    return sum(vals) / len(vals) if vals else None


def main():
    args = parse_args()
    b, s = args.batch, args.seq
    root = args.root
    out_dir = args.out_dir or Path(f"/tmp/iree_safe_dynamic_m_benchmark_b{b}_s{s}")
    out_dir.mkdir(parents=True, exist_ok=True)

    old_vmfb = args.old_dynamic_vmfb or root / "deepseek_dynamic_b_s_last_flatten_matmul_input_i32_with_demote.vmfb"
    safe_vmfb = args.safe_dynamic_m_vmfb or root / "deepseek_dynamic_b_s_last_safe_dynamic_m_input_i32_with_demote.vmfb"
    params = args.params or root / "dynamic_shape_b_s/build_flatten_matmul_input_i32_with_demote/deepseek_r1_8b_params.irpa"
    static_vmfb = args.static_vmfb or root / f"deepseek_r1_8b_onnx_iree_cuda_b{b}_s{s}_last_nonpad_index_flatten_matmul_input_i32_with_demote.vmfb"
    static_params = args.static_params or root / f"flatten_shape_b{b}_s{s}/build_flatten_matmul_last_nonpad_index_input_i32_with_demote/deepseek_r1_8b_params.irpa"

    for required in [old_vmfb, safe_vmfb, params]:
        if not required.exists():
            raise FileNotFoundError(required)

    def run_case(name: str, module: Path, param_archive: Path, last_token_indices: str | None = None):
        out_json = out_dir / f"{name}.googlebench.json"
        cmd = [
            str(args.iree_build / "tools/iree-benchmark-module"),
            f"--module={module}",
            f"--parameters=model={param_archive}",
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
        print(f"\n## {name}\nmodule: {module}")
        run(cmd, args.gpu)

    run_case("old_dynamic", old_vmfb, params)
    run_case("safe_dynamic_m", safe_vmfb, params)
    if static_vmfb.exists() and static_params.exists():
        run_case("static_exact", static_vmfb, static_params, f"{b}xi32={s - 1}")
    else:
        print(f"\n## static_exact\nskip: missing {static_vmfb} or {static_params}")

    old = mean_ms(out_dir / "old_dynamic.googlebench.json")
    safe = mean_ms(out_dir / "safe_dynamic_m.googlebench.json")
    static = mean_ms(out_dir / "static_exact.googlebench.json")
    if old is None or safe is None:
        raise RuntimeError(f"missing benchmark results in {out_dir}")
    print("\n## summary")
    print(f"old_dynamic_ms: {old:.3f}")
    print(f"safe_dynamic_m_ms: {safe:.3f}")
    print(f"speedup: {old / safe:.3f}x")
    print(f"improvement: {(old - safe) / old * 100.0:.2f}%")
    if static is not None:
        print(f"static_exact_ms: {static:.3f}")
        print(f"safe_vs_static_slowdown: {safe / static:.3f}x")
        print(f"safe_vs_static_gap: {(safe - static) / static * 100.0:.2f}%")
    print(f"json_dir: {out_dir}")


if __name__ == "__main__":
    main()
