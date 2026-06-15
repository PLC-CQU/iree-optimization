#!/usr/bin/env python3
"""Benchmark the batch=1 last-token IREE CUDA VMFB.

Uses iree-benchmark-module so external parameter archives are loaded exactly
like the successful iree-run-module smoke test.
"""

import argparse
import json
import shlex
import subprocess
from datetime import datetime
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", default="deepseek_r1_8b_onnx_iree_cuda_b1_s128_last_sm86.vmfb")
    parser.add_argument("--params", default="from_model_cuda_build_b1_last/deepseek_r1_8b_params.irpa")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu", default=None, help="Optional CUDA_VISIBLE_DEVICES value, e.g. 0")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq", type=int, default=128)
    parser.add_argument("--input-dtype", choices=["i64", "i32"], default="i64")
    parser.add_argument("--input-id-value", type=int, default=0)
    parser.add_argument("--attention-mask-value", type=int, default=1)
    parser.add_argument("--input-ids-file", default=None, help="Optional .npy file for input_ids.")
    parser.add_argument("--attention-mask-file", default=None, help="Optional .npy file for attention_mask.")
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--min-time", default="10x", help="Google Benchmark min_time, e.g. 10x or 5s")
    parser.add_argument("--warmup-time", default="1.0", help="Warmup time in seconds.")
    parser.add_argument("--out", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(args.out or f"benchmark_b1_last_cuda_{stamp}.json")
    raw_out = out.with_suffix(".googlebench.json")

    input_ids_arg = (
        f"--input=@{args.input_ids_file}"
        if args.input_ids_file
        else f"--input={args.batch}x{args.seq}x{args.input_dtype}={args.input_id_value}"
    )
    attention_mask_arg = (
        f"--input=@{args.attention_mask_file}"
        if args.attention_mask_file
        else f"--input={args.batch}x{args.seq}x{args.input_dtype}={args.attention_mask_value}"
    )

    cmd = [
        "iree-benchmark-module",
        f"--module={args.module}",
        f"--parameters=model={args.params}",
        "--parameter_mode=file",
        f"--device={args.device}",
        "--function=main_graph",
        input_ids_arg,
        attention_mask_arg,
        f"--benchmark_repetitions={args.repetitions}",
        f"--benchmark_min_time={args.min_time}",
        f"--benchmark_min_warmup_time={args.warmup_time}",
        "--benchmark_format=console",
        f"--benchmark_out={raw_out}",
        "--benchmark_out_format=json",
        "--benchmark_time_unit=ms",
    ]

    env = None
    if args.gpu is not None:
        env = {"CUDA_VISIBLE_DEVICES": args.gpu}
        import os

        env = {**os.environ, **env}

    print("Command:")
    print("  " + " \\\n  ".join(shlex.quote(part) for part in cmd))
    if args.gpu is not None:
        print(f"CUDA_VISIBLE_DEVICES={args.gpu}")

    started = datetime.now()
    result = subprocess.run(cmd, text=True, capture_output=True, env=env)
    finished = datetime.now()

    summary = {
        "started": started.isoformat(),
        "finished": finished.isoformat(),
        "returncode": result.returncode,
        "command": cmd,
        "cuda_visible_devices": args.gpu,
        "module": args.module,
        "params": args.params,
        "input_id_value": args.input_id_value,
        "attention_mask_value": args.attention_mask_value,
        "input_dtype": args.input_dtype,
        "input_ids_file": args.input_ids_file,
        "attention_mask_file": args.attention_mask_file,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "google_benchmark_json": str(raw_out),
    }
    if raw_out.exists():
        try:
            summary["google_benchmark"] = json.loads(raw_out.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            summary["google_benchmark_parse_error"] = str(exc)

    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    print(f"Saved summary: {out}")
    if result.returncode != 0:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
