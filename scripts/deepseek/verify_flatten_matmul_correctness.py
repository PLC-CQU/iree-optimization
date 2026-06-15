#!/usr/bin/env python3
"""Compare baseline and flatten-matmul IREE outputs for b8/s64 last-token logits."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-module", default="deepseek_r1_8b_onnx_iree_cuda_b8_s64_last_baseline.vmfb")
    parser.add_argument("--baseline-params", default="seq64_b8_opt/build_common/deepseek_r1_8b_params.irpa")
    parser.add_argument("--optimized-module", default="deepseek_r1_8b_onnx_iree_cuda_b8_s64_last_flatten_matmul.vmfb")
    parser.add_argument("--optimized-params", default="seq64_b8_opt/build_flatten_matmul/deepseek_r1_8b_params.irpa")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seq", type=int, default=64)
    parser.add_argument("--input-ids-value", type=int, default=0)
    parser.add_argument("--attention-mask-value", type=int, default=1)
    parser.add_argument("--out-dir", type=Path, default=Path("correctness_b8_s64_flatten_matmul"))
    parser.add_argument("--report", type=Path, default=Path("correctness_b8_s64_flatten_matmul.json"))
    return parser.parse_args()


def run_module(label: str, module: str, params: str, output_path: Path, args):
    cmd = [
        "iree-run-module",
        f"--module={module}",
        f"--parameters=model={params}",
        "--parameter_mode=file",
        f"--device={args.device}",
        "--function=main_graph",
        f"--input={args.batch}x{args.seq}xi64={args.input_ids_value}",
        f"--input={args.batch}x{args.seq}xi64={args.attention_mask_value}",
        f"--output=@{output_path}",
    ]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": args.gpu} if args.gpu is not None else None
    print(f"\nRunning {label}:")
    print("  " + " \\\n  ".join(shlex.quote(part) for part in cmd))
    started = datetime.now()
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    finished = datetime.now()
    return {
        "label": label,
        "command": cmd,
        "started": started.isoformat(),
        "finished": finished.isoformat(),
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "output": str(output_path),
    }


def load_output(path: Path, batch: int):
    arr = np.load(path)
    if arr.dtype == np.uint8:
        arr = arr.view(np.float32)
    else:
        arr = arr.astype(np.float32, copy=False)
    if arr.ndim == 1 and arr.size % batch == 0:
        arr = arr.reshape(batch, arr.size // batch)
    return arr


def compare_arrays(reference: np.ndarray, candidate: np.ndarray):
    ref = reference.astype(np.float32, copy=False)
    cand = candidate.astype(np.float32, copy=False)
    diff = cand - ref
    abs_diff = np.abs(diff)
    denom = np.maximum(np.abs(ref), 1e-6)
    rel_diff = abs_diff / denom

    flat_index = int(np.argmax(abs_diff))
    max_index = np.unravel_index(flat_index, abs_diff.shape)

    return {
        "shape": list(ref.shape),
        "dtype_reference": str(reference.dtype),
        "dtype_candidate": str(candidate.dtype),
        "max_abs_error": float(abs_diff[max_index]),
        "mean_abs_error": float(abs_diff.mean()),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "max_relative_error": float(rel_diff.max()),
        "mean_relative_error": float(rel_diff.mean()),
        "allclose_atol_1e_3_rtol_1e_3": bool(np.allclose(ref, cand, atol=1e-3, rtol=1e-3)),
        "allclose_atol_1e_2_rtol_1e_2": bool(np.allclose(ref, cand, atol=1e-2, rtol=1e-2)),
        "max_error_index": [int(i) for i in max_index],
        "reference_at_max_error": float(ref[max_index]),
        "candidate_at_max_error": float(cand[max_index]),
    }


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    baseline_npy = args.out_dir / "baseline_logits.npy"
    optimized_npy = args.out_dir / "flatten_matmul_logits.npy"

    baseline_run = run_module("baseline", args.baseline_module, args.baseline_params, baseline_npy, args)
    optimized_run = run_module("flatten_matmul", args.optimized_module, args.optimized_params, optimized_npy, args)

    report = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "batch": args.batch,
            "seq": args.seq,
            "input_ids_value": args.input_ids_value,
            "attention_mask_value": args.attention_mask_value,
            "gpu": args.gpu,
        },
        "runs": {
            "baseline": baseline_run,
            "flatten_matmul": optimized_run,
        },
    }

    if baseline_run["returncode"] == 0 and optimized_run["returncode"] == 0:
        baseline = load_output(baseline_npy, args.batch)
        optimized = load_output(optimized_npy, args.batch)
        if baseline.shape != optimized.shape:
            report["comparison"] = {
                "error": "shape mismatch",
                "baseline_shape": list(baseline.shape),
                "optimized_shape": list(optimized.shape),
            }
        else:
            report["comparison"] = compare_arrays(baseline, optimized)
    else:
        report["comparison"] = {"error": "one or both module runs failed"}

    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved report: {args.report}")
    print(json.dumps(report.get("comparison", {}), indent=2, ensure_ascii=False))

    if baseline_run["returncode"] != 0:
        raise SystemExit(baseline_run["returncode"])
    if optimized_run["returncode"] != 0:
        raise SystemExit(optimized_run["returncode"])


if __name__ == "__main__":
    main()
