#!/usr/bin/env python3
"""Aggregate repeated-single Gemma benchmark summaries across static shapes."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapes", nargs="+", required=True, help="Shapes like b4_s32 b8_s128")
    parser.add_argument("--dtype-label", default="fp16")
    parser.add_argument("--model-label", default="gemma_e4b_it")
    parser.add_argument("--model-name", default="googlegemma-4-E4B-it")
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results" / "gemma_repeated_static_sweep_summary.json",
    )
    return parser.parse_args()


def parse_shape(shape: str) -> tuple[int, int]:
    match = re.fullmatch(r"b([0-9]+)_s([0-9]+)", shape)
    if not match:
        raise ValueError(f"invalid shape: {shape}")
    return int(match.group(1)), int(match.group(2))


def experiment_name(model_label: str, batch: int, seq: int, dtype_label: str) -> str:
    return f"{model_label}_b{batch}_s{seq}_{dtype_label}_cuda_standard"


def main():
    args = parse_args()
    rows = []
    for shape in args.shapes:
        batch, seq = parse_shape(shape)
        exp_name = experiment_name(args.model_label, batch, seq, args.dtype_label)
        path = ROOT / "experiments" / exp_name / "benchmark_repeated_single_compare.json"
        row = {
            "shape": shape,
            "batch": batch,
            "seq": seq,
            "token_slots": batch * seq,
            "experiment": exp_name,
            "summary_file": str(path),
        }
        if not path.exists():
            row["status"] = "missing"
            rows.append(row)
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        perf = data.get("performance", {})
        row["status"] = "ok"
        row["performance"] = perf
        baseline = perf.get("baseline", {})
        flatten = perf.get("flatten_matmul", {})
        if baseline.get("successful_runs") and flatten.get("successful_runs"):
            row["baseline_latency_ms_mean"] = baseline.get("latency_ms_mean")
            row["flatten_latency_ms_mean"] = flatten.get("latency_ms_mean")
            row["speedup"] = perf.get("speedup")
            row["latency_reduction_percent"] = perf.get("latency_reduction_percent")
            row["baseline_tokens_per_second"] = baseline.get("tokens_per_second")
            row["flatten_tokens_per_second"] = flatten.get("tokens_per_second")
        rows.append(row)

    summary = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "model": args.model_name,
            "model_label": args.model_label,
            "method": "repeated independent single-shot benchmark-module runs",
            "shapes": args.shapes,
        },
        "results": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved summary: {args.out}")


if __name__ == "__main__":
    main()
