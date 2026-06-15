#!/usr/bin/env python3
"""Collect Gemma baseline/flatten benchmark logs into one CSV and JSON file."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent


BENCH_RE = re.compile(
    r"^BM_(?P<name>\S+)\s+"
    r"(?P<time>[0-9.]+)\s+(?P<cpu>[0-9.]+)\s+"
    r"(?P<unit>ns|us|ms|s)\s+"
    r"(?P<iters>[0-9]+)"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiments",
        nargs="+",
        required=True,
        help="Experiment names under ./experiments, e.g. gemma_e4b_it_b1_s32_fp16_cuda_standard",
    )
    parser.add_argument("--out-csv", type=Path, default=ROOT / "experiments" / "gemma_benchmark_summary.csv")
    parser.add_argument("--out-json", type=Path, default=ROOT / "experiments" / "gemma_benchmark_summary.json")
    return parser.parse_args()


def unit_to_ms(value: float, unit: str) -> float:
    if unit == "ns":
        return value / 1_000_000.0
    if unit == "us":
        return value / 1_000.0
    if unit == "ms":
        return value
    if unit == "s":
        return value * 1000.0
    raise ValueError(f"unknown unit: {unit}")


def shape_from_experiment(name: str) -> tuple[int | None, int | None]:
    match = re.search(r"_b(?P<b>[0-9]+)_s(?P<s>[0-9]+)_", name)
    if not match:
        return None, None
    return int(match.group("b")), int(match.group("s"))


def parse_log(path: Path):
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = BENCH_RE.match(line.strip())
        if match:
            real_time = float(match.group("time"))
            cpu_time = float(match.group("cpu"))
            unit = match.group("unit")
            return {
                "benchmark": match.group("name"),
                "real_time": real_time,
                "cpu_time": cpu_time,
                "time_unit": unit,
                "latency_ms": unit_to_ms(real_time, unit),
                "iterations": int(match.group("iters")),
            }
    text = path.read_text(encoding="utf-8", errors="replace")
    if "CUDA_ERROR_OUT_OF_MEMORY" in text:
        return {"error": "CUDA_ERROR_OUT_OF_MEMORY"}
    return {"error": "no benchmark result found"}


def main():
    args = parse_args()
    rows = []
    for experiment in args.experiments:
        exp = ROOT / "experiments" / experiment
        batch, seq = shape_from_experiment(experiment)
        for variant, rel in [
            ("baseline", "standard_cuda/logs/benchmark_cuda.log"),
            ("flatten_matmul", "standard_cuda_flatten_matmul/logs/benchmark_cuda.log"),
        ]:
            parsed = parse_log(exp / rel)
            row = {
                "experiment": experiment,
                "variant": variant,
                "batch": batch,
                "seq": seq,
            }
            if parsed is None:
                row.update({"status": "missing"})
            elif "error" in parsed:
                row.update({"status": "error", "error": parsed["error"]})
            else:
                row.update({"status": "ok", **parsed})
            rows.append(row)

    ok_by_shape = {}
    for row in rows:
        if row.get("status") == "ok":
            ok_by_shape.setdefault((row["batch"], row["seq"]), {})[row["variant"]] = row
    for row in rows:
        pair = ok_by_shape.get((row["batch"], row["seq"]), {})
        baseline = pair.get("baseline")
        flatten = pair.get("flatten_matmul")
        if baseline and flatten:
            base_ms = baseline["latency_ms"]
            flat_ms = flatten["latency_ms"]
            speedup = base_ms / flat_ms if flat_ms else None
            row["baseline_latency_ms"] = base_ms
            row["flatten_latency_ms"] = flat_ms
            row["speedup_vs_baseline"] = speedup
            row["latency_delta_ms"] = flat_ms - base_ms

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "experiment",
        "variant",
        "batch",
        "seq",
        "status",
        "latency_ms",
        "time_unit",
        "iterations",
        "baseline_latency_ms",
        "flatten_latency_ms",
        "speedup_vs_baseline",
        "latency_delta_ms",
        "error",
    ]
    with args.out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    args.out_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote CSV: {args.out_csv}")
    print(f"Wrote JSON: {args.out_json}")
    for row in rows:
        if row.get("status") == "ok":
            print(
                f"{row['experiment']} {row['variant']}: "
                f"{row['latency_ms']:.3f} ms"
            )
        else:
            print(f"{row['experiment']} {row['variant']}: {row.get('status')} {row.get('error', '')}")


if __name__ == "__main__":
    main()
