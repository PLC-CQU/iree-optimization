#!/usr/bin/env python3
"""Run repeated single-shot Qwen benchmarks and aggregate baseline vs flatten."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import statistics
import subprocess
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BENCH_RE = re.compile(
    r"^BM_main_graph/process_time/real_time\s+"
    r"(?P<time>[0-9.]+)\s+(?P<unit>ns|us|ms|s)\s+"
    r"(?P<cpu>[0-9.]+)\s+(?P<cpu_unit>ns|us|ms|s)\s+"
    r"(?P<iters>[0-9]+).*items_per_second=(?P<ips>[0-9.eE+-]+)/s"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-name", default="qwen25_3b_b4_s32_fp16_cuda_standard")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seq", type=int, default=32)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--variants", nargs="+", choices=["baseline", "flatten_matmul"], default=["baseline", "flatten_matmul"])
    parser.add_argument("--cuda-async-allocations", choices=["default", "true", "false"], default="default")
    parser.add_argument("--per-run-timeout", type=int, default=300)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--stop-on-failure", action="store_true")
    return parser.parse_args()


def to_ms(value: float, unit: str) -> float:
    if unit == "ns":
        return value / 1_000_000.0
    if unit == "us":
        return value / 1_000.0
    if unit == "ms":
        return value
    if unit == "s":
        return value * 1000.0
    raise ValueError(unit)


def paths(exp_name: str, variant: str):
    exp = ROOT / "experiments" / exp_name
    if variant == "baseline":
        return {
            "module": exp / "standard_cuda" / "qwen_cuda.vmfb",
            "params": exp / "standard_cuda" / "qwen_params.irpa",
            "log_dir": exp / "logs" / "repeated_single_baseline",
        }
    return {
        "module": exp / "standard_cuda_flatten_matmul" / "qwen_flatten_cuda.vmfb",
        "params": exp / "standard_cuda_flatten_matmul" / "qwen_flatten_params.irpa",
        "log_dir": exp / "logs" / "repeated_single_flatten_matmul",
    }


def run_one(args, variant: str, index: int):
    ps = paths(args.experiment_name, variant)
    ps["log_dir"].mkdir(parents=True, exist_ok=True)
    log_path = ps["log_dir"] / f"run_{index:02d}.log"
    cmd = [
        "iree-benchmark-module",
        f"--module={ps['module']}",
        f"--parameters=model={ps['params']}",
        f"--device={args.device}",
        "--function=main_graph",
        f"--input={args.batch}x{args.seq}xi64=0",
        f"--input={args.batch}x{args.seq}xi64=1",
        "--benchmark_repetitions=1",
        "--benchmark_min_time=1x",
    ]
    if args.cuda_async_allocations != "default":
        cmd.insert(1, f"--cuda_async_allocations={args.cuda_async_allocations}")

    env = {**os.environ, "CUDA_VISIBLE_DEVICES": args.gpu}
    started = datetime.now()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            timeout=args.per_run_timeout,
        )
    except subprocess.TimeoutExpired as exc:
        log_path.write_text(
            "\n".join(
                [
                    f"Started: {started.isoformat()}",
                    f"Finished: {datetime.now().isoformat()}",
                    "Return code: timeout",
                    "Command:",
                    " ".join(shlex.quote(part) for part in cmd),
                    "",
                    "STDOUT:",
                    exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or ""),
                    "",
                    "STDERR:",
                    exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or ""),
                ]
            ),
            encoding="utf-8",
        )
        return {
            "variant": variant,
            "run": index,
            "status": "failed",
            "returncode": "timeout",
            "reason": f"timeout after {args.per_run_timeout}s",
            "log": str(log_path),
        }

    log_path.write_text(
        "\n".join(
            [
                f"Started: {started.isoformat()}",
                f"Finished: {datetime.now().isoformat()}",
                f"Return code: {result.returncode}",
                "Command:",
                " ".join(shlex.quote(part) for part in cmd),
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
    text = (result.stdout or "") + "\n" + (result.stderr or "")
    match = None
    for line in text.splitlines():
        m = BENCH_RE.match(line.strip())
        if m:
            match = m
            break
    if result.returncode != 0 or match is None:
        reason = "benchmark row not found"
        for line in text.splitlines():
            if "CUDA_ERROR" in line or "RESOURCE_EXHAUSTED" in line or "ABORTED" in line:
                reason = line.strip()
                break
        return {
            "variant": variant,
            "run": index,
            "status": "failed",
            "returncode": result.returncode,
            "reason": reason,
            "log": str(log_path),
        }

    return {
        "variant": variant,
        "run": index,
        "status": "ok",
        "latency_ms": to_ms(float(match.group("time")), match.group("unit")),
        "cpu_time_ms": to_ms(float(match.group("cpu")), match.group("cpu_unit")),
        "items_per_second": float(match.group("ips")),
        "iterations": int(match.group("iters")),
        "log": str(log_path),
    }


def aggregate(rows, args, variant: str):
    ok = [row for row in rows if row["variant"] == variant and row["status"] == "ok"]
    failed = [row for row in rows if row["variant"] == variant and row["status"] != "ok"]
    if not ok:
        return {"status": "failed", "successful_runs": 0, "failed_runs": len(failed), "failures": failed}
    latencies = [row["latency_ms"] for row in ok]
    ips = [row["items_per_second"] for row in ok]
    ips_mean = statistics.mean(ips)
    return {
        "status": "ok" if not failed else "partial",
        "successful_runs": len(ok),
        "failed_runs": len(failed),
        "latency_ms_mean": statistics.mean(latencies),
        "latency_ms_median": statistics.median(latencies),
        "latency_ms_min": min(latencies),
        "latency_ms_max": max(latencies),
        "latency_ms_stddev": statistics.stdev(latencies) if len(latencies) >= 2 else 0.0,
        "batch_iterations_per_second": ips_mean,
        "sequences_per_second": ips_mean * args.batch,
        "tokens_per_second": ips_mean * args.batch * args.seq,
        "failures": failed,
    }


def main():
    args = parse_args()
    exp = ROOT / "experiments" / args.experiment_name
    out = args.out or exp / "benchmark_repeated_single_compare.json"
    rows = []
    for variant in args.variants:
        for i in range(1, args.runs + 1):
            print(f"=== {variant} run {i}/{args.runs} ===")
            row = run_one(args, variant, i)
            rows.append(row)
            if row["status"] == "ok":
                print(f"{variant} run {i}: {row['latency_ms']:.3f} ms")
            else:
                print(f"{variant} run {i}: FAILED {row['reason']}")
                if args.stop_on_failure:
                    break

    performance = {variant: aggregate(rows, args, variant) for variant in args.variants}
    baseline = performance.get("baseline")
    flat = performance.get("flatten_matmul")
    if baseline and flat and baseline.get("successful_runs") and flat.get("successful_runs"):
        base_ms = baseline["latency_ms_mean"]
        flat_ms = flat["latency_ms_mean"]
        performance["speedup"] = base_ms / flat_ms if flat_ms else None
        performance["latency_reduction_percent"] = (base_ms - flat_ms) / base_ms * 100.0

    summary = {
        "experiment": args.experiment_name,
        "shape": {"batch": args.batch, "seq": args.seq},
        "runs_requested": args.runs,
        "gpu": args.gpu,
        "device": args.device,
        "rows": rows,
        "performance": performance,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(performance, indent=2))
    print(f"Saved summary: {out}")


if __name__ == "__main__":
    main()
