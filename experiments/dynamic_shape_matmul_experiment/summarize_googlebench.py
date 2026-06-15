#!/usr/bin/env python3
import json
import sys
from pathlib import Path


def pick(benchmarks, suffix):
    for item in benchmarks:
        if item.get("name", "").endswith(suffix):
            return item
    return None


def main(argv):
    if len(argv) < 2:
        print("usage: summarize_googlebench.py FILE.googlebench.json ...", file=sys.stderr)
        return 1

    rows = []
    for path_str in argv[1:]:
        path = Path(path_str)
        with path.open() as f:
            data = json.load(f)
        benchmarks = data.get("benchmarks", [])
        mean = pick(benchmarks, "_mean")
        median = pick(benchmarks, "_median")
        stddev = pick(benchmarks, "_stddev")
        cv = pick(benchmarks, "_cv")
        rows.append(
            {
                "name": path.name.replace(".googlebench.json", ""),
                "mean_us": mean.get("real_time") if mean else None,
                "median_us": median.get("real_time") if median else None,
                "stddev_us": stddev.get("real_time") if stddev else None,
                "cv_pct": cv.get("real_time") if cv else None,
            }
        )

    baseline = next((r for r in rows if r["name"] == "dynamic"), None)
    print("name,mean_us,median_us,stddev_us,cv_pct,speedup_vs_dynamic_mean")
    for row in rows:
        speedup = ""
        if baseline and row["mean_us"]:
            speedup = baseline["mean_us"] / row["mean_us"]
        print(
            f"{row['name']},{row['mean_us']},{row['median_us']},"
            f"{row['stddev_us']},{row['cv_pct']},{speedup}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
