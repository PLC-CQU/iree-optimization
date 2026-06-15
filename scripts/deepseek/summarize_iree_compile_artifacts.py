#!/usr/bin/env python3
"""Summarize IREE executable configuration dumps."""

import argparse
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path


EXEC_RE = re.compile(r"hal\.executable(?:\s+public)?\s+@([A-Za-z0-9_.$-]+)")
VARIANT_RE = re.compile(r"hal\.executable\.variant(?:\s+public)?\s+@([A-Za-z0-9_.$-]+)")
TRANSLATION_RE = re.compile(r"translation_info\s*=\s*([^,\]}]+)")
WORKGROUP_RE = re.compile(r"workgroup_size\s*=\s*\[([^\]]+)\]")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def summarize_mlir(path: Path):
    text = path.read_text(encoding="utf-8", errors="replace")
    executables = EXEC_RE.findall(text)
    variants = VARIANT_RE.findall(text)
    translations = [item.strip() for item in TRANSLATION_RE.findall(text)]
    workgroups = [item.strip() for item in WORKGROUP_RE.findall(text)]

    lines = text.splitlines()
    excerpt_rows = []
    for i, line in enumerate(lines):
        if "hal.executable.variant" not in line:
            continue
        window = "\n".join(lines[i : min(i + 8, len(lines))])
        excerpt_rows.append(
            {
                "line": i + 1,
                "variant_line": line.strip(),
                "translation_info": (TRANSLATION_RE.search(window).group(1).strip() if TRANSLATION_RE.search(window) else None),
                "workgroup_size": (WORKGROUP_RE.search(window).group(1).strip() if WORKGROUP_RE.search(window) else None),
            }
        )

    return {
        "timestamp": datetime.now().isoformat(),
        "mlir": str(path),
        "num_executables": len(executables),
        "num_variants": len(variants),
        "translation_info_counts": dict(Counter(translations)),
        "workgroup_size_counts": dict(Counter(workgroups)),
        "variants": excerpt_rows,
    }


def write_report(summary, out_dir: Path):
    lines = [
        "# IREE Compile Artifact Summary",
        "",
        f"- MLIR: `{summary['mlir']}`",
        f"- Executables: {summary['num_executables']}",
        f"- Variants: {summary['num_variants']}",
        "",
        "## Translation Info Counts",
        "",
    ]
    for key, count in sorted(summary["translation_info_counts"].items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- `{key}`: {count}")
    lines.extend(["", "## Workgroup Size Counts", ""])
    for key, count in sorted(summary["workgroup_size_counts"].items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- `{key}`: {count}")
    lines.extend(["", "## Variants", ""])
    for row in summary["variants"]:
        lines.append(
            f"- line {row['line']}: `{row['variant_line']}`; "
            f"translation=`{row['translation_info']}`; workgroup=`{row['workgroup_size']}`"
        )
    report = "\n".join(lines) + "\n"
    (out_dir / "summary.md").write_text(report, encoding="utf-8")


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_mlir(args.mlir)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(summary, args.out_dir)
    print(json.dumps({k: summary[k] for k in ("num_executables", "num_variants")}, indent=2))
    print(f"Saved JSON: {args.out_dir / 'summary.json'}")
    print(f"Saved report: {args.out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
