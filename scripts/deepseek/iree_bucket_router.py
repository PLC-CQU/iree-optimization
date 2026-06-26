#!/usr/bin/env python3
"""Route dynamic requests to precompiled static IREE bucket VMFBs.

This is a deployment-side router: it does not rewrite model IR. The IREE
compiler pass optimizes each bucket module; this script chooses which optimized
bucket to run for a concrete request shape.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Bucket:
    batch: int
    seq: int
    module: Path
    params: Path
    function: str = "main_graph"
    output_token: str = "index"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--input-ids", type=Path, required=True)
    parser.add_argument("--attention-mask", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--iree-run-module", default="iree-run-module")
    parser.add_argument("--pad-token-id", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_path(base: Path, path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else base / path


def load_manifest(path: Path) -> tuple[Path, list[Bucket]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    base = resolve_path(path.parent, data.get("base_dir", ".")).resolve()
    buckets = []
    for item in data["buckets"]:
        buckets.append(
            Bucket(
                batch=int(item["batch"]),
                seq=int(item["seq"]),
                module=resolve_path(base, item["module"]),
                params=resolve_path(base, item["params"]),
                function=item.get("function", data.get("function", "main_graph")),
                output_token=item.get("output_token", data.get("output_token", "index")),
            )
        )
    return base, buckets


def choose_bucket(batch: int, seq: int, buckets: list[Bucket]) -> Bucket:
    candidates = [bucket for bucket in buckets if bucket.batch >= batch and bucket.seq >= seq]
    if not candidates:
        shapes = ", ".join(f"b{bucket.batch}_s{bucket.seq}" for bucket in buckets)
        raise ValueError(f"no bucket can hold b{batch}_s{seq}; available buckets: {shapes}")
    return min(candidates, key=lambda bucket: (bucket.batch * bucket.seq, bucket.batch, bucket.seq))


def pad_inputs(input_ids: np.ndarray, attention_mask: np.ndarray, bucket: Bucket, pad_token_id: int):
    if input_ids.ndim != 2 or attention_mask.ndim != 2:
        raise ValueError("input_ids and attention_mask must be rank-2 arrays [batch, seq]")
    if input_ids.shape != attention_mask.shape:
        raise ValueError(f"input shape mismatch: {input_ids.shape} vs {attention_mask.shape}")

    batch, seq = input_ids.shape
    if batch > bucket.batch or seq > bucket.seq:
        raise ValueError(f"request b{batch}_s{seq} does not fit bucket b{bucket.batch}_s{bucket.seq}")

    bucket_ids = np.full((bucket.batch, bucket.seq), pad_token_id, dtype=input_ids.dtype)
    bucket_mask = np.zeros((bucket.batch, bucket.seq), dtype=attention_mask.dtype)
    bucket_ids[:batch, :seq] = input_ids
    bucket_mask[:batch, :seq] = attention_mask

    last_token_indices = np.zeros((bucket.batch,), dtype=input_ids.dtype)
    lengths = attention_mask.astype(bool).sum(axis=1)
    last_token_indices[:batch] = np.maximum(lengths - 1, 0)
    return bucket_ids, bucket_mask, last_token_indices


def run_bucket(args, bucket: Bucket, inputs: dict[str, Path], raw_out: Path) -> list[str]:
    cmd = [
        args.iree_run_module,
        f"--module={bucket.module}",
        f"--parameters=model={bucket.params}",
        "--parameter_mode=file",
        f"--device={args.device}",
        f"--function={bucket.function}",
        f"--input=@{inputs['input_ids']}",
        f"--input=@{inputs['attention_mask']}",
    ]
    if bucket.output_token == "index":
        cmd.append(f"--input=@{inputs['last_token_indices']}")
    cmd.append(f"--output=@{raw_out}")

    print("Selected bucket:", f"b{bucket.batch}_s{bucket.seq}")
    print("Command:")
    print("  " + " \\\n  ".join(shlex.quote(str(part)) for part in cmd))
    if args.dry_run:
        return cmd

    env = os.environ.copy()
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.gpu
    subprocess.run([str(part) for part in cmd], check=True, env=env)
    return cmd


def load_iree_output(path: Path) -> np.ndarray:
    arr = np.load(path)
    if arr.dtype == np.uint8:
        arr = arr.view(np.float32)
    return arr


def crop_output(arr: np.ndarray, request_batch: int, request_seq: int) -> np.ndarray:
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2:
        return arr[:request_batch]
    if arr.ndim >= 3:
        return arr[:request_batch, :request_seq]
    return arr


def main():
    args = parse_args()
    _, buckets = load_manifest(args.manifest)
    input_ids = np.load(args.input_ids)
    attention_mask = np.load(args.attention_mask)
    batch, seq = input_ids.shape
    bucket = choose_bucket(batch, seq, buckets)
    bucket_ids, bucket_mask, last_token_indices = pad_inputs(
        input_ids, attention_mask, bucket, args.pad_token_id
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="iree_bucket_router_") as tmp:
        tmpdir = Path(tmp)
        paths = {
            "input_ids": tmpdir / "bucket_input_ids.npy",
            "attention_mask": tmpdir / "bucket_attention_mask.npy",
            "last_token_indices": tmpdir / "bucket_last_token_indices.npy",
        }
        np.save(paths["input_ids"], bucket_ids)
        np.save(paths["attention_mask"], bucket_mask)
        np.save(paths["last_token_indices"], last_token_indices)
        raw_out = tmpdir / "bucket_output.npy"
        run_bucket(args, bucket, paths, raw_out)
        if args.dry_run:
            return
        output = crop_output(load_iree_output(raw_out), batch, seq)
        np.save(args.out, output)
        print(f"Saved routed output: {args.out}")


if __name__ == "__main__":
    main()
