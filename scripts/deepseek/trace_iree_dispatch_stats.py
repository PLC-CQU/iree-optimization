#!/usr/bin/env python3
"""Run a traced IREE VMFB and save per-dispatch tensor statistics."""

import argparse
import json
from datetime import datetime
from pathlib import Path

import iree.runtime as rt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", type=Path, required=True)
    parser.add_argument("--params", type=Path, required=True)
    parser.add_argument("--function", default="main_graph")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--parameter-scope", default="model")
    parser.add_argument(
        "--max-elements-for-std",
        type=int,
        default=8_000_000,
        help="Skip std/min/max for tensors larger than this to keep tracing tolerable.",
    )
    return parser.parse_args()


def finite_stats(arr: np.ndarray, max_elements_for_std: int):
    arr64 = arr.astype(np.float64, copy=False) if arr.dtype.kind in "fiu" else arr
    row = {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "num_elements": int(arr.size),
        "sum": float(arr64.sum()),
        "mean": float(arr64.mean()) if arr.size else 0.0,
    }
    if arr.size <= max_elements_for_std:
        row.update(
            {
                "std": float(arr64.std()) if arr.size else 0.0,
                "min": float(arr64.min()) if arr.size else 0.0,
                "max": float(arr64.max()) if arr.size else 0.0,
            }
        )
    else:
        row.update({"std": None, "min": None, "max": None})
    return row


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    config = rt.Config(args.device)

    params = rt.ParameterIndex()
    params.load(str(args.params))
    provider = params.create_provider(scope=args.parameter_scope)

    records = []

    def callback(key: str, buffer_views: list[rt.HalBufferView]):
        for tensor_index, bv in enumerate(buffer_views):
            arr = bv.map().asarray(bv.shape, rt.HalElementType.map_to_dtype(bv.element_type))
            row = {
                "ordinal": len(records),
                "key": key,
                "tensor_index": tensor_index,
            }
            row.update(finite_stats(arr, args.max_elements_for_std))
            records.append(row)

    hal_module = rt.create_hal_module(
        config.vm_instance,
        config.device,
        debug_sink=rt.HalModuleDebugSink(callback),
    )
    io_module = rt.create_io_parameters_module(config.vm_instance, provider)
    model_module = rt.VmModule.copy_buffer(config.vm_instance, args.module.read_bytes())
    modules = rt.load_vm_modules(hal_module, io_module, model_module, config=config)

    inputs = [np.load(path) for path in args.input]
    fn = getattr(modules[-1], args.function)
    outputs = fn(*inputs)
    if not isinstance(outputs, tuple):
        outputs = (outputs,)
    output_stats = [finite_stats(np.asarray(out), args.max_elements_for_std) for out in outputs]

    summary = {
        "timestamp": datetime.now().isoformat(),
        "module": str(args.module),
        "params": str(args.params),
        "function": args.function,
        "device": args.device,
        "inputs": [str(path) for path in args.input],
        "num_trace_records": len(records),
        "output_stats": output_stats,
    }
    (args.out_dir / "trace_stats.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# IREE Dispatch Trace Stats",
        "",
        f"- Module: `{args.module}`",
        f"- Params: `{args.params}`",
        f"- Trace records: {len(records)}",
        "",
        "## First Records",
        "",
    ]
    for row in records[:200]:
        lines.append(
            f"- {row['ordinal']:04d} `{row['key']}`[{row['tensor_index']}] "
            f"shape={row['shape']} dtype={row['dtype']} "
            f"mean={row['mean']:.8g} sum={row['sum']:.8g}"
        )
    (args.out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Saved JSON: {args.out_dir / 'trace_stats.json'}")
    print(f"Saved report: {args.out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
