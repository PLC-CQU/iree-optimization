#!/usr/bin/env python3
"""Rewrite rank-3 activation MatMul nodes into explicit rank-2 MatMul nodes.

The exported Llama graph contains many nodes of the form:

  MatMul([B, S, K], [K, N]) -> [B, S, N]

ONNX semantics allow the RHS to be broadcast across the leading dimensions.
Some compiler paths may lower this as a batched contraction with large
broadcasted/read-only tensors. This rewrite makes the intended layout explicit:

  Reshape([B, S, K] -> [B*S, K])
  MatMul([B*S, K], [K, N]) -> [B*S, N]
  Reshape([B*S, N] -> [B, S, N])

Attention MatMul nodes are intentionally left untouched because their RHS is a
runtime tensor, not a constant weight initializer.
"""

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path

import onnx
from onnx import TensorProto, helper


@dataclass
class RewriteRecord:
    node_name: str
    output_name: str
    lhs: str
    rhs: str
    original_lhs_shape: list[int | None]
    rhs_shape: list[int]
    original_output_shape: list[int | None]
    flattened_lhs_shape: list[int | str]
    flattened_output_shape: list[int | str]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--seq", type=int, default=None)
    parser.add_argument(
        "--auto-shape",
        action="store_true",
        help="Infer batch and sequence length from a rank-2 graph input if --batch/--seq are omitted.",
    )
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Analyze and report without writing an output ONNX.")
    parser.add_argument(
        "--dynamic-shape",
        action="store_true",
        help="Rewrite dynamic [B,S,K] MatMul by constructing Reshape shapes with ONNX Shape/Gather/Mul/Concat.",
    )
    parser.add_argument("--report", type=Path, help="Write a JSON rewrite report.")
    parser.add_argument(
        "--max-report-records",
        type=int,
        default=32,
        help="Maximum rewritten/skipped examples stored in the JSON report.",
    )
    return parser.parse_args()


def sanitize_name(name: str) -> str:
    name = re.sub(r"[^0-9A-Za-z_]+", "_", name)
    return name.strip("_") or "matmul"


def dims_from_value_info(value_info) -> list[int | None]:
    dims = []
    for dim in value_info.type.tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            dims.append(dim.dim_value)
        else:
            dims.append(None)
    return dims


def build_shape_map(model) -> dict[str, list[int | None]]:
    shape_map: dict[str, list[int | None]] = {}
    for value in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
        if value.type.HasField("tensor_type") and value.type.tensor_type.HasField("shape"):
            shape_map[value.name] = dims_from_value_info(value)
    return shape_map


def build_initializer_shape_map(model) -> dict[str, list[int]]:
    return {initializer.name: list(initializer.dims) for initializer in model.graph.initializer}


def infer_batch_seq(model) -> tuple[int, int] | None:
    """Infer [B, S] from the first statically shaped rank-2 graph input."""
    for graph_input in model.graph.input:
        if not graph_input.type.HasField("tensor_type"):
            continue
        tensor_type = graph_input.type.tensor_type
        if not tensor_type.HasField("shape"):
            continue
        dims = dims_from_value_info(graph_input)
        if len(dims) == 2 and dims[0] is not None and dims[1] is not None:
            return int(dims[0]), int(dims[1])
    return None


def make_shape_const(name: str, dims: list[int]):
    tensor = helper.make_tensor(
        name=f"{name}_value",
        data_type=TensorProto.INT64,
        dims=[len(dims)],
        vals=dims,
    )
    return helper.make_node("Constant", inputs=[], outputs=[name], name=f"{name}_const", value=tensor)


def should_rewrite(node, shape_map, initializer_shapes, batch: int | None, seq: int | None, dynamic_shape: bool):
    if node.op_type != "MatMul" or len(node.input) != 2 or len(node.output) != 1:
        return None, "not_candidate_matmul"

    lhs_name, rhs_name = node.input
    if rhs_name not in initializer_shapes:
        return None, "rhs_not_initializer"

    lhs_shape = shape_map.get(lhs_name)
    rhs_shape = initializer_shapes.get(rhs_name)
    out_shape = shape_map.get(node.output[0])

    if not lhs_shape or not rhs_shape or not out_shape:
        return None, "missing_shape"
    if len(lhs_shape) != 3 or len(rhs_shape) != 2 or len(out_shape) != 3:
        return None, "rank_mismatch"

    if not dynamic_shape:
        if lhs_shape[0] != batch or lhs_shape[1] != seq:
            return None, "lhs_batch_seq_mismatch"
        if out_shape[0] != batch or out_shape[1] != seq:
            return None, "output_batch_seq_mismatch"
    elif batch is not None and seq is not None:
        if lhs_shape[0] not in (None, batch) or lhs_shape[1] not in (None, seq):
            return None, "lhs_batch_seq_mismatch"
        if out_shape[0] not in (None, batch) or out_shape[1] not in (None, seq):
            return None, "output_batch_seq_mismatch"

    if lhs_shape[2] is None or rhs_shape[0] is None or rhs_shape[1] is None:
        return None, "dynamic_contracting_or_output_dim"
    if out_shape[2] is not None and rhs_shape[1] != out_shape[2]:
        return None, "contracting_or_output_dim_mismatch"
    if lhs_shape[2] != rhs_shape[0]:
        return None, "contracting_or_output_dim_mismatch"

    return {
        "lhs_k": int(lhs_shape[2]),
        "rhs_n": int(rhs_shape[1]),
        "out_n": int(rhs_shape[1]),
        "lhs_shape": list(lhs_shape),
        "rhs_shape": list(rhs_shape),
        "out_shape": list(out_shape),
    }, "rewrite"


def lookup_tensor_shape(name: str, shape_map, initializer_shapes):
    if name in initializer_shapes:
        return initializer_shapes[name]
    return shape_map.get(name)


def make_scalar_shape_const(name: str, value: int):
    return make_shape_const(name, [value])


def make_dynamic_shape_nodes(base: str, lhs_name: str, k_dim: int, n_dim: int):
    lhs_shape = f"{base}_lhs_runtime_shape"
    b_index = f"{base}_b_index"
    s_index = f"{base}_s_index"
    b_dim = f"{base}_b_dim"
    s_dim = f"{base}_s_dim"
    bs_dim = f"{base}_bs_dim"
    k_const = f"{base}_k_dim"
    n_const = f"{base}_n_dim"
    flat_shape = f"{base}_flatten_lhs_shape"
    restore_shape = f"{base}_restore_out_shape"

    return [
        helper.make_node("Shape", inputs=[lhs_name], outputs=[lhs_shape], name=f"{base}_shape_lhs"),
        make_shape_const(b_index, [0]),
        helper.make_node("Gather", inputs=[lhs_shape, b_index], outputs=[b_dim], name=f"{base}_gather_b", axis=0),
        make_shape_const(s_index, [1]),
        helper.make_node("Gather", inputs=[lhs_shape, s_index], outputs=[s_dim], name=f"{base}_gather_s", axis=0),
        helper.make_node("Mul", inputs=[b_dim, s_dim], outputs=[bs_dim], name=f"{base}_mul_bs"),
        make_scalar_shape_const(k_const, k_dim),
        helper.make_node("Concat", inputs=[bs_dim, k_const], outputs=[flat_shape], name=f"{base}_concat_flat_shape", axis=0),
        make_scalar_shape_const(n_const, n_dim),
        helper.make_node(
            "Concat",
            inputs=[b_dim, s_dim, n_const],
            outputs=[restore_shape],
            name=f"{base}_concat_restore_shape",
            axis=0,
        ),
    ], flat_shape, restore_shape


def rewrite_model(
    model,
    batch: int | None,
    seq: int | None,
    dry_run: bool = False,
    max_report_records: int = 32,
    dynamic_shape: bool = False,
):
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=False, data_prop=False)
    shape_map = build_shape_map(inferred)
    initializer_shapes = build_initializer_shape_map(model)

    rewritten_nodes = []
    rewritten_count = 0
    matmul_total = 0
    weight_matmul_total = 0
    skip_reasons = Counter()
    rewritten_records: list[RewriteRecord] = []
    skipped_examples = []

    for index, node in enumerate(model.graph.node):
        if node.op_type == "MatMul":
            matmul_total += 1
            if len(node.input) == 2 and node.input[1] in initializer_shapes:
                weight_matmul_total += 1

        rewrite_info, reason = should_rewrite(node, shape_map, initializer_shapes, batch, seq, dynamic_shape)
        if rewrite_info is None:
            if node.op_type == "MatMul":
                skip_reasons[reason] += 1
                if len(skipped_examples) < max_report_records:
                    skipped_examples.append(
                        {
                            "node_name": node.name,
                            "output_name": node.output[0] if node.output else "",
                            "reason": reason,
                            "lhs_shape": shape_map.get(node.input[0]) if len(node.input) >= 1 else None,
                            "rhs_shape": lookup_tensor_shape(node.input[1], shape_map, initializer_shapes)
                            if len(node.input) >= 2
                            else None,
                            "output_shape": shape_map.get(node.output[0]) if node.output else None,
                            "rhs_is_initializer": len(node.input) >= 2 and node.input[1] in initializer_shapes,
                        }
                    )
            rewritten_nodes.append(node)
            continue

        base = sanitize_name(node.name or node.output[0] or f"matmul_{index}")
        lhs_name, rhs_name = node.input
        out_name = node.output[0]
        flat_lhs = f"{base}_flatten_lhs"
        flat_out = f"{base}_flatten_out"
        if dynamic_shape:
            shape_nodes, lhs_shape_name, out_shape_name = make_dynamic_shape_nodes(
                base,
                lhs_name,
                rewrite_info["lhs_k"],
                rewrite_info["out_n"],
            )
            flattened_lhs_shape = ["B*S", rewrite_info["lhs_k"]]
            flattened_output_shape = ["B*S", rewrite_info["out_n"]]
        else:
            lhs_shape_name = f"{base}_flatten_lhs_shape"
            out_shape_name = f"{base}_restore_out_shape"
            shape_nodes = [
                make_shape_const(lhs_shape_name, [batch * seq, rewrite_info["lhs_k"]]),
                make_shape_const(out_shape_name, [batch, seq, rewrite_info["out_n"]]),
            ]
            flattened_lhs_shape = [batch * seq, rewrite_info["lhs_k"]]
            flattened_output_shape = [batch * seq, rewrite_info["out_n"]]

        if dynamic_shape:
            new_nodes = list(shape_nodes)
        else:
            new_nodes = [shape_nodes[0]]
        new_nodes.extend(
            [
                helper.make_node(
                    "Reshape",
                    inputs=[lhs_name, lhs_shape_name],
                    outputs=[flat_lhs],
                    name=f"{base}_flatten_lhs",
                ),
                helper.make_node(
                    "MatMul",
                    inputs=[flat_lhs, rhs_name],
                    outputs=[flat_out],
                    name=f"{base}_flat_matmul",
                ),
            ]
        )
        if not dynamic_shape:
            new_nodes.append(shape_nodes[1])
        new_nodes.append(
            helper.make_node(
                "Reshape",
                inputs=[flat_out, out_shape_name],
                outputs=[out_name],
                name=f"{base}_restore_out",
            )
        )
        rewritten_nodes.extend(new_nodes)
        if len(rewritten_records) < max_report_records:
            rewritten_records.append(
                RewriteRecord(
                    node_name=node.name,
                    output_name=out_name,
                    lhs=lhs_name,
                    rhs=rhs_name,
                    original_lhs_shape=rewrite_info["lhs_shape"],
                    rhs_shape=rewrite_info["rhs_shape"],
                    original_output_shape=rewrite_info["out_shape"],
                    flattened_lhs_shape=flattened_lhs_shape,
                    flattened_output_shape=flattened_output_shape,
                )
            )
        rewritten_count += 1

    if not dry_run:
        del model.graph.node[:]
        model.graph.node.extend(rewritten_nodes)

    report = {
        "input_batch": batch,
        "input_seq": seq,
        "shape_mode": "dynamic" if dynamic_shape else "static",
        "matmul_total": matmul_total,
        "weight_matmul_total": weight_matmul_total,
        "rewritten_weight_matmul_nodes": rewritten_count,
        "skipped_matmul_by_reason": dict(skip_reasons),
        "inserted_nodes_per_rewrite": 13 if dynamic_shape else 5,
        "estimated_inserted_nodes": rewritten_count * (13 if dynamic_shape else 5),
        "estimated_removed_nodes": rewritten_count,
        "estimated_net_node_delta": rewritten_count * (12 if dynamic_shape else 4),
        "rewrite_rule": {
            "before": "MatMul([B,S,K], [K,N]) -> [B,S,N]",
            "after": "Reshape([B,S,K] -> [B*S,K]) -> MatMul([B*S,K], [K,N]) -> Reshape([B*S,N] -> [B,S,N])",
            "safety_condition": "Only rewrite MatMul whose RHS is an initializer weight and whose shapes match [B,S,K] x [K,N] -> [B,S,N]. Dynamic mode constructs B*S with ONNX shape operations.",
        },
        "rewritten_examples": [asdict(record) for record in rewritten_records],
        "skipped_examples": skipped_examples,
    }
    return report


def main():
    args = parse_args()
    model = onnx.load(args.input, load_external_data=False)
    batch = args.batch
    seq = args.seq
    if (batch is None or seq is None) and args.auto_shape:
        inferred_shape = infer_batch_seq(model)
        if inferred_shape is None and not args.dynamic_shape:
            raise RuntimeError("Failed to infer batch/seq from graph inputs; pass --batch and --seq explicitly.")
        if inferred_shape is not None:
            batch, seq = inferred_shape
    if (batch is None or seq is None) and not args.dynamic_shape:
        raise RuntimeError("Batch and sequence length are required. Pass --batch/--seq or use --auto-shape.")

    report = rewrite_model(
        model,
        batch,
        seq,
        dry_run=args.dry_run,
        max_report_records=args.max_report_records,
        dynamic_shape=args.dynamic_shape,
    )

    if not args.dry_run:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        onnx.save(model, args.output)

    if args.check and not args.dry_run:
        onnx.checker.check_model(str(args.output))

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Input: {args.input}")
    print(f"Output: {args.output if not args.dry_run else '(dry run; not written)'}")
    print(f"Batch: {batch}")
    print(f"Seq: {seq}")
    print(f"Total MatMul nodes: {report['matmul_total']}")
    print(f"Weight MatMul nodes: {report['weight_matmul_total']}")
    print(f"Rewritten weight MatMul nodes: {report['rewritten_weight_matmul_nodes']}")
    print(f"Skipped MatMul by reason: {report['skipped_matmul_by_reason']}")
    if args.report:
        print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
