#!/usr/bin/env python3
"""Inline small ONNX dense_resource constants in iree-import-onnx MLIR.

iree-import-onnx can print ONNX attribute tensors as builtin dense_resource
entries. In the local IREE 3.11 toolchain, the Torch/ONNX importer does not
legalize torch.operator "onnx.Constant" when its torch.onnx.value is still a
dense_resource. This script rewrites those small resources to inline
DenseElementsAttr values while leaving stream.parameter globals untouched.
"""

import argparse
import math
import re
import struct
from pathlib import Path


RESOURCE_RE = re.compile(r"([A-Za-z0-9_.$-]+):\s+\"0x([0-9A-Fa-f]+)\"")
DENSE_RE = re.compile(r"dense_resource<([^>]+)>\s+:\s+tensor<([^>]+)>")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument(
        "--max-inline-bytes",
        type=int,
        default=1 << 20,
        help="Refuse to inline a single resource larger than this many payload bytes.",
    )
    return parser.parse_args()


def parse_resources(text: str) -> dict[str, bytes]:
    resources = {}
    for name, hex_data in RESOURCE_RE.findall(text):
        raw = bytes.fromhex(hex_data)
        if len(raw) < 4:
            continue
        # MLIR resource blobs printed by iree-import-onnx have a 4 byte header.
        resources[name] = raw[4:]
    return resources


def parse_tensor_type(type_text: str):
    parts = type_text.split("x")
    elem_type = parts[-1]
    dims = []
    for part in parts[:-1]:
        if part == "?":
            raise ValueError(f"dynamic resource tensor shape is not supported: tensor<{type_text}>")
        dims.append(int(part))
    return dims, elem_type


def element_count(dims: list[int]) -> int:
    return math.prod(dims) if dims else 1


def decode_payload(payload: bytes, elem_type: str, count: int):
    if elem_type == "si64":
        fmt = "<" + "q" * count
        return list(struct.unpack(fmt, payload[: 8 * count]))
    if elem_type == "i64":
        fmt = "<" + "q" * count
        return list(struct.unpack(fmt, payload[: 8 * count]))
    if elem_type == "ui64":
        fmt = "<" + "Q" * count
        return list(struct.unpack(fmt, payload[: 8 * count]))
    if elem_type == "si32" or elem_type == "i32":
        fmt = "<" + "i" * count
        return list(struct.unpack(fmt, payload[: 4 * count]))
    if elem_type == "ui32":
        fmt = "<" + "I" * count
        return list(struct.unpack(fmt, payload[: 4 * count]))
    if elem_type == "i1":
        return [bool(v) for v in payload[:count]]
    if elem_type == "f32":
        fmt = "<" + "f" * count
        return list(struct.unpack(fmt, payload[: 4 * count]))
    if elem_type == "f16":
        return [struct.unpack("<e", payload[i : i + 2])[0] for i in range(0, 2 * count, 2)]
    raise ValueError(f"unsupported resource element type: {elem_type}")


def format_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if math.isinf(value):
        return "0x7F800000" if value > 0 else "0xFF800000"
    if math.isnan(value):
        return "0x7FC00000"
    text = format(value, ".9g")
    if "e" not in text and "E" not in text and "." not in text:
        text += ".0"
    return text


def format_nested(values, dims):
    if not dims:
        return format_value(values[0])
    if len(dims) == 1:
        return "[" + ", ".join(format_value(value) for value in values) + "]"
    stride = math.prod(dims[1:])
    chunks = [
        format_nested(values[index : index + stride], dims[1:])
        for index in range(0, len(values), stride)
    ]
    return "[" + ", ".join(chunks) + "]"


def format_dense(values, dims):
    if len(values) == 1:
        return f"dense<{format_value(values[0])}>"
    if all(value == values[0] for value in values):
        return f"dense<{format_value(values[0])}>"
    return f"dense<{format_nested(values, dims)}>"


def rewrite(text: str, resources: dict[str, bytes], max_inline_bytes: int):
    missing = set()
    rewritten = 0

    def replace(match: re.Match):
        nonlocal rewritten
        name = match.group(1)
        type_text = match.group(2)
        payload = resources.get(name)
        if payload is None:
            missing.add(name)
            return match.group(0)
        if len(payload) > max_inline_bytes:
            return match.group(0)

        dims, elem_type = parse_tensor_type(type_text)
        count = element_count(dims)
        if count == 0:
            return match.group(0)
        values = decode_payload(payload, elem_type, count)
        rewritten += 1
        return f"{format_dense(values, dims)} : tensor<{type_text}>"

    # Only rewrite uses in the module body. The trailing resource block can stay.
    rewritten_text = DENSE_RE.sub(replace, text)
    return rewritten_text, rewritten, missing


def main():
    args = parse_args()
    text = args.input.read_text(encoding="utf-8")
    resources = parse_resources(text)
    output, rewritten, missing = rewrite(text, resources, args.max_inline_bytes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output, encoding="utf-8")
    print(f"resources found: {len(resources)}")
    print(f"dense_resource uses rewritten: {rewritten}")
    if missing:
        print(f"missing resource definitions: {len(missing)}")
    print(f"wrote: {args.output}")


if __name__ == "__main__":
    main()
