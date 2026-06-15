#!/usr/bin/env python3
"""Export a DeepSeek/Llama last-token ONNX graph with dynamic batch and sequence.

The fixed-shape exporter in this repo bakes the sampled batch/sequence into the
graph. This companion exporter keeps both input dimensions symbolic so a single
dynamic VMFB can be benchmarked against real request shapes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="./model")
    parser.add_argument("--out", default="dynamic_shape_b_s/deepseek_dynamic_b_s_last.onnx")
    parser.add_argument("--sample-batch", type=int, default=1)
    parser.add_argument("--sample-seq", type=int, default=32)
    parser.add_argument("--device", default=None)
    parser.add_argument("--last-token-only", action="store_true")
    parser.add_argument("--opset", type=int, default=17)
    return parser.parse_args()


class DynamicLastTokenWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, last_token_only: bool):
        super().__init__()
        self.model = model
        self.last_token_only = last_token_only

    def forward(self, input_ids, attention_mask):
        position_ids = attention_mask.to(torch.long).cumsum(dim=-1) - 1
        position_ids = position_ids.masked_fill(attention_mask == 0, 0)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            return_dict=False,
        )
        logits = outputs[0]
        if self.last_token_only:
            return logits[:, -1, :]
        return logits


def main():
    args = parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    model.config.use_cache = False
    model.eval()

    device = torch.device(args.device) if args.device else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    print("Using device:", device)
    model.to(device)

    wrapper = DynamicLastTokenWrapper(model, args.last_token_only).eval()
    input_ids = torch.zeros((args.sample_batch, args.sample_seq), dtype=torch.long, device=device)
    attention_mask = torch.ones((args.sample_batch, args.sample_seq), dtype=torch.long, device=device)

    output_axes = {0: "batch"}
    if not args.last_token_only:
        output_axes[1] = "seq"

    print("Exporting dynamic ONNX to", out)
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (input_ids, attention_mask),
            str(out),
            export_params=True,
            opset_version=args.opset,
            do_constant_folding=True,
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq"},
                "attention_mask": {0: "batch", 1: "seq"},
                "logits": output_axes,
            },
        )
    print("ONNX exported:", out)


if __name__ == "__main__":
    main()
