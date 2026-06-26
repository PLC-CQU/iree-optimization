#!/usr/bin/env python3
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path


def parse_shape(shape):
    match = re.fullmatch(r"(\d+)x(\d+)x(\d+)", shape)
    if not match:
        raise ValueError(f"expected BxSxK shape, got {shape!r}")
    return tuple(int(x) for x in match.groups())


def run(cmd, dry_run=False):
    print("+ " + " ".join(str(x) for x in cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def write_static_rank3_matmul(path, b, s, k, n):
    path.write_text(
        f"""module {{
  func.func @main(%lhs: tensor<{b}x{s}x{k}xf32>,
                  %rhs: tensor<{k}x{n}xf32>) -> tensor<{b}x{s}x{n}xf32> {{
    %zero = arith.constant 0.0 : f32
    %init = tensor.empty() : tensor<{b}x{s}x{n}xf32>
    %filled = linalg.fill ins(%zero : f32)
      outs(%init : tensor<{b}x{s}x{n}xf32>) -> tensor<{b}x{s}x{n}xf32>
    %out = linalg.generic {{
      indexing_maps = [
        affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>,
        affine_map<(d0, d1, d2, d3) -> (d3, d2)>,
        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
      ],
      iterator_types = ["parallel", "parallel", "parallel", "reduction"]
    }} ins(%lhs, %rhs : tensor<{b}x{s}x{k}xf32>, tensor<{k}x{n}xf32>)
      outs(%filled : tensor<{b}x{s}x{n}xf32>) {{
    ^bb0(%l: f32, %r: f32, %acc: f32):
      %mul = arith.mulf %l, %r : f32
      %add = arith.addf %mul, %acc : f32
      linalg.yield %add : f32
    }} -> tensor<{b}x{s}x{n}xf32>
    return %out : tensor<{b}x{s}x{n}xf32>
  }}
}}
""",
        encoding="utf-8",
    )


def write_dynamic_rank3_matmul(path, k, n):
    path.write_text(
        f"""module {{
  func.func @main(%lhs: tensor<?x?x{k}xf32>,
                  %rhs: tensor<{k}x{n}xf32>) -> tensor<?x?x{n}xf32> {{
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0.0 : f32
    %b = tensor.dim %lhs, %c0 : tensor<?x?x{k}xf32>
    %s = tensor.dim %lhs, %c1 : tensor<?x?x{k}xf32>
    %init = tensor.empty(%b, %s) : tensor<?x?x{n}xf32>
    %filled = linalg.fill ins(%zero : f32)
      outs(%init : tensor<?x?x{n}xf32>) -> tensor<?x?x{n}xf32>
    %out = linalg.generic {{
      indexing_maps = [
        affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>,
        affine_map<(d0, d1, d2, d3) -> (d3, d2)>,
        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
      ],
      iterator_types = ["parallel", "parallel", "parallel", "reduction"]
    }} ins(%lhs, %rhs : tensor<?x?x{k}xf32>, tensor<{k}x{n}xf32>)
      outs(%filled : tensor<?x?x{n}xf32>) {{
    ^bb0(%l: f32, %r: f32, %acc: f32):
      %mul = arith.mulf %l, %r : f32
      %add = arith.addf %mul, %acc : f32
      linalg.yield %add : f32
    }} -> tensor<?x?x{n}xf32>
    return %out : tensor<?x?x{n}xf32>
  }}
}}
""",
        encoding="utf-8",
    )


def compile_mlir(args, input_mlir, output_vmfb):
    if output_vmfb.exists() and not args.force:
        print(f"cache hit: {output_vmfb}")
        return
    cmd = [
        args.iree_compile,
        "--iree-hal-target-backends=cuda",
        f"--iree-cuda-target={args.cuda_arch}",
        f"--iree-gpu-test-target={args.cuda_arch}",
        str(input_mlir),
        "-o",
        str(output_vmfb),
    ]
    run(cmd, args.dry_run)


def inspect_codegen(args, input_mlir, output_mlir):
    cmd = [
        args.iree_compile,
        "--iree-hal-target-backends=cuda",
        f"--iree-cuda-target={args.cuda_arch}",
        f"--iree-gpu-test-target={args.cuda_arch}",
        "--compile-to=executable-configurations",
        str(input_mlir),
        "-o",
        str(output_mlir),
    ]
    run(cmd, args.dry_run)
    if args.dry_run:
        return
    text = output_mlir.read_text(encoding="utf-8")
    print(f"TileAndFuse={text.count('TileAndFuse')}")
    print(f"VectorDistribute={text.count('VectorDistribute')}")
    for line in text.splitlines():
        if "hal.executable.export public" in line or "translation_info" in line:
            print(line.strip())


def benchmark(args, vmfb, b, s, k, n):
    cmd = [
        args.iree_benchmark_module,
        f"--module={vmfb}",
        "--device=cuda",
        "--function=main",
        f"--input={b}x{s}x{k}xf32=1",
        f"--input={k}x{n}xf32=1",
        f"--benchmark_repetitions={args.benchmark_repetitions}",
        f"--benchmark_min_time={args.benchmark_min_time}",
        "--benchmark_time_unit=us",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    print("+ " + " ".join(cmd))
    if not args.dry_run:
        subprocess.run(cmd, check=True, env=env)


def main(argv):
    parser = argparse.ArgumentParser(
        description="Compile and cache a static IREE module for the runtime observed rank3 matmul shape."
    )
    parser.add_argument("--lhs-shape", required=True, help="Runtime LHS shape as BxSxK, for example 4x128x256")
    parser.add_argument("--n", type=int, required=True, help="RHS/output N dimension")
    parser.add_argument("--cache-dir", default="/tmp/iree_shape_compile_cache")
    parser.add_argument("--iree-compile", default="/home/zhongjialin/projects/iree-build/tools/iree-compile")
    parser.add_argument("--iree-benchmark-module", default="/home/zhongjialin/projects/iree-build/tools/iree-benchmark-module")
    parser.add_argument("--cuda-arch", default="sm_86")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--compile-dynamic-baseline", action="store_true")
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--benchmark-repetitions", default="5")
    parser.add_argument("--benchmark-min-time", default="1s")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    b, s, k = parse_shape(args.lhs_shape)
    n = args.n
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    key = f"rank3_matmul_b{b}_s{s}_k{k}_n{n}_{args.cuda_arch}"
    static_mlir = cache_dir / f"{key}.mlir"
    static_vmfb = cache_dir / f"{key}.vmfb"
    static_config = cache_dir / f"{key}_exec_config.mlir"
    write_static_rank3_matmul(static_mlir, b, s, k, n)
    compile_mlir(args, static_mlir, static_vmfb)

    result = {
        "runtime_shape": {"b": b, "s": s, "k": k, "n": n},
        "static_mlir": str(static_mlir),
        "static_vmfb": str(static_vmfb),
    }

    if args.inspect:
        inspect_codegen(args, static_mlir, static_config)
        result["static_exec_config"] = str(static_config)

    if args.benchmark:
        benchmark(args, static_vmfb, b, s, k, n)

    if args.compile_dynamic_baseline:
        dyn_key = f"rank3_matmul_dynamic_k{k}_n{n}_{args.cuda_arch}"
        dynamic_mlir = cache_dir / f"{dyn_key}.mlir"
        dynamic_vmfb = cache_dir / f"{dyn_key}.vmfb"
        dynamic_config = cache_dir / f"{dyn_key}_exec_config.mlir"
        write_dynamic_rank3_matmul(dynamic_mlir, k, n)
        compile_mlir(args, dynamic_mlir, dynamic_vmfb)
        result["dynamic_mlir"] = str(dynamic_mlir)
        result["dynamic_vmfb"] = str(dynamic_vmfb)
        if args.inspect:
            inspect_codegen(args, dynamic_mlir, dynamic_config)
            result["dynamic_exec_config"] = str(dynamic_config)
        if args.benchmark:
            benchmark(args, dynamic_vmfb, b, s, k, n)

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
