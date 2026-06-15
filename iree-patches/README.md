# IREE 补丁说明

本目录记录当前实验使用的 IREE 编译器改动。

## 文件说明

```text
tracked_changes.patch
  原 IREE checkout 中已有文件的改动 diff。

new-files/
  新增的 pass / test 源文件。

source-snapshots/
  已修改 tracked 文件的完整快照，便于 review。
```

## 应用方式

在本地 IREE checkout 中执行：

```bash
cd /path/to/iree
cp -a /path/to/iree-optimization/iree-patches/new-files/* .
git apply /path/to/iree-optimization/iree-patches/tracked_changes.patch
```

然后按本地 IREE 环境重新编译。

## 主要改动

- 新增 `iree-global-opt-flatten-rank3-matmul` pass，将 rank-3 by rank-2 matmul-like contraction 改写为 flattened 2D `linalg.matmul`。
- 新增 `iree-global-opt-assume-input-shape-bounds` pass，为动态输入添加 bounded shape assumption，并支持 attention context barrier 实验。
- CUDA lowering 中加入 safe dynamic-M matmul heuristic：当只有 M 动态、K/N 静态时，用代表性 M bound 选择 MMA config。
- 对 dynamic K 与复杂 dynamic contraction 保留 fallback，避免不安全的 attention dynamic batch matmul 强行进入 MMA。
- 更新相关 BUILD / CMake / lit test。

更多背景和实验结果见 `docs/reports/iree_dynamic_shape_optimization_stage_report.md`。
