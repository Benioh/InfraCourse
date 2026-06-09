# L23 · PyTorch SDPA 与 FlashAttention benchmark

<!-- LECTURE_FIRST_START -->

本讲讲推理和训练都会遇到的 attention 内核选择问题：eager attention 会显式生成 `[B, H, T, T]` 的 score 和 weight，序列一长就把 HBM 带宽和显存峰值压满；PyTorch SDPA 在合适的 GPU、dtype、head_dim 和 mask 条件下会走 FlashAttention 类 backend，把 QK、softmax、V 的中间过程分块完成，减少中间矩阵落到 HBM 的次数。

## 学习路线

建议按下面顺序走，先把系统讲通，再写 patch。

1. 读 [system_map.md](system_map.md)：确认 L23 在 Serving 性能路径中的位置。
2. 读 [lecture.md](lecture.md)：从 eager attention 的内存问题讲到 SDPA dispatch 和 benchmark 证据。
3. 读 [source_walkthrough.md](source_walkthrough.md)：按主路径阅读 patch、MiniInfra online softmax 和真实项目里的 SDPA 调用。
4. 做 quiz：确认概念、边界和指标解释。
5. 做 patch：实现 eager baseline、SDPA 调用和 benchmark 指标。
6. 跑 drill：用 CPU smoke 或 GPU profile 生成 `bench.csv`、`metrics.jsonl` 和 `bench_summary.json`。
7. 填写 [outputs/serving_metrics_template.md](outputs/serving_metrics_template.md)，留下可复查的指标结论。

## 本讲定位

| 问题 | 本讲回答 |
|---|---|
| 所属主线 | Serving systems / attention kernel performance |
| 核心瓶颈 | eager attention 显式保存 `[T, T]` 中间矩阵，长序列下 HBM 流量和峰值显存上升很快 |
| 关键机制 | SDPA 在可用条件下选择 flash / efficient / math backend，用分块 softmax 降低中间矩阵落盘 |
| 源码落点 | `patch/reference/flash_bench.py`、`scripts/run_bench.py`、`mini_infra/gpu/triton_softmax.py`、真实项目中的 SDPA wrapper |
| lab 检验 | attention 数值等价、causal mask、指标字典、CUDA 条件下的速度和显存趋势 |

## 学完后能做什么

- 解释 eager attention 为什么会在长序列下产生 O(T^2) 中间内存。
- 说明 FlashAttention/SDPA 改变的是中间结果存放和 HBM 访问路径，注意力公式保持一致。
- 判断一次 benchmark 是否有足够证据：device、dtype、shape、causal、iters、同步、显存统计是否记录完整。
- 看懂真实项目里为什么要在调用 SDPA 前做 layout transpose、dtype 对齐、mask 选择和 backend 约束。
- 完成 patch，并能把测试失败定位到公式、mask、dispatch 或 benchmark 统计中的一个环节。

## Patch 闭环

```bash
cat labs/l22_flash_attn_v2_bench/patch/task.md
$EDITOR labs/l22_flash_attn_v2_bench/patch/starter/flash_bench.py
make patch-test M=l22_flash_attn_v2_bench
```

CPU smoke：

```bash
IMPL=reference python labs/l22_flash_attn_v2_bench/scripts/run_bench.py \
  --config configs/cpu_smoke.yaml \
  --run-id l23_cpu_smoke
```

GPU benchmark：

```bash
PROFILE=4090_bench IMPL=reference bash labs/l22_flash_attn_v2_bench/scripts/run_bench.sh l23_4090_bench
```

CPU 只能验证数值和 artifact 结构。速度和显存结论必须在 CUDA、合适 dtype 和足够长序列上报告。

## 课后产物

| 产物 | 用途 |
|---|---|
| [outputs/debug_checklist.md](outputs/debug_checklist.md) | 排查 SDPA/FlashAttention 数值、dispatch、显存和 benchmark 误判 |
| [outputs/source_reading_card.md](outputs/source_reading_card.md) | 复习 eager baseline、SDPA 调用、online softmax 和真实 wrapper |
| [outputs/serving_metrics_template.md](outputs/serving_metrics_template.md) | 记录一次 attention benchmark 的条件、指标、源码对应和结论 |

<!-- LECTURE_FIRST_END -->

## 进入下一讲

通过 L23 后进入 L24 AWQ-lite W8 per-channel 量化 serving。量化会继续使用本讲的指标习惯：任何速度、显存或精度判断都要绑定硬件、dtype、shape 和比较对象。
