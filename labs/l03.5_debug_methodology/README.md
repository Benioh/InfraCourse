# L03.5 · Debug 方法论：从现象到根因的系统路径

这一讲在 DDP（L03）之后插入，解决一个贯穿整个课程的问题：当训练出了问题（loss 不对、多卡 hang、显存异常、GPU idle、结果不一致），你应该按什么流程排查？

大部分工程师的 debug 方式是"看到报错就猜→改→再跑→再猜"。这种方式在简单问题上可以，但在分布式训练、长序列、混合精度等复杂场景下效率极低。本讲建立一套系统的 debug 方法论：最小复现 → 建立 baseline → tensor 对齐 → 问题定位。

## 学习路线

1. 读 [system_map.md](system_map.md)：确认本讲在课程中的位置。
2. 读 [lecture.md](lecture.md)：四步 debug 方法论 + 常见问题模式。
3. 读 [source_walkthrough.md](source_walkthrough.md)：跟读 tensor 对齐和最小复现工具。
4. 做 quiz：确认你能选择正确的 debug 策略。
5. 做 patch：实现 tensor 对齐和问题分类函数。
6. 跑 smoke：生成一次 debug 流程的证据。
7. 填写 [outputs/debug_report_template.md](outputs/debug_report_template.md)。

## 本讲定位

| 问题 | 本讲回答 |
|---|---|
| 它属于哪条主线 | Debug methodology & systematic troubleshooting |
| 它解决什么问题 | 从"瞎猜"变成"有流程的系统排查" |
| 它连接哪些证据 | tensor diff report、minimal repro config、baseline comparison |
| 它连接哪些源码 | `patch/reference/debug_toolkit.py`、`scripts/tensor_alignment_demo.py` |
| lab 检验什么 | 最小复现策略选择、tensor 对齐实现、问题模式分类 |

## 你会学到什么

### 第一步：最小复现
- 为什么要先缩小问题范围：单机、1-2 张卡、小 batch、短 sequence、固定 seed、跑 1 step。
- 如何构造最小复现配置：哪些参数必须固定、哪些可以缩小。
- 固定随机种子的正确方式（torch、numpy、python random、CUDA、dataloader worker）。
- 最小复现成功的标准：问题在简化配置下仍可稳定复现。

### 第二步：建立 Baseline
- Baseline 的选择策略：原始 HF model、单卡版本、eager attention、未 packed、未并行。
- 如何确保 baseline 和实验组只有一个变量不同。
- 如何处理不可控因素（CUDA 非确定性、数据顺序）。

### 第三步：Tensor 对齐
- 逐层对比两个模型的输出：input_ids → embedding → attention → MLP → logits → loss。
- 对齐指标：absolute difference、relative difference、cosine similarity。
- 误差的传播和放大：前面层微小差异如何在深层被放大。
- attention_mask、position_ids、loss mask 的常见对齐问题。
- packed sequence 场景下如何对齐。

### 第四步：问题模式定位
- **Loss 对不上**：检查 loss mask、reduction 方式、label 对齐。
- **Packed 后效果崩**：检查 attention mask 是否正确隔离样本、position_ids 是否重置。
- **多卡 hang**：检查 NCCL 日志、确认所有 rank 到达同一 collective、检查 tensor shape 一致性。
- **多卡比单卡慢**：检查通信 overlap、数据均衡、straggler effect。
- **显存异常**：memory profiler + 检查 tensor 生命周期。
- **GPU idle**：nsys timeline + 检查 sync 点和 data loading。
- **时间测量不准**：是否忘了 sync。

## Patch 闭环

```bash
cat labs/l03.5_debug_methodology/patch/task.md
$EDITOR labs/l03.5_debug_methodology/patch/starter/debug_toolkit.py
make patch-test M=l03.5_debug_methodology
```

测试覆盖：

| 测试 | 验证 |
|---|---|
| `test_tensor_diff_report` | 能计算两个 tensor 的 abs_diff、rel_diff、cosine_sim |
| `test_layer_alignment` | 能逐层对比两个模型的输出，找到第一个 diverge 层 |
| `test_minimal_repro_config` | 能从完整配置中生成最小复现配置 |
| `test_classify_symptom` | 能根据症状描述判断问题类别 |
| `test_seed_everything` | 固定种子后多次执行结果一致 |
| `test_diverge_detection` | 能判断两个序列从哪一步开始 diverge |

## Smoke 闭环

```bash
python labs/l03.5_debug_methodology/scripts/run_smoke.py \
  --config configs/4090_debug.yaml \
  --mode smoke
```

## 课后产物

| 产物 | 用途 |
|---|---|
| [outputs/debug_report_template.md](outputs/debug_report_template.md) | 一次 debug 过程的记录模板 |
| [outputs/source_reading_card.md](outputs/source_reading_card.md) | Debug 方法论速查卡 |

## 进入下一讲

`make patch-test M=l03.5_debug_methodology` 通过后，进入 [L04 GPU Kernel](../l04_gpu_kernel/README.md)。后续每个 Lab 遇到问题时，都可以回到本讲的方法论。
