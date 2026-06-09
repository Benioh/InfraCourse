# Debug Checklist：L32 Train-Infer Mismatch

## 1. 固定现场

- 记录命令、配置、git commit、Python/PyTorch 版本、框架版本、dtype、硬件和随机种子。
- 保存 rollout token、rollout logprob、training logprob、loss mask、advantage 和 response length 的样本。
- 区分当前运行是 patch-test、notebook、离线分析还是真实 RL 训练。

## 2. 先确认 token 对齐

| 检查项 | 证据 | 失败时的判断 |
|---|---|---|
| token 序列 | rollout 输出和 training 输入逐 token 一致 | 采样后处理或模板拼接破坏了 logprob 对应关系 |
| loss mask | response token 为 1，prompt/tool/padding 按课程约定屏蔽 | ratio 统计混入了不该训练的位置 |
| shape | logprob 与 mask shape 完全一致 | 后续 IS 公式没有可靠语义 |
| dtype/device | 张量 dtype 和 device 明确记录 | device mismatch 或精度路径可能污染判断 |

## 3. 再看 mismatch 指标

- K3 KL：aligned case 应接近 0，异常上升时继续拆 ratio 分布。
- ratio 分布：记录 mean、min、max、p50、p95、p99。
- TIS/MIS 命中率：分别记录低界、高界和 mask fraction。
- Veto：记录 catastrophic token fraction 与 sequence fraction。
- Batch Normalize：记录 normalize 前均值、batch norm factor 和最终权重均值。

## 4. 沿源码主路径复查

- `labs/l29.5_train_infer_mismatch/patch/starter/mismatch.py`：学生实现的六个算子。
- `labs/l29.5_train_infer_mismatch/patch/reference/mismatch.py`：最小正确公式和边界处理。
- `labs/l29.5_train_infer_mismatch/patch/tests/test_patch.py`：10 个行为合同。
- `github_repo/slime/examples/train_infer_mismatch_helper/mis.py`：真实 TIS、RS、veto、batch norm 和 metrics 分支。

## 5. 常见错误判断

- 只看 reward 曲线，没有记录 ratio 和 K3。
- 只看平均 K3，忽略少量极端 token。
- 把 batch normalize 当异常样本过滤器。
- padding 位置没有 mask，导致 Geometric IS 被无效值污染。
- token 序列和 logprob 已经错位，却继续讨论 IS 阈值。

## 6. 结束条件

- 问题能用一组最小张量或一条复现命令说明。
- 关键指标和 artifact 已落盘。
- 能指出源码中 ratio、mask、veto 或 batch norm 的状态变化位置。
- 结论已写入 `rl_rollout_template.md`，并包含下一步动作。
