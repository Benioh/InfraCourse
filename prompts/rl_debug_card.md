# RL Debug Card

适用范围：L10-L11 的 verl/SLiME、reward collapse、KL 爆炸、rollout 慢、actor/rollout 资源拆分和 weight sync 问题。

## Observe

请先收集：

- prompt 样例、response 样例、reward parser 输出和失败样例。
- KL、reward mean/std、response length、rollout latency、actor step time。
- rollout server 参数、Megatron actor checkpoint、weight sync 日志。
- 是否只跑 L10.5 rollout-only，还是已经接入 Ray/Megatron actor。

## Ask

推荐提问：

```text
你是 RL infra reviewer。请先区分 reward/parser、rollout serving、KL/reference、actor update、weight sync 五类根因。
不要建议调大训练规模；先给最小 self-test 和 rollout-only 对照。
```

## Patch Plan

要求 AI 输出：

- 当前问题最可能属于哪一段链路。
- 最小 self-test：固定 prompt/response/reward 期望值。
- 最小 rollout-only 对照：不接 actor、不接 Ray、不接 weight sync。
- 接入 SLiME 前必须验证的 checkpoint/tokenizer/TP 参数。

## Human Check

人工必须确认：

- reward parser 是否能处理真实 response 格式。
- KL reference model 与 actor/tokenizer 是否一致。
- rollout 使用的是新权重还是旧权重。
- mock rollout 或 validation-only 是否被写成真实 RL 成功。

## Apply

优先小改动：

- 增加 reward self-test 和失败样例 artifact。
- 增加 rollout latency、response length、reward_input_ready metrics。
- 把 weight sync 前后的 checkpoint id 写入日志。
- 用 L10.5 先隔离 rollout，再接 L11 全链路。

## Test

验证顺序：

1. reward parser self-test。
2. rollout-only smoke。
3. 小 batch actor update。
4. weight sync 后再次 rollout。
5. `make grade` + `make self-check`。

## Explain

报告里写清：

- reward collapse 是 parser、采样、KL 还是训练更新导致。
- rollout 慢是否已在 L10.5 复现。
- weight sync 是否有 checkpoint id 和加载日志证据。

## Commit

提交前检查：

- 不提交私有数据或大 checkpoint。
- 不删除 reward/parser 断言。
- 不把一次 toy reward 上升写成 RL 训练成功。
