# Serving Debug Card

适用范围：L07-L09 的 vLLM/SGLang 服务、OpenAI-compatible API、TTFT/ITL、KV cache、prefix cache、PD 分离与 metrics 问题。

## Observe

请先收集：

- server 启动命令、模型路径、tokenizer/chat template、端口、dtype、max_model_len。
- benchmark workload：prompt/output token 分布、并发、请求数、warmup、是否 streaming。
- `serve.log`、benchmark output、metrics endpoint 截图或 JSON。
- TTFT、ITL/TPOT、E2E、tokens/sec、cache hit、queue 指标。

## Ask

推荐提问：

```text
你是 serving infra reviewer。请把“服务慢”拆成 TTFT、ITL、E2E、queue、cache、tokenizer 六类。
只基于我提供的 server config、workload 和 metrics 提出排查顺序。
不要建议换模型或重写服务，先做最小 benchmark 对照。
```

## Patch Plan

要求 AI 输出：

- 用户体感问题对应的首要指标。
- 最小 benchmark：固定 prompt/output token、并发、采样窗口。
- 最小配置改动：端口、max_model_len、cache、batch、PD 路由。
- 需要新增的 metrics 字段或日志。

## Human Check

人工必须确认：

- workload 是否可比，是否混入更长 prompt 或更长 output。
- TTFT 是否包含 queue/tokenizer 时间。
- prefix cache 是否真的命中相同前缀。
- PD 结论是否来自 8×H200，而不是本地配置审查。

## Apply

优先小改动：

- 固定 benchmark 输入输出长度并写入 artifacts。
- 增加 cache hit、queue、prefill/decode 分项日志。
- 修正端口、chat template、模型路径和 tokenizer 路径。
- 对 PD 只先修路由和 metrics，不直接宣称性能收益。

## Test

验证顺序：

1. `/v1/models` 或 health endpoint。
2. 单请求功能 smoke。
3. 固定 workload benchmark。
4. cache/PD 对照实验。
5. `make self-check M=<mission>` 检查报告证据。

## Explain

报告里写清：

- 慢的是 TTFT、ITL、E2E 还是 queue。
- workload、硬件和服务参数是否一致。
- 结论是否只适用于本地 4090 或 8×H200。

## Commit

提交前检查：

- 不提交模型权重或大 trace。
- 不把短 prompt benchmark 推广到长上下文 workload。
- 保留 server config、benchmark command 和 metrics 原始文件。
