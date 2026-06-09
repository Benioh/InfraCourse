# L20 Debug Checklist

## 1. 固定现场

- 记录命令、模型、端口、max model len、并发、prompt 长度、max tokens、采样参数。
- 保存 `command.sh`、`serve.log`、`metrics.jsonl`、`report.md`。
- 区分 `validation_only` 和真实 served metrics。

## 2. Serving 排查顺序

| 层 | 先看什么 | 常见结论 |
|---|---|---|
| 环境 | vLLM 是否安装、CUDA、模型路径 | 服务无法启动 |
| 端口 | `server_port_open`、端口冲突 | 客户端连不上 |
| 请求 | model、messages、max_tokens | API payload 不符合预期 |
| sampler | temperature、top_p、top_k、typical_p | 输出随机性或质量异常 |
| 指标 | TTFT、ITL、requests/sec | 服务慢或只是 validation |

## 3. Typical-p 排查顺序

1. `typical_p >= 1.0` 时应原样返回 logits。
2. 检查 softmax、surprisal、entropy、distance 的 shape 都是 `(B, V)` 或 `(B, 1)`。
3. 按 distance 升序排序，再用同样索引 gather probs。
4. `typical_p` 很小时仍至少保留 1 个 token。
5. scatter 回原 vocab 顺序后，只 mask 被删除 token。

## 4. 结束条件

- patch-test 能证明 logits filter 合同。
- smoke artifact 能证明 server 是否存在、命令是什么、状态是什么。
- 真实 benchmark 必须有 TTFT 和 ITL，只有 health check 不能当作 serving 性能结论。
