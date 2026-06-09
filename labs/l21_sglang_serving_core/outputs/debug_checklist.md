# L22 Debug Checklist

这张 checklist 用来排查 SGLang prefix cache miss、TTFT 偏高、metrics 缺失和 benchmark 结论不可信。

## 1. 先确认运行边界

- [ ] 本次是 patch-test、MiniInfra smoke、server validation，还是连接真实 SGLang server 的 benchmark。
- [ ] 保存 `command.sh`、`config.resolved.yaml`、`serve.log`、`metrics.jsonl` 和 report。
- [ ] 如果 metrics 行里是 `validation_only`，不要写成真实性能结果。
- [ ] 记录 model、port、max_new_tokens、temperature、workload、硬件和 git 状态。

## 2. 检查 tokenized prefix

- [ ] 对两个“应该共享”的请求输出 token id 序列。
- [ ] 计算最长公共前缀长度，不只比较 prompt 字符串。
- [ ] 检查 chat template、BOS/EOS、role marker、空格、换行。
- [ ] 检查 RAG 文档排序、tool schema 字段顺序和 tokenizer 版本。
- [ ] 检查 namespace / `extra_key` 是否把请求隔离。

## 3. 检查 cache 状态

- [ ] cache 是否启用。
- [ ] repeated-prefix 请求的 `matched_prefix_tokens` 是否增长。
- [ ] cache hit rate 是否随 warmup 后的重复请求上升。
- [ ] cache node/token 数是否持续增长或被过度 evict。
- [ ] LRU 是否在 match 和 insert 时刷新访问时间。
- [ ] evict 是否只删叶子，是否保护共享前缀。

## 4. 拆 TTFT

- [ ] waiting queue 是否增长。
- [ ] tokenization / template 是否变慢。
- [ ] prefill 总 token 数是否因为命中减少。
- [ ] suffix 是否仍然很长。
- [ ] first decode 是否被 batch、KV pressure 或 worker 排队影响。
- [ ] output processing 或 client timeout 是否影响测量。

## 5. 对齐真实 SGLang 复杂度

- [ ] `MatchResult.device_indices` 是否为空或过短。
- [ ] `last_device_node` / lock ref 是否保护了运行中请求。
- [ ] page size 对齐是否截短了可缓存 tail。
- [ ] eviction policy、priority 或 host cache 是否改变了保留策略。
- [ ] schedule policy 是否按 prefix match 改变 waiting queue 排序。

## 6. 形成结论

报告建议写成：

```md
现象：
- repeated-prefix workload:
- status: served / validation_only
- TTFT p50/p95:
- cache_hit_rate:

判断：
- cache miss / queue / suffix prefill / decode / measurement

证据：
- token id LCP:
- metrics row:
- 源码路径:

下一步：
- 固定模板 / 调整 namespace / 扩大 cache / 开启 metrics / 连接真实 server
```
