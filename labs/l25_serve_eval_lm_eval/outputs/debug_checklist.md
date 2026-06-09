# L26 Serving Eval Debug Checklist

## 1. 先确认评测边界

- 命令、git revision、配置文件、模型、endpoint、采样参数和样本集是否记录完整。
- 当前是 patch-test、stub eval，还是真实 vLLM/SGLang/OpenAI-compatible endpoint。
- `n_shots`、`n_eval`、temperature、max_tokens、stop sequence 是否和 baseline 一致。
- `n_total` 是否等于预期评测样本数，不包含 few-shot examples。

## 2. exact_match 低，first_number_match 高

- 抽查 `metrics.jsonl` 中的 prediction 和 reference。
- 检查是否多了单位、句号、解释文本、大小写差异或前后空格。
- 判断任务是否要求严格格式；若要求 JSON、单词或选项字母，numeric metric 不能替代任务指标。
- 检查 prompt 是否明确要求只输出答案。

## 3. exact_match 和 first_number_match 都低

- 检查 few-shot prompt 顺序和答案是否泄漏到 eval item。
- 检查 temperature 是否固定为 0。
- 检查 stop sequence 是否阻止模型继续生成下一题。
- 检查 endpoint 是否打到了预期模型和端口。
- 检查 tokenizer/chat template 是否和 completion prompt 形状匹配。
- 检查 timeout、异常和空响应是否被单独记录。

## 4. 完成率异常

- 比较 `config.resolved.yaml` 中的 `n_eval` 和 summary 中的 `n_total`。
- 查 `artifacts/fallback.txt` 或服务日志，确认是否有异常。
- 区分 skipped、timeout、exception、empty response 和 wrong answer。
- 如果真实服务压力较高，补 latency、并发和 retry 统计。

## 5. Endpoint 配置

- vLLM/SGLang completion API 应使用 `/v1/completions`。
- chat 模型若只支持 chat endpoint，应使用 chat sampler 或转换 prompt。
- API key、base_url、model id、max_tokens 和 stop 必须写入配置快照。
- 真实服务评测要保留 prerequisite serve command 或服务版本信息。

## 6. 报告最低字段

- 数据集名称、样本数、shot 数、题目顺序和随机种子。
- endpoint、model、backend、dtype、硬件和服务启动命令。
- exact_match、first_number_match、completion rate、duration。
- 每样本 prediction、reference、exact、numeric 命中。
- 失败样本分类和下一步动作。
