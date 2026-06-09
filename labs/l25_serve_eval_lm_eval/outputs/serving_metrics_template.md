# L26 Serving Eval Metrics Template

## 1. 实验信息

| 字段 | 填写 |
|---|---|
| 日期 | |
| git revision | |
| 命令 | |
| run id | |
| backend | |
| endpoint | |
| model | |
| dtype / quantization | |
| hardware | |
| dataset | |
| n_shots / n_eval | |
| temperature / top_p | |
| max_tokens | |
| stop sequence | |

## 2. 质量指标

| 指标 | 数值 | 备注 |
|---|---:|---|
| completion rate | | `n_total / n_eval` |
| exact_match | | 严格字符串匹配 |
| first_number_match | | 数字抽取匹配 |
| duration_s | | 端到端评测耗时 |
| samples failed by exception | | |
| empty responses | | |

## 3. 样本分桶

| 分桶 | 数量 | 典型样本 | 判断 |
|---|---:|---|---|
| exact 正确 | | | |
| numeric 正确但 exact 错 | | | |
| numeric 错 | | | |
| timeout / exception | | | |
| empty response | | | |
| stop 缺失导致续写 | | | |

## 4. 服务证据

| 证据 | 路径或内容 |
|---|---|
| config.resolved.yaml | |
| command.sh | |
| prerequisite serve command | |
| metrics.jsonl | |
| eval_summary.json | |
| report.md | |
| server log | |

## 5. 结论

- 质量结论：
- 格式问题：
- 数值问题：
- 完成率风险：
- 服务端风险：
- 下一步动作：
