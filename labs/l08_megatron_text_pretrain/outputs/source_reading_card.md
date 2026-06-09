# L09 源码阅读卡片

## 主路径

```text
MiniInfra train_step
  -> Megatron OptimizerParamScheduler
  -> Megatron training.py train_step / training_log
  -> patch CosineWithRestartsLR
  -> run_train.py artifact
  -> parse_megatron_log.py metrics
```

## 必读片段

| 文件 | 行号 | 结论 |
|---|---:|---|
| `mini_infra/megatron/training/training.py` | L83-L116 | 教学 train_step 展示 zero_grad、forward/backward、optimizer、scheduler、metrics 的顺序 |
| `mini_infra/megatron/training/training.py` | L169-L175 | checkpoint 保存 scheduler state |
| `github_repo/Megatron-LM/megatron/core/optimizer_param_scheduler.py` | L218-L282 | Megatron scheduler 先处理 warmup 和边界，再按 decay style 计算 lr |
| `github_repo/Megatron-LM/megatron/core/optimizer_param_scheduler.py` | L284-L300 | `step(increment)` 遍历所有 param groups 写 lr 和 weight decay |
| `github_repo/Megatron-LM/megatron/training/training.py` | L1945-L1983 | optimizer 成功后才推进 scheduler，并按 batch 相关增量更新 |
| `github_repo/Megatron-LM/megatron/training/training.py` | L3211-L3221 | 主 loop 从 optimizer param groups 读取 canonical lr，再交给 `training_log` |
| `labs/l08_megatron_text_pretrain/patch/reference/lr_scheduler.py` | L33-L49 | reference 展示 total clamp、segment 查找和 cosine 公式 |
| `labs/l08_megatron_text_pretrain/patch/tests/test_patch.py` | L67-L76 | 多 param group 测试保证 lr 输出同步 |
| `labs/l08_megatron_text_pretrain/scripts/run_train.py` | L54-L72 | drill 检查数据、indexed dataset 和 Megatron runtime |
| `labs/l08_megatron_text_pretrain/scripts/parse_megatron_log.py` | L46-L68 | 日志解析器把训练文本转成结构化字段和缺失字段列表 |

## 读源码时的判断句

- Scheduler 的核心输出在 optimizer param groups 的 `lr` 字段里，返回值只是辅助接口。
- Megatron scheduler 的计数单位可能是 consumed samples，不能机械等同于 iteration。
- `update_successful` 为假时，LR 不应推进。
- LR 日志来自 optimizer param groups，需要 canonical 规则处理多 rank 和默认组。
- Fallback drill 只证明启动边界和 artifact 记录，真实训练要看 loss、lr、tokens/sec 等字段。

## 自检问题

1. `train_step` 中 scheduler 调用在 optimizer 前还是后？
2. Megatron `step(increment)` 的 increment 由哪些 batch 参数决定？
3. Patch reference 中 restart step 怎样回到 `max_lr`？
4. 多 param group 不同步会给训练和日志带来什么风险？
5. 哪个 artifact 能说明本地没有跑真实 Megatron？
