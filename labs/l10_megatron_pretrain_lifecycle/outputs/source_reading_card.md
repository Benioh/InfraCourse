# L11 源码阅读卡片

## 主路径

```text
pretrain_gpt.py task entry
  -> training.pretrain
  -> train_step
  -> metrics.jsonl
  -> checkpoint
  -> report / acceptance
```

## 必读片段

| 文件 | 行号 | 结论 |
|---|---:|---|
| `labs/l10_megatron_pretrain_lifecycle/patch/starter/train_step.py` | L17-L38 | starter 列出 train_step 输入、optimizer 返回值和 TODO 合同 |
| `labs/l10_megatron_pretrain_lifecycle/patch/reference/train_step.py` | L13-L32 | reference 校验 loss，并解析 optimizer step result |
| `labs/l10_megatron_pretrain_lifecycle/patch/reference/train_step.py` | L44-L76 | reference 展示完整调用顺序和 metrics 字段 |
| `labs/l10_megatron_pretrain_lifecycle/patch/tests/test_patch.py` | L47-L69 | 测试调用顺序、loss 平均、tokens、grad_norm、skip 和 lr |
| `mini_infra/megatron/training/training.py` | L83-L116 | MiniInfra 中同构 train_step 主路径 |
| `mini_infra/megatron/training/training.py` | L169-L187 | pretrain 保存 checkpoint、config、schedule 和 checkpoint artifact |
| `labs/l10_megatron_pretrain_lifecycle/scripts/run_lifecycle.py` | L268-L293 | lifecycle 循环调用 train_step，写 metrics 和 checkpoint marker |
| `labs/l10_megatron_pretrain_lifecycle/scripts/run_lifecycle.py` | L294-L327 | acceptance 计算 loss drop、restart 和 final loss |
| `mini_infra/megatron/training/checkpointing.py` | L17-L43 | checkpoint 保存 model、optimizer、scheduler、parallel state 和 latest marker |
| `github_repo/Megatron-LM/megatron/training/training.py` | L1945-L1983 | 真实 Megatron optimizer update 成功后推进 scheduler |

## 判断句

- `train_step` 的核心是边界，不只是 forward。
- optimizer skip 时 scheduler 不推进，metrics 写 `skipped_iter=1`。
- 多 microbatch loss 要按平均值进入日志。
- lifecycle drill 证明 tiny model 同构路径，不能替代真实集群训练。
- checkpoint 证据至少要包含 model、optimizer、scheduler、parallel state 或对应 marker。

## 自检问题

1. `optimizer.step()` 返回 `None` 时本关怎样处理？
2. 哪个测试证明 scheduler 在 skip 时不推进？
3. `acceptance.json` 的 `all_passed` 由哪些条件组成？
4. 真实 Megatron 的 scheduler increment 与本关有什么差异？
5. 如果 resume 后 loss 跳变，应先查哪些 state？
