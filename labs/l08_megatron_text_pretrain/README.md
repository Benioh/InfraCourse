# L04 · Megatron 预训练：Cosine With Restarts LR Scheduler

> 本关只做一件事：**实现一个带热重启的 cosine LR scheduler**——Megatron 没内置但很多论文要求。

写完这关你能给任何训练框架加新调度策略，而不是只会调 `--lr 1e-4`。

## 闭环

```bash
cat labs/l08_megatron_text_pretrain/patch/task.md
$EDITOR labs/l08_megatron_text_pretrain/patch/starter/lr_scheduler.py
make patch-test M=l08_megatron_text_pretrain   # 7 个测试，CPU 即可
```

## 测试覆盖

| 测试 | 验证 |
|---|---|
| `test_initial_lr_is_max` | step 0 时 lr == max_lr |
| `test_lr_decreases_within_segment` | 段内单调递减 |
| `test_lr_at_restart_step_is_max` | restart 时刻重置回 max |
| `test_lr_clamps_after_total` | step >= total 时钳到 min |
| `test_multiple_param_groups_synced` | 多 param_group 同步 |
| `test_get_lr_matches_optimizer` | get_lr() 与 optimizer 一致 |
| `test_no_restarts_is_pure_cosine` | restart=[] 退化为标准 cosine |

## 卡住怎么办

1. 看 Megatron `optimizer_param_scheduler.py` 源码（github_repo/）。
2. `make patch-hint M=l08_megatron_text_pretrain`。
3. `make patch-show-solution M=l08_megatron_text_pretrain`。

## 进入下一关

`make patch-test` 全绿后，继续做源码理解口试。下一关 [L04.5 长上下文 CP](../l09_long_context_cp/README.md) 让你写 ring attention。
