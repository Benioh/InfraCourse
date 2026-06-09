# L11 Debug Checklist：Train Step 与 Lifecycle

## 1. 调用顺序

| 检查项 | 正确行为 |
|---|---|
| zero grad | optimizer 有 `zero_grad` 时先调用 |
| forward/backward | 在清梯度后执行，返回 dict |
| optimizer step | forward/backward 后调用一次 |
| scheduler step | 仅 optimizer update 成功时调用 |
| metrics | 最后读取 loss、lr、skip 和 tokens |

## 2. Loss 与 microbatch

1. 输出有 `loss` 或 `losses`。
2. `losses` 非空。
3. 多 microbatch loss 取平均。
4. `num_microbatches` 与 loss 列表长度一致。
5. 缺失或空列表抛 `TrainStepError`。

## 3. Optimizer 与 scheduler

| optimizer 返回 | success | scheduler | skipped_iter |
|---|---:|---|---:|
| `True` | true | step | 0 |
| `None` | true | step | 0 |
| `False` | false | 不推进 | 1 |
| `{"success": false}` | false | 不推进 | 1 |
| 其他类型 | error | 不进入 | error |

## 4. Lifecycle artifact

| artifact | 检查内容 |
|---|---|
| `metrics.jsonl` | iteration、loss、lr、num_microbatches、skipped_iter、tokens |
| `artifacts/acceptance.json` | loss_drop、restart_ok、final_loss_ok、all_passed |
| checkpoint marker | `latest_checkpointed_iteration.txt` 是否存在且与文件匹配 |
| `report.md` | 是否说明实现来源、scheduler 来源、预测和结果 |
| `real_megatron_command.sh` | 集群配置下是否写出真实命令模板 |

## 5. 常见问题

| 现象 | 优先检查 |
|---|---|
| loss 不降 | zero_grad 顺序、loss 平均、data iterator、learning rate |
| LR 不动 | scheduler 是否传入、optimizer 是否一直 skip、`get_lr()` 是否可用 |
| skip storm | grad_norm、loss scale、optimizer 返回值 |
| checkpoint 缺失 | save_every、save_dir、marker 写入权限 |
| report 结论不可信 | metrics 缺字段或 acceptance 条件未记录 |

## 6. 结论分级

| 证据强度 | 可以说明什么 |
|---|---|
| patch tests 通过 | 单步训练合同成立 |
| CPU lifecycle 通过 | tiny model 同构生命周期能产出 loss/lr/checkpoint 证据 |
| real command 写出 | 集群运行入口可复查 |
| 真实 Megatron metrics | 真实训练路径产生了可观测指标 |
| resume 验证 | checkpoint、optimizer、scheduler 和 data state 更完整 |
