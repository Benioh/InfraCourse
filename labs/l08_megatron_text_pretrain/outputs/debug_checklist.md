# L09 Debug Checklist：Megatron 训练 step 与 LR Scheduler

## 1. 先确认运行证据

| 检查项 | 证据 | 判断 |
|---|---|---|
| 命令快照 | `command.sh`、`train.log` | 是否记录了实际命令和 expected command |
| 配置 | `config.resolved.yaml` | `seq_length`、batch、TP/PP、precision 是否与预期一致 |
| 数据 | `data/wikitext/train.jsonl`、indexed `.bin/.idx` | 缺数据时只能算启动边界验证 |
| Megatron runtime | `fallback_reason.txt` | 包不可导入时不能声明真实训练完成 |
| metrics | `metrics.jsonl` | `status` 是 fallback 还是 ready for manual launch |

## 2. LR 曲线错位

1. 查 resolved config：`lr`、`min_lr`、warmup、decay、restart、total steps。
2. 查 scheduler state：`step_count`、`num_steps` 或 consumed samples。
3. 查 batch 公式：`micro_batch_size * data_parallel_size * num_microbatches`。
4. 查 restart 边界：restart step 是否属于新 segment 起点。
5. 查 total clamp：超过 total 后是否仍然继续套 cosine。
6. 查多 param group：所有需要同步的 groups 是否写入了 lr。

## 3. Loss 抖动或 nan

| 现象 | 优先检查 |
|---|---|
| restart 点附近 loss 上升 | LR 是否重置到 `max_lr`，grad norm 是否同步变大 |
| 第一次训练很快 nan | warmup、初始 lr、loss scale、batch size |
| resume 后 loss 突然跳 | scheduler state、optimizer state、data iterator、tokenizer 和 topology |
| skipped iteration 增多 | optimizer update success、grad norm、loss scale、溢出日志 |
| lr 日志缺失 | canonical lr logging、param group 是否有 `lr` 字段 |

## 4. Resume 连续性

| 状态 | 需要连续的字段 |
|---|---|
| model | 权重、parallel topology、dtype |
| optimizer | 动量、方差、master weights、param groups |
| scheduler | `num_steps` / `step_count`、lr 配置、decay style、restart 配置 |
| data | consumed samples、数据 split、tokenizer、IndexedDataset prefix |
| logging | run id、iteration、global step、checkpoint lineage |

## 5. Patch 排查

1. `boundaries = [0] + restart_steps + [total_steps]` 是否正确。
2. `restart_steps=[]` 时 boundaries 是否为 `[0, total_steps]`。
3. `_set_lr()` 是否遍历所有 `optimizer.param_groups`。
4. `_compute_lr(step)` 是否先处理 `step >= total_steps`。
5. segment 查找是否使用 `start <= step < end`。
6. `ratio` 是否使用段内相对位置，而不是全局 step。
7. `step()` 是否先更新计数，再把新 lr 写回 optimizer。

## 6. 结论分级

| 证据强度 | 可以说明什么 |
|---|---|
| patch tests 通过 | scheduler 最小合同成立 |
| parser self-test 通过 | 日志解析器能提取示例字段 |
| fallback drill 完成 | 启动边界和缺失条件被记录 |
| real Megatron log 有 loss/lr/tokens/sec | 真实训练 step 至少产生了可观测指标 |
| checkpoint + resume log 连续 | 恢复边界得到更强验证 |
