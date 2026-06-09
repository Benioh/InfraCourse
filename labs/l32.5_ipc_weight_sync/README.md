# L37 · CUDA IPC Weight Sync：handle tuple 与共享 storage

<!-- LECTURE_FIRST_START -->

L37 讲 RL co-locate weight sync 的真实传输对象。L36 已经练过本地 `state_dict` 同步合同；这一讲继续看训练侧怎样把 tensor 变成 CUDA IPC-shaped handle tuple，让 rollout 进程按 TP rank 重建指向同一块 storage 的 tensor。

## 学习路线

1. 读 [system_map.md](system_map.md)：确认 L37 在 RL 与对齐主线中的位置。
2. 读 [lecture.md](lecture.md)：从 handle tuple、rank gather、LocalSerializedTensor、flush 时序讲到 SLiME/SGLang 对照。
3. 读 [source_walkthrough.md](source_walkthrough.md)：按 patch、SGLang update、SLiME bucket 和 serializer 主路径阅读。
4. 可选跑 notebook：[n22_weight_sync_handle_tuple.ipynb](../../notebooks/n22_weight_sync_handle_tuple.ipynb)。
5. 做 quiz：确认 handle 不含 tensor data、rank 0 gather、共享 storage、bucket 和 smoke 边界。
6. 做 patch：实现 CPU 版 IPC-shaped weight sync 并通过 7 个测试。
7. 填写 [outputs/rl_rollout_template.md](outputs/rl_rollout_template.md)，记录 handle/data 比例、rank gather 和 flush 时序。

## 本讲定位

| 问题 | 本讲回答 |
|---|---|
| 所属主线 | RLHF and rollout systems |
| 核心风险 | co-locate 同步退化成数据复制、rank gather 错位、LocalSerializedTensor 按错 rank、旧 KV cache 与新权重混用 |
| 关键机制 | CUDA IPC handle tuple、shared storage、ForkingPickler、LocalSerializedTensor、FlattenedTensorBucket、flush_cache |
| 源码落点 | patch controller、SGLang `ModelRunner.update_weights_from_tensor`、SLiME tensor updater、SGLang serializer |
| lab 检验 | handle 大小、data_ptr、rank gather、LST get、state 替换和最后 flush |

## Patch 闭环

```bash
cat labs/l32.5_ipc_weight_sync/patch/task.md
$EDITOR labs/l32.5_ipc_weight_sync/patch/starter/ipc_weight_sync.py
make patch-test M=l32.5_ipc_weight_sync
```

参考实现验收：

```bash
IMPL=reference make patch-test M=l32.5_ipc_weight_sync
```

本讲目录没有 dedicated smoke target。CPU 侧验证以 patch-test 为准；真实 CUDA IPC 需要同机 GPU、multiprocessing spawn、SGLang endpoint 和生命周期日志。

## 课后产物

| 产物 | 用途 |
|---|---|
| [outputs/debug_checklist.md](outputs/debug_checklist.md) | 排查 handle/data 比例、rank gather、storage 共享、flush 时序和 IPC 生命周期 |
| [outputs/source_reading_card.md](outputs/source_reading_card.md) | 复习 patch、SGLang update、SLiME bucket 和 serializer 主路径 |
| [outputs/rl_rollout_template.md](outputs/rl_rollout_template.md) | 记录一次 co-locate weight sync 复盘 |

<!-- LECTURE_FIRST_END -->

## 进入下一讲

通过 L37 后进入 L38 Rollout Freshness。下一讲会把本讲的 `weight_version` 和同步边界扩展成 policy version、staleness 和可接受旧样本判断。
