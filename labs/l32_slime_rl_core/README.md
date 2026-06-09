# L36 · SLiME RL Core：Weight Sync Coordinator

<!-- LECTURE_FIRST_START -->

L36 讲 RL 训练和 rollout engine 之间的权重交接。L35 解决了 prompt 如何并发提交给 rollout server；这一讲继续看 actor 训练完成后，新权重如何到达推理侧，以及 shape、dtype、版本和同步耗时如何成为排障证据。

## 学习路线

1. 读 [system_map.md](system_map.md)：确认 L36 在 RLHF / rollout systems 主线中的位置。
2. 读 [lecture.md](lecture.md)：从 weight sync 的系统位置、局部合同、SLiME 源码对照讲到同步排障。
3. 读 [source_walkthrough.md](source_walkthrough.md)：按 patch、MiniInfra、SLiME train、Megatron updater 和 SGLang engine 主路径阅读。
4. 做 quiz：确认 dtype/shape mismatch、版本推进、NCCL/Ray 分工和 smoke 边界。
5. 做 patch：实现 `WeightSyncCoordinator` 并通过测试。
6. 跑 smoke：验证 SLiME 配置、toy 数据和同步指标 artifact。
7. 填写 [outputs/rl_rollout_template.md](outputs/rl_rollout_template.md)，记录一次 weight sync 复盘。

## 本讲定位

| 问题 | 本讲回答 |
|---|---|
| 所属主线 | RLHF and rollout systems |
| 核心风险 | rollout engine 使用旧 policy、shape/dtype 同步错误、sync 时间吞掉训练吞吐、版本证据缺失 |
| 关键机制 | state_dict 合同、shape/dtype gate、bytes_synced、weight_version、Ray 协调、SGLang 在线更新 |
| 源码落点 | patch coordinator、MiniInfra SLiME loop、SLiME train loop、Megatron updater、SGLang engine |
| lab 检验 | 基本同步、shape mismatch、dtype mismatch、extra train key、同步统计 |

## Patch 闭环

```bash
cat labs/l32_slime_rl_core/patch/task.md
$EDITOR labs/l32_slime_rl_core/patch/starter/weight_sync.py
make patch-test M=l32_slime_rl_core
```

参考实现验收：

```bash
IMPL=reference make patch-test M=l32_slime_rl_core
```

Smoke：

```bash
python labs/l32_slime_rl_core/scripts/run_slime_lab.py --run-id l36_smoke
```

## 课后产物

| 产物 | 用途 |
|---|---|
| [outputs/debug_checklist.md](outputs/debug_checklist.md) | 排查 sync 慢、版本未推进、shape/dtype mismatch 和 rollout policy stale |
| [outputs/source_reading_card.md](outputs/source_reading_card.md) | 复习 patch、MiniInfra、SLiME、Megatron updater 和 SGLang engine 主路径 |
| [outputs/rl_rollout_template.md](outputs/rl_rollout_template.md) | 记录一次 actor/rollout 资源切分和 weight sync 复盘 |

<!-- LECTURE_FIRST_END -->

## 进入下一讲

通过 L36 后进入 L37 CUDA IPC Weight Sync。下一讲会把本讲的本地 state_dict 合同推进到跨进程权重共享和 IPC handle 边界。
