# L34 · CUDA Graph Cache + Memory Savor：RL co-locate 的显存与 replay 边界

<!-- LECTURE_FIRST_START -->

本讲处理 RL co-locate 中两个相互牵制的底层问题：rollout 阶段希望用 CUDA Graph replay 降低 decode 的 CPU launch 开销，training 阶段又需要腾出显存给权重、optimizer state 和 activation。若直接释放 rollout 侧显存，graph 录制时绑定的地址可能失效；若两边显存同时常驻，又容易在阶段切换处 OOM。

## 学习路线

1. 读 [system_map.md](system_map.md)：确认 L34 在 RLHF / rollout systems 主线中的位置。
2. 读 [lecture.md](lecture.md)：从 decode launch overhead、地址稳定、Memory Savor 和 co-locate 切换顺序讲到验收边界。
3. 读 [source_walkthrough.md](source_walkthrough.md)：按 patch、pytest、Megatron CUDA Graph 和 SLiME memory utility 主路径阅读。
4. 跑 notebook：[n20_cuda_graph_replay.ipynb](../../notebooks/n20_cuda_graph_replay.ipynb)。
5. 做 quiz：确认 static buffer、VMM/offload、graph cache bucket 和切换顺序。
6. 做 patch：实现 `GraphCache` 与 `MemorySavor` 的 CPU 语义合同。
7. 填写 [outputs/rl_rollout_template.md](outputs/rl_rollout_template.md)，沉淀一次 co-locate 排查复盘。

## 本讲定位

| 问题 | 本讲回答 |
|---|---|
| 所属主线 | RLHF and rollout systems |
| 核心风险 | graph replay 依赖稳定地址，co-locate 又需要在 train/rollout 间腾挪显存 |
| 关键机制 | shape-key graph cache、capture/replay 计数、handle-based pause/resume、paused bytes 统计 |
| 源码落点 | patch starter/reference/tests，Megatron CUDA Graph metadata，SLiME memory utility |
| lab 检验 | 8 个 CPU 测试覆盖 capture、replay、shape 分桶、pause、resume 和 bytes 统计 |

## Patch 闭环

```bash
cat labs/l30.5_cuda_graph_savor/patch/task.md
$EDITOR labs/l30.5_cuda_graph_savor/patch/starter/cuda_graph_cache.py
make patch-test M=l30.5_cuda_graph_savor
```

本讲没有独立 smoke target；Makefile 只提供 patch 相关目标。真实 CUDA Graph 加速和 GPU 地址稳定需要在有 GPU 的环境中另行验证。

## 课后产物

| 产物 | 用途 |
|---|---|
| [outputs/debug_checklist.md](outputs/debug_checklist.md) | 排查 graph cache miss、地址变化、pause/resume 和 co-locate OOM |
| [outputs/source_reading_card.md](outputs/source_reading_card.md) | 复习 patch、Megatron CUDA Graph 和 SLiME memory utility 主路径 |
| [outputs/rl_rollout_template.md](outputs/rl_rollout_template.md) | 记录一次 graph / memory savor 运行的配置、指标和边界 |

<!-- LECTURE_FIRST_END -->

## 进入下一讲

通过 L34 后进入 Async Rollout Pool。下一讲会处理 rollout 并发上限、保序返回和 rollout-only smoke。
