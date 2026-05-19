# L10.7 · CUDA Graph + Memory Savor：RL co-locate 的两根支柱

> **真实背景**：slime co-locate 训练每一轮要在同一张卡上交替跑 inference 和
> training。如果不做特殊处理，KV cache + 模型 weights + activation 三者并存必然
> OOM。slime 的解法 = **`torch_memory_saver`** 让 SGLang offload + Megatron 的
> `CuMemAllocator`。同时为了让 decode 不被 CPU 调度卡住，inference 跑 **CUDA
> Graph** replay。两件事必须配合好：暂停时释放物理页，恢复时重建并让 graph 仍能
> replay。
>
> 本关在 CPU 上把这两个 primitive 写出来，跑通它们的协作。

灵感来源：
- `Awesome-ML-SYS-Tutorial / torch/cuda-graph/readme.md`（CUDA Graph）
- `Awesome-ML-SYS-Tutorial / torch/cuda-graph/readme-2.md`（再探，含 Dual AR omni）
- `Awesome-ML-SYS-Tutorial / rlhf/sys-design/readme-1.md`（offload / upload 节奏）

## 闭环

```bash
cat labs/l30.5_cuda_graph_savor/patch/task.md
$EDITOR labs/l30.5_cuda_graph_savor/patch/starter/cuda_graph_cache.py
make patch-test M=l30.5_cuda_graph_savor
```

## 测试覆盖

| 测试 | 验证 |
|---|---|
| `test_capture_then_replay_correct` | 第一次 capture 输出 = forward_fn 直接调用 |
| `test_replay_reuses_static_buffer` | 同 bs 第二次调用复用 input_buffer (data_ptr 不变) |
| `test_different_bs_triggers_new_capture` | 不同 bs → 新 capture |
| `test_savor_pause_releases_bytes` | pause() 后 physical_bytes == 0 |
| `test_savor_resume_restores_shape` | resume() 后 tensor shape/dtype 与 pause 前一致 |
| `test_colocate_scenario` | 模拟 train ↔ rollout 切换，整套主线 |

## 卡住怎么办

`make patch-hint M=l30.5_cuda_graph_savor` / `make patch-show-solution`。
