# L10 源码带读：Online Softmax、Ring Plan 与 Megatron CP

这份带读按主路径组织。先看 patch 的数学合同，再看 MiniInfra 的 ring plan，最后对照 Megatron/TE 的 CP 配置和通信入口。

## 0. 源码地图

```text
labs/l09_long_context_cp/patch/starter/ring_attention.py
labs/l09_long_context_cp/patch/reference/ring_attention.py
labs/l09_long_context_cp/patch/tests/test_patch.py
mini_infra/megatron/core/context_parallel/ring_attention.py
mini_infra/megatron/core/context_parallel/rope.py
mini_infra/megatron/core/context_parallel/yarn.py
github_repo/Megatron-LM/megatron/core/model_parallel_config.py
github_repo/Megatron-LM/megatron/core/parallel_state.py
github_repo/Megatron-LM/megatron/core/extensions/transformer_engine.py
labs/l09_long_context_cp/scripts/run_seqlen.py
labs/l09_long_context_cp/scripts/eval_yarn.py
```

## 1. 先读 patch starter

文件：`labs/l09_long_context_cp/patch/starter/ring_attention.py`

重点看：

1. L20-L33：函数签名和 shape 约定。
2. L34-L41：读取 shape、计算 scale、切 K/V chunk。
3. L43-L48：初始化 running 状态。
4. L49-L60：online softmax 循环和最终 normalize。

读完要能回答：`running_max`、`running_denom` 和 `running_out` 的 shape 分别是什么？为什么 K/V 沿 `dim=2` 切？

## 2. 再读 patch reference

文件：`labs/l09_long_context_cp/patch/reference/ring_attention.py`

重点看：

1. L16-L27：shape、scale、chunk 和 running 状态。
2. L29-L39：当前 chunk 的 scores、chunk max、new max 和旧状态缩放。
3. L40-L47：当前 chunk 的 exp、denom、out 和最终输出。

读完要能回答：第一次循环为什么旧状态贡献为 0？`running_out` 为什么要和 `running_denom` 使用同一个缩放系数？

## 3. 读 patch tests

文件：`labs/l09_long_context_cp/patch/tests/test_patch.py`

重点看：

1. L24-L26：测试使用 PyTorch SDPA 定义参考输出。
2. L29-L38：单 chunk 与 full attention 对齐。
3. L41-L50：四个 chunk 与 full attention 对齐。
4. L53-L63：不均匀 chunk。
5. L66-L75：长 K/V。
6. L78-L89：autograd backward。

读完要能回答：哪些测试证明数学等价？哪些测试只说明梯度路径能跑通？

## 4. 读 MiniInfra ring plan

文件：`mini_infra/megatron/core/context_parallel/ring_attention.py`

重点看：

1. L33-L50：`RingStep` 记录一次 ring 传递事件。
2. L53-L56：教学版 attention memory 估算。
3. L59-L69：`ring_attention_plan` 的输入、chunk 和 bytes per chunk。
4. L70-L94：按 cp rank 展开 ring steps、通信字节和 CP 后峰值显存。
5. L97-L110：不同 seq_len 和 cp_size 的 sweep。

读完要能回答：CP size 增大时，峰值显存估算怎样变？通信字节为什么与 `cp_size * (cp_size - 1)` 相关？

## 5. 读 RoPE 和 YaRN 边界

文件：

- `mini_infra/megatron/core/context_parallel/rope.py`
- `mini_infra/megatron/core/context_parallel/yarn.py`

阅读顺序：

1. `rope.py` L27-L36：反频率公式和偶数维检查。
2. `rope.py` L39-L47：位置角度和二维旋转。
3. `rope.py` L50-L58：按 pair 应用 RoPE。
4. `yarn.py` L29-L40：训练上下文到目标上下文的 scale。
5. `yarn.py` L43-L55：temperature 和 summary。

读完要能回答：RoPE/YaRN 解决位置外推问题，为什么不能替代 ring attention 的显存路径？

## 6. 读 Megatron CP 配置

文件：`github_repo/Megatron-LM/megatron/core/model_parallel_config.py`

重点看：

1. L45-L53：`context_parallel_size` 和 hierarchical CP 配置。
2. L56-L63：每个 DPxCP rank 的最大序列长度边界。
3. L65-L70：hybrid context parallel 对 packed samples 的用途。

读完要能回答：`context_parallel_size` 控制什么？为什么 packed sample 会引入 hybrid CP？

## 7. 读 parallel_state CP group

文件：`github_repo/Megatron-LM/megatron/core/parallel_state.py`

重点看：

1. L953-L963：创建 context parallel process group。
2. L964-L980：当前 rank 属于某个 CP group 时保存 group 和 global ranks。
3. L1503-L1516：读取 CP group 和 global ranks 的 getter。
4. L1823-L1836：读取 CP world size 和 rank。

读完要能回答：真实 CP 通信为什么需要 process group？global ranks 会交给谁消费？

## 8. 读 Transformer Engine attention CP 参数

文件：`github_repo/Megatron-LM/megatron/core/extensions/transformer_engine.py`

重点看：

1. L1411-L1416：默认构造包含 TP、CP、HCP 的 process group collection。
2. L1437-L1449：CP size 大于 1 时传入 `cp_group`、global ranks 和 stream。
3. L1450-L1464：设置 `cp_comm_type`，并处理 hierarchical CP。
4. L2633-L2648：RoPE 的 THD 格式接口带有 `cp_size` 和 `cp_rank`。

读完要能回答：Megatron/TE 里 CP group、communication type 和 RoPE 的 CP 参数在哪里进入 attention 路径？

## 9. 读 drill 脚本

文件：

- `labs/l09_long_context_cp/scripts/run_seqlen.py`
- `labs/l09_long_context_cp/scripts/eval_yarn.py`

阅读顺序：

1. `run_seqlen.py` L12-L18：输入 seq 和 CP size，输出 ring plan。
2. `eval_yarn.py` L12-L20：输入训练上下文和目标上下文，输出 YaRN 简化估算。

读完要能回答：这两个脚本各自能证明什么？哪些结论仍然需要真实训练或评估？

## 10. 读完后的自检问题

1. `scores_chunk` 的 shape 是什么？
2. 为什么旧的 `running_out` 也要乘 `exp(old_max - new_max)`？
3. `Sk=7, num_chunks=2` 时实现为什么不能假设每块长度相同？
4. CP ring 中每个 rank 为什么要处理来自其他 rank 的 K/V chunk？
5. RoPE/YaRN 的输出证据和 ring attention 的输出证据有什么差异？
