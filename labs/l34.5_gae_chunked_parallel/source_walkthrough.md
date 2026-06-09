# L40 源码带读：GAE Chunked Parallel

这份带读按“patch 合同 -> 测试 -> SLiME 生产路径”组织。先读清楚 `next_value`、`next_adv` 和 chunk boundary，再看生产代码里的 pad、scan、slice 和 returns。

## 0. 源码地图

```text
labs/l34.5_gae_chunked_parallel/patch/starter/gae_chunk.py
labs/l34.5_gae_chunked_parallel/patch/reference/gae_chunk.py
labs/l34.5_gae_chunked_parallel/patch/tests/test_patch.py
github_repo/slime/slime/utils/ppo_utils.py
```

## 1. Patch Starter

文件：`labs/l34.5_gae_chunked_parallel/patch/starter/gae_chunk.py`

阅读顺序：

- L21-L33：`gae_naive` 的输入、shape 约定和递推公式。
- L35-L47：naive TODO 中的 `next_value`、`next_adv` 和反向循环。
- L51-L63：`gae_chunked_parallel` 的签名和 chunk_size。
- L64-L70：starter 说明教学版保留数学等价，不代表真实并行 kernel。
- L71-L80：chunked 初始化和 single-chunk fallback。
- L82-L94：从最后一个 chunk 往前，chunk 内按 naive 公式递推。
- L95-L102：把 chunk 起点 advantage 和 value 交给左侧 chunk。

读完要得到的结论：starter 的难点不在公式数量，而在两个 boundary 状态是否传对。

可以先跳过：性能注释里的 GPU 挑战，先把 CPU 等价合同写稳。

## 2. Patch Reference

文件：`labs/l34.5_gae_chunked_parallel/patch/reference/gae_chunk.py`

阅读顺序：

- L8-L18：naive 初始化 T、输出张量、`next_value` 和 `next_adv`。
- L19-L27：naive 从右到左写入 advantages。
- L30-L42：chunked 的 fallback 和输出初始化。
- L45-L52：chunked 初始化右侧 boundary，并进入 chunk loop。
- L53-L64：chunk 内递推，并把起点 adv/value 交给左侧。

读完要得到的结论：reference 不是高性能实现，它是 boundary 语义的可读版。

可以先跳过：`torch.zeros_like` 的 dtype 细节。

## 3. Patch Tests

文件：`labs/l34.5_gae_chunked_parallel/patch/tests/test_patch.py`

阅读顺序：

- L22-L32：三步手算样例检查方向。
- L35-L45：短序列 naive 和 chunked 等价。
- L48-L58：T=1024 长链等价。
- L61-L71：T 不整除 chunk_size 时最后短 chunk 仍正确。
- L74-L83：`chunk_size >= T` 时 fallback 等价。
- L86-L96：`last_value` 必须传播到最后 token。
- L99-L109：batched 输入输出 shape 和数值都正确。

读完要得到的结论：测试覆盖数值语义和索引边界，不覆盖真实 GPU 性能。

可以先跳过：`_impl` 的 import 细节。

## 4. SLiME 单样本 GAE

文件：`github_repo/slime/slime/utils/ppo_utils.py`

阅读顺序：

- L311-L318：`get_advantages_and_returns` 的输入。
- L341-L349：CP 大于 1 时先 gather，本地路径直接使用 tensors。
- L351-L360：单样本 vanilla 反向递推，并计算 returns。
- L362-L371：CP 路径把 full result 切回本地片段。

读完要得到的结论：生产路径除了 advantage，还要返回 `returns = advantages + values`。

可以先跳过：Megatron CP import 的实现细节。

## 5. SLiME Batched GAE 入口

文件：`github_repo/slime/slime/utils/ppo_utils.py`

阅读顺序：

- L374-L382：batched 入口接收长度列表、value/reward 列表和 `chunked` 开关。
- L397-L405：检查 batch、CP、device 和 dtype。
- L425-L435：把变长样本 pad 到 `max_len`。
- L436-L442：`chunked=False` 走 `vanilla_gae`。
- L443-L449：`chunked=True` 走 `chunked_gae`。
- L473-L479：按原 response length 去掉 padding。

读完要得到的结论：真实系统先整理变长数据，再选择 GAE 实现。

可以先跳过：CP slice 分支，先看非 CP 的 pad/slice 主路径。

## 6. SLiME Vanilla 与 Chunked GAE

文件：`github_repo/slime/slime/utils/ppo_utils.py`

阅读顺序：

- L488-L499：`vanilla_gae` 的 batch 反向循环。
- L501-L503：vanilla returns。
- L506-L518：`chunked_gae` 的目标和依赖长度说明。
- L521-L532：chunked 输入输出 shape。
- L544-L556：构造 deltas，并翻转到 reversed sequence。
- L561-L570：pad 到 chunk_size 的整数倍并 reshape。
- L585-L596：构造 chunk 内 scan kernel。
- L598-L609：用矩阵乘法计算所有 chunk 的 local scan。
- L625-L637：跨 chunk 传播 `s_prev`。
- L639-L646：去 padding、翻回原顺序、计算 returns。

读完要得到的结论：生产 chunked GAE 通过 local scan 加 state propagation 保持原公式语义。

可以先跳过：矩阵 `M` 的显存/性能权衡，先确认它表达的是 chunk 内递推。

## 读完后的自检问题

1. `next_value` 和 `next_adv` 分别从哪里来？
2. chunk `[start,end)` 处理完后要交给左侧 chunk 哪两个状态？
3. 哪个测试能发现 `last_value` 被丢掉？
4. SLiME 为什么要先 pad 到 `max_len`？
5. CPU patch-test 通过以后，还不能证明哪类性能结论？
