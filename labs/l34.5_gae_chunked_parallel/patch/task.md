# L11.7 Patch · GAE 分 chunk 并行（数值与 naive 等价）

## 你要交付什么

```python
def gae_naive(
    rewards: torch.Tensor,        # (T,) 或 (B, T)
    values: torch.Tensor,         # (T,) 或 (B, T)
    last_value: torch.Tensor,     # 标量 或 (B,) — value(s_T)
    gamma: float,
    lam: float,
) -> torch.Tensor:
    """标准反向递推 GAE。教学 baseline。"""

def gae_chunked_parallel(
    rewards: torch.Tensor,
    values: torch.Tensor,
    last_value: torch.Tensor,
    gamma: float,
    lam: float,
    chunk_size: int,
) -> torch.Tensor:
    """分 chunk 并行版。要求与 gae_naive 输出在数值上 allclose(atol=1e-6)。"""
```

**禁止**：直接调用任何已实现的 GAE library。
**允许**：torch.zeros / cumsum / 切片 / 简单 loop。

补丁规模目标：60–100 行。

## 数学

GAE 反向递推（naive）：

$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t), \quad
A_t = \delta_t + \gamma\lambda A_{t+1}, \quad A_T = 0
$$

把序列分成长度为 $k$ 的 chunk：$[0, k), [k, 2k), \dots$。

**关键观察**：在 chunk $[ck, ck+k)$ 内部，如果暂时假设 $A_{ck+k} = 0$，可以**只用
chunk 内部的 reward / value** 算出 chunk-local 的 $A^{\text{local}}_t$。

**boundary correction**：真实的 $A_t$ 与 chunk-local 之间的差是

$$
A_t = A^{\text{local}}_t + (\gamma\lambda)^{ck+k-t} \cdot A_{ck+k}
$$

也就是说每个 chunk 只需要从下一个 chunk 拿一个标量（$A_{ck+k}$），就能修正全
chunk 的输出。这让"chunk 内部递推"可以并行化（CUDA / 矩阵化都行）；chunk 之
间的 boundary 链长度从 $T$ 缩到 $T/k$。

## 算法（推荐写法）

```text
1. 反向遍历 chunk c = num_chunks-1, ..., 0:
     a. 在 chunk 内部反向递推算 A_local（边界处用 A_next_chunk_start 当 A_{T_chunk}）
     b. 记下 A_local[chunk_start]，作为下一轮（c-1）的 next_chunk_start
2. 输出 advantages = 拼接所有 chunk 的 A_local
```

为什么这"够并行"？因为步骤 (a) 的 chunk 内部递推工作量是 O(k)，与其它 chunk
**互不依赖**——全部 chunk 可以同时算 chunk-local 部分（GPU kernel / SIMD），
chunk 之间只交换一个标量。chunk 数量大时（T=32K, k=64 → 512 chunks），
parallelism 巨大。

## 不变量

1. 与 naive 在所有 (T, chunk_size, gamma, lam) 组合下 `allclose(atol=1e-6)`。
2. `chunk_size >= T` 时退化为 naive。
3. T 不被 chunk_size 整除时最后一个 chunk 长度更短，结果仍正确。
4. 支持 batched 输入：`rewards.shape == (B, T)`，输出同 shape。

## 怎么验证

```bash
make patch-test M=l34.5_gae_chunked_parallel
```

## 写完之后你能做什么

- 解释为什么 RL 长上下文场景下 GAE 是 CPU bottleneck，以及 chunk 化为什么能解决。
- 解释 boundary correction 的几何意义：一个标量传播了整 chunk 的链式衰减影响。
- 在面试里讲清"算法不变 + 实现精明 → 100-300× 加速"的工程美学。
- 在 Capstone Stage C 用这个 GAE 替换 verl 默认实现，量出 step time 下降。
