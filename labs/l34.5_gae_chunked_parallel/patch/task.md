# L40 Patch · GAE Chunked Parallel

## 你要交付什么

实现两个函数：

```python
def gae_naive(
    rewards: torch.Tensor,        # (T,) or (B, T)
    values: torch.Tensor,         # (T,) or (B, T)
    last_value: torch.Tensor,     # scalar or (B,)
    gamma: float,
    lam: float,
) -> torch.Tensor: ...

def gae_chunked_parallel(
    rewards: torch.Tensor,
    values: torch.Tensor,
    last_value: torch.Tensor,
    gamma: float,
    lam: float,
    chunk_size: int,
) -> torch.Tensor: ...
```

`gae_naive` 是标准反向递推 baseline。`gae_chunked_parallel` 要在输出上与 `gae_naive` 数值等价，并正确处理 chunk boundary。

## 不变量

1. 时间维永远是最后一维。
2. `delta = r_t + gamma * next_value - v_t`。
3. `adv = delta + gamma * lam * next_adv`。
4. `last_value` 是链尾 next value；一维输入为标量，batched 输入为 `(B,)`。
5. `chunk_size >= T` 时直接退回 naive。
6. T 不能整除 chunk_size 时，最后一个短 chunk 也必须正确。
7. 每个 chunk 从右向左处理；处理完后把 `advantages[..., start]` 和 `values[..., start]` 交给左侧 chunk。
8. `(B,T)` 输入输出 shape 必须保持 `(B,T)`。

## 怎么验证

```bash
make patch-test M=l34.5_gae_chunked_parallel
```

参考实现验收：

```bash
IMPL=reference make patch-test M=l34.5_gae_chunked_parallel
```

本讲没有 dedicated smoke/drill target；CPU 验收以 patch-test 为准。

## 写完之后你能做什么

- 解释 GAE 的 `next_value` 和 `next_adv` 递推状态。
- 判断 chunk boundary 是否传错 advantage 或 value。
- 对照 SLiME `chunked_gae` 说明 pad、local scan、`s_prev` 和 returns 的作用。
- 在真实长上下文实验中区分数值等价、源码映射和 GPU 加速证据。
