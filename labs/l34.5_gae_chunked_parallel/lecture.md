# L40：GAE Chunked Parallel

L40 进入 GAE 的实现层。L39 讨论没有 value critic 时如何从 group reward 构造 advantage；这里回到带 value critic 的 PPO 路线。训练侧已经有 rollout reward 和 value 预测，现在要把它们变成 token-level advantage，并在长上下文场景里控制反向递推的实现成本。

本讲 patch 有两个函数：`gae_naive` 是标准反向递推 baseline，`gae_chunked_parallel` 用 chunk boundary 写出等价结构。教学版仍然是 CPU 顺序实现，它验证的是数学语义和边界状态；真实速度数字需要在 SLiME 或 GPU kernel 路径里单独计时。

## 1. 本讲目标

- 写出 GAE 的 delta、advantage 递推和代码状态。
- 区分 `next_value`、`next_adv` 和 `last_value`。
- 实现 chunk boundary handoff，让 chunked 输出与 naive 数值等价。
- 对照 SLiME 的 batched pad、vanilla GAE、chunked GAE、slice back 和 returns。
- 说明 patch-test、源码映射和真实 GPU 加速比分别能证明什么。

## 2. GAE 在训练链路中的位置

PPO 训练通常先拿到 response token、reward 和 value。value 是 critic 对每个 token 后续回报的估计；GAE 把单步 TD error 累积成更平滑的 advantage。policy loss 随后使用这些 advantage 去更新 actor，returns 则用于训练 critic。

这一步介于 reward/value 产出和 policy/value loss 之间。它不负责采样 rollout，不负责 reward parser，也不负责 KL clip；它负责把沿时间维的未来信息折扣回每个 token。长上下文 RL 会让 T 变大，反向递推链的成本也随之变得可见。

## 3. Naive GAE

GAE 的单步 TD error 是：

```text
delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
```

advantage 递推是：

```text
A_t = delta_t + gamma * lambda * A_{t+1}
```

代码里有两个右侧状态。`next_value` 表示 `V(s_{t+1})`，用于计算 delta；`next_adv` 表示 `A_{t+1}`，用于计算当前 advantage。循环方向必须是 `reversed(range(T))`。每次写入 `advantages[..., t]` 后，当前 `v_t` 和 `adv` 才变成下一轮向左移动时的右侧状态。

`last_value` 是链尾的 next value。对 `(T,)` 输入，它是标量；对 `(B,T)` 输入，它是 `(B,)`。如果忽略它，最后一个 token 的 delta 会少掉 `gamma * V(s_T)`，`test_terminal_value_propagates` 会失败。

## 4. Chunk Boundary

chunked GAE 的关键问题是：对 chunk `[start,end)`，如果已经知道右侧的 `A_end`，chunk 内部就能从 `end-1` 算到 `start`。最后一个 chunk 的 `A_end` 是 0；更左侧 chunk 的 `A_end` 来自右侧相邻 chunk 的起点 advantage。

教学版按这个顺序写：

```text
for chunk from right to left:
    next_adv_in_chunk = right_boundary_adv
    running_next_value = right_boundary_value
    for t from end-1 down to start:
        run the naive formula
    right_boundary_adv = advantages[..., start]
    right_boundary_value = values[..., start]
```

这个写法保留了高性能版本的语义：chunk 内公式不变，chunk 间只交接边界状态。真实并行版可以把 chunk 内 scan 换成矩阵、Triton 或 CUDA kernel；前提仍是同一个 boundary 语义。

## 5. 长上下文瓶颈

naive GAE 的浮点工作量是 O(T)，内存也很小，但它沿时间维严格串行。T 到 32K 或 64K 时，这条链可能占用明显训练时间。把公式展开成 T x T 矩阵能并行，却会引入 O(T^2) 空间成本，长序列下很难承受。

chunked 写法避开这两个极端。它不减少数学公式本身，而是把长链切成多个局部 scan，再用 boundary state 连接。SLiME 的 `chunked_gae` 先构造 deltas、反转时间、pad 到 chunk 大小的整数倍，用上三角矩阵完成 chunk 内 local scan，再用 `s_prev` 在 chunk 之间传播状态。

## 6. SLiME 生产路径

SLiME 的 batched 入口先接收不同长度的 rewards 和 values。它会按最大 response length pad 成 `[B,T]`，再根据 `chunked` 开关选择 `vanilla_gae` 或 `chunked_gae`。计算完成后，非 CP 路径按原长度切回 list；CP 路径还要切回本 rank 的片段。

`vanilla_gae` 与 patch 的 naive 版本同构：batch 维同时处理，时间维从右向左递推。`chunked_gae` 多了生产细节：构造 reversed deltas、padding、chunk view、scan kernel、local scan、`s_prev` 传播、去 padding、翻回原时间顺序。最终返回 `(advantages, returns)`，其中 `returns = advantages + values`。

## 7. 测试闭环

七个 tests 对应七类错误。

`test_naive_known_values` 用三步手算样例检查方向。`test_chunked_matches_naive_short` 和 `test_chunked_matches_naive_long` 分别检查短序列和 T=1024 的等价。`test_chunked_handles_remainder` 覆盖 T 不能整除 chunk size。`test_chunked_one_chunk_equals_naive` 覆盖 fallback。`test_terminal_value_propagates` 检查 `last_value`。`test_batched_input` 检查 `(B,T)` shape。

这些测试不验证真实 GPU 加速，也不验证长训稳定。它们的价值是先证明 chunked 写法没有改变 GAE 数值语义。

## 8. 排障顺序

GAE 错误通常先看边界。

1. 确认 rewards、values、last_value 的 shape，时间维是否是最后一维。
2. 用 `[1,1,1]`、zero value、`gamma=1`、`lambda=1` 手算方向。
3. 检查最后一个 token 是否吃到 `last_value`。
4. 检查 chunk 边界附近的 `start-1`、`start`、`end-1`、`end`。
5. 检查余数 chunk 和 `chunk_size >= T` fallback。
6. 对照 SLiME 时，再看 pad、slice back、returns 和 CP 分支。

## Lab 验收边界

patch 命令：

```bash
make patch-test M=l34.5_gae_chunked_parallel
```

参考实现验证：

```bash
IMPL=reference make patch-test M=l34.5_gae_chunked_parallel
```

本讲没有 dedicated smoke/drill target。CPU 验收以 patch-test 为准；真实速度结论需要记录硬件、T、batch size、dtype、chunk_size、baseline、计时范围和 SLiME/GPU kernel 版本。

---

## 补充：GAE 数学推导与工程直觉

### GAE 完整推导

从 TD(λ) 的角度理解 GAE。单步 TD error：

```
δ_t = r_t + γ·V(s_{t+1}) - V(s_t)
```

GAE 是不同步数 advantage 估计的指数加权平均：

```
A^(1)_t = δ_t                                    = r_t + γV_{t+1} - V_t
A^(2)_t = δ_t + γλ·δ_{t+1}                       = r_t + γr_{t+1} + γ²V_{t+2} - V_t  (加权)
A^(k)_t = Σ_{l=0}^{k-1} (γλ)^l · δ_{t+l}

A^GAE_t = Σ_{l=0}^{∞} (γλ)^l · δ_{t+l}  = δ_t + γλ · A^GAE_{t+1}
```

最后一行就是递推公式。

### γ 和 λ 的意义

| 参数 | 控制什么 | 过大 | 过小 |
|---|---|---|---|
| γ (gamma) | 未来 reward 的折扣 | 长视野但高方差 | 短视野但低方差（近视） |
| λ (lambda) | 多步 bootstrap 程度 | 高方差但低偏差（Monte Carlo 方向） | 低方差但高偏差（TD(0) 方向） |

**实践中常用值**：
- γ = 0.99 或 1.0（token-level RL 中通常用 1.0，因为 episode 就是一条 response）
- λ = 0.95（平衡偏差和方差）

### Token-level vs Sequence-level

在 LLM RL 中，reward 通常只在序列末尾给出（稀疏 reward）：

```
r_0 = r_1 = ... = r_{T-2} = 0,  r_{T-1} = R (final reward)
```

此时 GAE 的效果是把末尾 reward 通过 value baseline 分配到每个 token。如果 value critic 很准（V ≈ 真实 value），advantage 会告诉每个 token "你比预期好/差了多少"。

### 为什么 Chunked 不改变结果

GAE 递推 `A_t = δ_t + γλ·A_{t+1}` 是一个线性递推。对任意分段 [start, end)：

```
已知 A_end（右边界的 advantage）
A_{end-1} = δ_{end-1} + γλ·A_end
A_{end-2} = δ_{end-2} + γλ·A_{end-1}
...
A_start = δ_start + γλ·A_{start+1}
```

只要右边界 `A_end` 正确，chunk 内结果和全局递推完全等价。这就是 chunked GAE 的数学保证。

### Returns 的计算

```
returns_t = advantages_t + values_t
```

这是 GAE 论文的标准做法。returns 用于训练 value critic（value loss = MSE(V(s_t), returns_t)）。advantage 用于 policy loss。两者从同一次 GAE 递推中得出。
