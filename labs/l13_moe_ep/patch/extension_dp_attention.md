# L13 扩展 · DP Attention（DeepSeek 风格 MoE 必备）

> 这是 L13 的可选扩展。**不在 patch-test 范围内**。
> 完成后你会理解 DeepSeek-V2 / V3 的 DP Attention 为什么能 1) 避免 KV cache 重复，
> 2) 在 MoE 模型上释放 EP 的吞吐。

## 背景

DeepSeek 的 MLA 只有 1 个 KV head。如果按经典 TP 切，每个 TP rank 上都要复制一份完整 KV cache，
得不偿失。SGLang 的解法是 **DP Attention**：attention 走 DP，每个 DP rank 处理自己的 batch；
MoE 走 EP，跨 DP rank 互通 expert 输出。

这是 SGLang vs vLLM 的关键差距之一，也是 RL co-locate MoE 模型时被反复提到的优化。

## 你要扩展什么

在 `patch/starter/moe_router.py` 同目录新建 `patch/starter/dp_attention.py`：

```python
def dp_attention_forward(
    qkv: torch.Tensor,           # (local_bs, seq_len, hidden_size)
    dp_world_size: int,          # DP 维度
    dp_rank: int,                # 当前 rank
) -> torch.Tensor:
    """每个 DP rank 只处理自己的 batch slice，attention 不跨 DP 通信。"""

def all_gather_dp_to_ep(
    attn_out: torch.Tensor,
    dp_world_size: int,
) -> torch.Tensor:
    """从 DP 切回 full batch，准备进入 MoE EP 阶段。"""

def reduce_scatter_ep_to_dp(
    moe_out: torch.Tensor,
    dp_world_size: int,
    dp_rank: int,
) -> torch.Tensor:
    """从 EP 收回 DP 切片。"""
```

## 不变量

1. `dp_attention_forward` 在 `dp_world_size == 1` 时退化为普通 attention。
2. `all_gather_dp_to_ep ∘ dp_attention_forward` 在所有 DP rank 上结果应当与 vanilla TP attention 一致。
3. KV cache 在 DP Attention 模式下每个 rank 只持有自己 DP 切片的 cache（单 KV head，无重复）。
4. MoE EP 阶段必须看到 full batch，否则 router 决策不一致。

## 怎么验证

自己加 `patch/tests/test_dp_attention.py`（可选）：

```python
def test_dp_attention_no_kv_duplication():
    # 模拟 dp_world_size=4，验证每个 rank 的 KV cache 大小是 vanilla TP 的 1/4
    ...
```

## 写完之后你能做什么

- 解释 DeepSeek MLA + DP Attention + MoE EP 三件套为什么是大 MoE 模型的最优组合。
- 估算切换 DP / TP 的临界点：什么情况下 DP 比 TP 更划算（提示：KV head 数 vs TP size）。
- 看懂 SGLang `dp_attention/` 目录的实现。

## 配套阅读

- `github_repo/Awesome-ML-SYS-Tutorial/sglang/dp-attention/readme.md`
- `github_repo/Awesome-ML-SYS-Tutorial/rlhf/sys-design/readme-4.md` —— DeepSeek MoE 与 EP 二次开发
