# L15 Patch · 1F1B Pipeline Schedule

## 你要交付什么

```python
def make_1f1b_schedule(num_stages: int, num_microbatches: int) -> list[list[tuple[str, int]]]:
    """Return per-stage timelines.

    schedule[s] is the ordered list of ops on stage s:
        [("F", microbatch_idx), ("B", microbatch_idx), ...]
    Each microbatch contributes exactly one F and one B per stage.
    """

def bubble_count(num_stages: int) -> int:
    """Total bubble length under the standard 1F1B schedule = 2 * (num_stages - 1)."""
```

补丁规模目标：30 到 50 行。

## 不变量

1. `num_stages > 0`，否则抛 `ValueError`。
2. `num_microbatches >= num_stages`，否则抛 `ValueError`。
3. 每个 stage 的 timeline 长度是 `2 * num_microbatches`。
4. stage `s` 的 warmup forward 数是 `num_stages - s - 1`。
5. stage `s` 的 cooldown backward 数是 `num_stages - s - 1`。
6. 同 stage 上 forward 的 microbatch index 递增。
7. 同 stage 上 backward 的 microbatch index 递增。
8. 同 stage 上对每个 `i`，`F(i)` 的位置早于 `B(i)`。

## 怎么验证

```bash
make patch-test M=l14_pipeline_1f1b
```

通过后跑一次 smoke：

```bash
bash labs/l14_pipeline_1f1b/scripts/run_pp_smoke.sh l15_pp_smoke
```

## 写完之后你能做什么

- 看懂 Megatron `forward_backward_pipelining_without_interleaving` 的三段主路径。
- 解释 GPipe、1F1B 和 interleaved 1F1B 的调度取舍。
- 用 bubble count、bubble ratio 和 stage timeline 判断 PP 配置是否合理。
