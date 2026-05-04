# L05.7 Patch · 1F1B Pipeline Schedule

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

补丁规模目标：30–50 行。

## 不变量

1. `num_microbatches >= num_stages`，否则抛 `ValueError`
2. 每个 stage 的 timeline 长度 = `2 * num_microbatches`
3. stage `s` 的 warmup forward 数 = `num_stages - s - 1`
4. stage `s` 的 cooldown backward 数 = `num_stages - s - 1`
5. 同 stage 上 forward 的 microbatch idx 严格递增 0,1,2,...
6. 同 stage 上 backward 的 microbatch idx 严格递增 0,1,2,...
7. 同 stage 上对每个 i：F(i) 的位置 < B(i) 的位置
8. 任意 stage `s+1` 的 F(i) 必须在 stage `s` 的 F(i) 之后（pipeline 依赖，下文不强制）

## 怎么验证

```bash
make patch-test M=l14_pipeline_1f1b
```

## 写完之后你能做什么

- 看懂 `Megatron-LM/megatron/core/pipeline_parallel/schedules.py`
- 解释 GPipe vs 1F1B vs interleaved 1F1B 的取舍
- 在 capstone 用 PP 切 13B+ 模型
