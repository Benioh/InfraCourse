# L05.8.5 Patch · Crash-safe checkpoint write & resume

## 你要交付什么

```python
def atomic_save(payload: dict, target: Path) -> None:
    """Write payload to <target>.tmp then os.rename to target. Must include fsync."""

def load_latest(checkpoint_dir: Path) -> dict | None:
    """Return the most recent fully-committed checkpoint, or None if none exist.
    Must clean up dangling .tmp files."""

def save_step(
    checkpoint_dir: Path,
    step: int,
    model_state: dict,
    optimizer_state: dict,
    rng_state: dict,
    extra: dict | None = None,
) -> Path:
    """Atomically write iter_<step:07d>.pt and update latest_checkpointed_iteration.txt
    only after the rename succeeded. Idempotent on (dir, step)."""
```

补丁规模目标：60–100 行。

## 不变量

1. `atomic_save` 必须 (a) 写 `.tmp`，(b) flush + fsync，(c) os.rename
2. `load_latest` 必须先扫描所有 `iter_*.pt`，挑选最大 step；遇 `.tmp` 直接 unlink
3. `save_step` 必须只在 rename 完成后更新 `latest_checkpointed_iteration.txt`
4. `save_step(dir, step=N, ...)` 重复调用不抛错，且不留下重复文件
5. `payload` 中包含 step / model_state / optimizer_state / rng_state；resume 后必须能复现一致 loss

## 怎么验证

```bash
make patch-test M=l16_resume_after_crash
```

## 写完之后你能做什么

- 解释为什么 Megatron / DeepSpeed 都用 `tmp + rename` 模式
- 在自己的训练脚本里加 SIGINT/SIGTERM handler 触发 graceful save
- 在 capstone 里给训练加自动 resume
