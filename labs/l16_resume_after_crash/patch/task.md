# L17 Patch · Crash-safe Checkpoint Write & Resume

## 你要交付什么

实现 `patch/starter/crash_safe.py` 中的三个函数：

```python
def atomic_save(payload: dict[str, Any], target: Path) -> None:
    """Atomically write JSON-serializable payload to target."""

def load_latest(checkpoint_dir: Path) -> dict[str, Any] | None:
    """Return latest committed checkpoint payload, or None.

    Must clean up any dangling .tmp files left behind by previous crashes.
    """

def save_step(
    checkpoint_dir: Path,
    step: int,
    model_state: dict[str, Any],
    optimizer_state: dict[str, Any],
    rng_state: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> Path:
    """Idempotent crash-safe save."""
```

补丁规模目标：50 到 90 行。

## 不变量

1. `atomic_save` 必须先写 `<target>.tmp`。
2. `atomic_save` 写完后必须 flush，并尽量调用 `os.fsync`。
3. `atomic_save` 必须用 `os.replace(tmp, target)` 提交正式文件。
4. `load_latest` 必须清理 dangling `.tmp` 文件。
5. `load_latest` 只能返回 committed `iter_*.pt` payload；没有 checkpoint 时返回 `None`。
6. `save_step` payload 必须包含 `step`、`model_state`、`optimizer_state`、`rng_state`、`extra`。
7. `save_step` 必须先保存 step checkpoint，再更新 latest marker。
8. 同 step 重复调用 `save_step` 必须幂等，不能留下重复正式文件。
9. crash+resume 后的 loss 应能与不中断 baseline 对齐。

## 怎么验证

```bash
make patch-test M=l16_resume_after_crash
```

通过后跑一次 drill：

```bash
IMPL=reference bash labs/l16_resume_after_crash/scripts/run_crash_drill.sh l17_validation
```

## 写完之后你能做什么

- 解释为什么训练 checkpoint 需要 tmp+fsync+replace 的提交边界。
- 在训练脚本里区分 half-written file 和 committed checkpoint。
- 用 baseline/resume loss diff 验证恢复状态是否连续。
