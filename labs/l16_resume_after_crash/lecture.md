# L17：Crash Resume

## 0. 本讲目标

- 理解训练进程被 kill 时 checkpoint 可能处于哪些不完整状态。
- 能解释 `.tmp + fsync + os.replace` 的 crash-safe 写入边界。
- 能说明 latest marker 更新顺序为什么必须晚于 checkpoint 文件提交。
- 能实现 `load_latest` 清理 dangling `.tmp` 并回退到 committed checkpoint。
- 能用 baseline vs crash+resume loss 对齐验证恢复状态完整性。

## 1. 问题入口：进程被杀时，保存流程可能停在任何一行

训练作业可能因为节点抢占、OOM、作业时间到期、运维重启或手动 kill 退出。退出发生的位置不可控：可能在 checkpoint payload 刚写一半，可能在文件写完但 marker 未更新，可能在 marker 已更新但数据还没真正落盘。Crash-resume 的目标是让重启流程只读取完整提交的 checkpoint，并恢复到可继续训练的状态。

L16 讲的是 checkpoint payload 和拓扑兼容性。L17 继续补上写入过程的可靠性。一个 checkpoint 要成为可恢复状态，必须同时满足：文件内容完整、latest marker 指向它、恢复时能清理上次崩溃留下的临时文件、payload 里包含 step/model/optimizer/RNG 等状态。

本关 patch 不模拟真实多机训练，只用 JSON checkpoint 和一个 tiny 优化过程测试关键合同。这个缩小版足够暴露常见错误：直接写正式文件、读取 `.tmp`、先更新 marker、丢失 optimizer 或 RNG、同 step 重复保存留下多个文件。

## 2. Atomic save：把“写文件”拆成提交协议

**定义：** atomic save 是先写临时文件，再把临时文件原子替换成正式文件的保存方式。本关的 `atomic_save(payload, target)` 要写 `<target>.tmp`，对文件执行 flush 和 `os.fsync`，然后用 `os.replace(tmp, target)` 提交。

直觉上，`.tmp` 是“未提交状态”，正式文件是“已提交状态”。进程在写 `.tmp` 时崩溃，loader 看到 `.tmp` 应该删除它；进程在 `os.replace` 后崩溃，正式文件已经可作为候选 checkpoint。

`os.replace` 在同一文件系统内提供原子替换语义。它避免 reader 看到同一个文件的半写内容。`fsync` 的作用是尽量把文件内容推到存储层，降低系统崩溃后数据丢失的概率。生产环境还可能需要 fsync 目录、分布式文件系统语义确认和 rank 间 barrier；本关只要求文件级最小合同。

## 3. Latest marker：提交顺序决定恢复入口

latest marker 记录最近一次成功保存的 step。`save_step` 应该先提交 `iter_0000050.pt`，再更新 `latest_checkpointed_iteration.txt`。如果顺序反过来，进程可能在 marker 指向 step 50 后崩溃，而 step 50 文件还不存在或只写了一半。

本关的 marker 也用 `atomic_save` 写入，内容是 `{"step": step}`。`load_latest` 会先清理 `.tmp`，再优先尝试 marker 指向的 checkpoint。如果 marker 坏了或指向的文件不存在，reference 会回退到最高编号的 committed checkpoint。这个回退策略服务于教学场景：学生能观察到“完整提交文件”和“marker”两个边界。真实训练框架通常会更严格地处理 marker 不一致，避免静默读取非预期 checkpoint。

关键原则是：loader 只能读取 committed checkpoint。`.tmp` 不属于 committed 集合。

## 4. 恢复状态：step、model、optimizer、RNG 都要对齐

Crash resume 要恢复的是训练轨迹，单个文件只是承载状态的容器。最小 payload 至少包含：

- `step`：恢复后从哪个 step 继续。
- `model_state`：当前模型参数。
- `optimizer_state`：动量、二阶矩或本关里的 `m`。
- `rng_state`：随机数状态，用于 dropout、shuffle、采样等复现。
- `extra`：loss history、数据游标或其他训练控制状态。

如果只恢复 model_state，下一步 forward 可能能跑，但 optimizer 的历史动量已经丢失。若 RNG 丢失，后续 dropout mask 或数据顺序可能不同。若 step 丢失，scheduler 或日志可能错位。L17 的测试用一个简化优化过程证明：保存 step 4 的 model 和 optimizer 后，恢复继续执行同一批 gradients，loss 序列应与 baseline 对齐。

## 5. Idempotent save：重复保存同一步不能制造混乱

真实系统里，同一个 step 的保存可能被重试。比如 signal handler 触发一次保存，训练 loop 的周期性保存又触发一次；或者上层调度系统重启后再次保存同一个 step。`save_step(dir, step=N, ...)` 应该是幂等的：可以覆盖同名文件，也可以在内容相同时 no-op，但不能生成多个互相竞争的 checkpoint。

本关 reference 通过固定文件名 `iter_{step:07d}.pt` 达到幂等。同 step 重复保存仍只有一个正式文件。测试只检查文件数量不增加，不强制要求 payload 保持旧值还是覆盖新值，因为这属于更高层的策略选择。

## 6. Patch 机制：三个函数的边界

`atomic_save(payload, target)` 的输入是 JSON 可序列化 payload 和目标路径。中间状态是 `<target>.tmp`。输出是正式 target 文件存在，tmp 文件消失。

`load_latest(checkpoint_dir)` 的输入是 checkpoint 目录。中间状态包括清理 dangling `.tmp`、枚举 `iter_*.pt`、读取 marker。输出是最新 committed payload；如果没有可用 checkpoint，返回 `None`。

`save_step(checkpoint_dir, step, model_state, optimizer_state, rng_state, extra)` 的输入是 step 和训练状态。中间状态是 payload 构造和两次 atomic save。输出是正式 checkpoint path。

下面的最小示例演示 reference 行为：

```bash
python - <<'PY'
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, "labs/l16_resume_after_crash/patch")
from reference.crash_safe import load_latest, save_step

with TemporaryDirectory() as tmp:
    root = Path(tmp)
    save_step(root, 1, {"w": 1.0}, {"m": 0.1}, {"seed": 7})
    (root / "iter_0000002.pt.tmp").write_text("PARTIAL")
    payload = load_latest(root)
    print(payload["step"], (root / "iter_0000002.pt.tmp").exists())
PY
```

关键输出应表示恢复到 step 1，且 dangling tmp 已被删除。

## 7. Megatron 对照：真实框架多了哪些状态

Megatron 保存 checkpoint 时会收集 RNG state、rerun state、model、optimizer、scheduler 和 floating point operation 计数。保存完成后，它会通过 tracker 文件记录 latest iteration；异步保存时，tracker 更新会被放进 finalize callback，避免异步任务还没完成时就宣称 checkpoint 可用。

加载时，Megatron 会设置 checkpoint version，读取 iteration，恢复 consumed samples，加载模型、optimizer、scheduler 和 RNG。这里的 RNG 恢复包括 Python、NumPy、Torch、CUDA 和 tensor parallel RNG tracker。L17 patch 只用一个 `rng_state` dict 表达这类状态，真实框架里的状态更多、更分散。

这也是 crash drill 要比较 loss 的原因。文件存在只能证明写入结果可读；loss 对齐才能证明 model、optimizer、step 和随机序列共同恢复到正确位置。

## 8. Drill：baseline vs crash+resume

`scripts/run_crash_drill.py` 用同一个 seed 生成固定 gradients。baseline 从初始状态连续训练 100 步。crash path 从同一初始状态训练到 step 50，保存 checkpoint，加载后继续剩余 50 步。脚本逐步写入 baseline loss、resumed loss 和相对误差，最后输出 `max_rel_diff`。

默认验收是 `max_rel_diff <= 0.01`。reference 实现通常能达到 0，因为它保存了模型和 optimizer 状态，且 gradients 由同一个 seed 预先生成。真实训练不一定 bit-exact，尤其存在非确定性 kernel、分布式通信顺序或数据 loader worker；但应该设定可解释的容忍阈值，并记录比较对象、硬件、seed、batch 和 deterministic 配置。

如果 drill 失败，先看三个位置：

- `artifacts/checkpoint/`：是否留下 `.tmp`，latest marker 指向哪里。
- `artifacts/crash_drill.json`：`max_rel_diff` 和 `accept`。
- `metrics.jsonl`：从哪个 step 开始 baseline 和 resumed loss 分叉。

## 9. 小结

L17 的知识链路是：进程可能在保存任意位置崩溃，checkpoint 写入要有临时文件和原子提交边界，latest marker 要晚于 checkpoint 提交，恢复时要删除 `.tmp` 并只读 committed 文件，payload 要恢复 step、model、optimizer 和 RNG，最终用 baseline/resume loss 对齐证明训练轨迹连续。patch 验证最小文件合同；Megatron 源码把同一套思路扩展到分布式 rank、异步保存、optimizer shard 和 RNG tracker。
