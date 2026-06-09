# Debug Checklist：L23 SDPA / FlashAttention benchmark

## 1. 先固定现场

- 记录命令、配置文件、git commit、Python 版本、PyTorch 版本、CUDA 版本、GPU 型号和驱动。
- 记录 `B/H/T/D`、dtype、device、causal、mask、dropout、head_dim、iters、warmup 和随机种子。
- 保存 `bench.csv`、`metrics.jsonl`、`bench_summary.json`、`config.resolved.yaml` 和 `command.sh`。
- 区分当前运行是 patch-test、CPU smoke、GPU benchmark，还是真实 serving/training profile。

## 2. 先判断是哪一类问题

| 问题 | 先看什么 | 常见原因 |
|---|---|---|
| 数值不一致 | fp32 小 shape、causal、mask、scale | mask 方向错、漏除 `sqrt(D)`、dtype 转换不一致 |
| 没有 speedup | device、dtype、seq_len、head_dim、backend | CPU 路径、fp32 fallback、序列太短、head_dim 不支持 |
| 峰值显存不下降 | 是否 CUDA、是否 reset peak、是否 materialize `[T,T]` | 统计点不对、baseline 未真正构造 score、缓存未清 |
| 计时波动大 | warmup、iters、synchronize、其他进程 | CUDA 异步计时、GPU 被共享、iters 太少 |
| artifact 不完整 | run_dir、metrics.jsonl、summary、command snapshot | drill 提前退出、torch 缺失、配置路径错误 |

## 3. 正确性排查顺序

1. 用 `seq_len=16`、`dtype=float32`、`device=cpu` 跑 eager baseline。
2. 检查 `scores = q @ k.transpose(-2, -1) / sqrt(D)`。
3. causal 时检查上三角未来位置是否填为 `-inf`。
4. 对比 `eager_attention()` 和 `flash_attention()` 的 `max_abs_diff`。
5. 若误差只在半精度大 shape 出现，先回到 fp32 小 shape 排除公式错误。

## 4. 性能排查顺序

1. 确认运行在 CUDA 上，dtype 是 bf16 或 fp16。
2. 确认 `seq_len` 足够长，短序列可能被 launch 开销主导。
3. 确认计时前后有 `torch.cuda.synchronize()`。
4. 确认 warmup 已经跑过，iters 不低于配置要求。
5. 用 profiler 或 backend 约束确认 SDPA 是否走目标 backend。
6. 报告 speedup 时同时写出比较对象和完整 shape。

## 5. 显存排查顺序

1. 仅在 CUDA 路径解释 `peak_mem_*_mb`。
2. 每段计时前调用 `torch.cuda.reset_peak_memory_stats()`。
3. eager baseline 必须显式构造 `[B,H,T,T]` 的 `scores`。
4. SDPA 路径不要额外保存完整 attention matrix。
5. 如果峰值异常，缩小 shape 复现，再逐步放大 `T`。

## 6. 结束条件

- 一个最小命令能复现问题。
- 正确性、性能和显存问题已经分层。
- artifact 足以恢复命令、配置、shape 和指标。
- 源码主路径中能指出问题发生在公式、mask、dispatch、计时还是统计。
- 结论写入 `serving_metrics_template.md`，并标明不能外推的条件。
