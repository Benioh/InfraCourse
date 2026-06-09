# L40 GAE Chunked Parallel 复盘模板

## Run 信息

- 日期：
- 命令：
- git commit：
- Python / PyTorch：
- device / dtype：
- B / T：
- chunk_size：
- gamma / lambda：
- last_value shape：

## 数值证据

| 检查项 | 数值 / 结果 | 解释 |
|---|---|---|
| naive known values |  |  |
| short allclose max diff |  |  |
| long allclose max diff |  |  |
| remainder chunk |  |  |
| single-chunk fallback |  |  |
| terminal value |  |  |
| batched shape |  |  |

## 源码对应

| 现象 | 源码位置 | 判断 |
|---|---|---|
| naive 递推 | `patch/reference/gae_chunk.py` |  |
| chunk boundary | `patch/reference/gae_chunk.py` |  |
| SLiME pad/slice | `github_repo/slime/slime/utils/ppo_utils.py` |  |
| SLiME local scan | `github_repo/slime/slime/utils/ppo_utils.py` |  |

## 性能证据

| 条件 | 数值 |
|---|---|
| baseline |  |
| workload |  |
| hardware |  |
|计时范围 |  |
| speedup |  |

## 结论

- 本次能证明什么：
- 不能证明什么：
- 下一步要改的配置、代码或实验：
