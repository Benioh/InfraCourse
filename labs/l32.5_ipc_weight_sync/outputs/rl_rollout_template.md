# L37 CUDA IPC Weight Sync 复盘模板

## Run 信息

- 日期：
- 机器 / GPU：
- 命令：
- run 目录：
- git commit：
- world_size / tp_rank：
- tensor shape / dtype：

## 预期

- 这次验证的机制：
- 只改变的变量：
- 成功标准：
- 已知边界：

## Handle 检查

| 检查项 | 数值 / 路径 | 判断 |
|---|---:|---|
| tensor_bytes |  |  |
| handle_bytes |  |  |
| handle/data ratio |  |  |
| data_ptr 是否一致 |  |  |
| pool size before / after |  |  |

## Rank 和 Flush

| 检查项 | 结果 | 判断 |
|---|---|---|
| rank 0 是否收齐 world_size |  |  |
| 非 0 rank 返回值 |  |  |
| LST values 顺序 |  |  |
| update 后 state keys |  |  |
| flush_cache 触发位置 |  |  |

## 源码对应

| 现象 | 源码位置 | 判断 |
|---|---|---|
| handle meta | `patch/reference/ipc_weight_sync.py` |  |
| SGLang unwrap | `github_repo/sglang/python/sglang/srt/model_executor/model_runner.py` |  |
| torch reduction patch | `github_repo/sglang/python/sglang/srt/utils/patch_torch.py` |  |
| flattened bucket | `github_repo/sglang/python/sglang/srt/weight_sync/tensor_bucket.py` |  |
| SLiME gather/update | `github_repo/slime/slime/backends/megatron_utils/update_weight/update_weight_from_tensor.py` |  |

## 结论

- 本次能证明什么：
- 本次不能证明什么：
- 如果 handle 太大，下一步检查：
- 如果 data_ptr 不一致，下一步检查：
- 如果 rank gather 错位，下一步检查：
- 下一次实验只改的变量：
