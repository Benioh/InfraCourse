# L40 Debug Checklist：GAE Chunked Parallel

## 1. 固定现场

- 记录命令、git commit、Python/PyTorch 版本、dtype、device 和随机种子。
- 记录输入 shape、T、B、chunk_size、gamma、lambda 和 last_value shape。
- 保存 patch-test 输出、失败 test 名称、最大误差和出错索引。
- 如果报告性能，补充硬件、batch size、T、chunk_size、baseline、计时范围和 kernel/SLiME 版本。

## 2. 先看数值边界

| 问题 | 要查什么 | 可能结论 |
|---|---|---|
| 方向反了 | 三步手算样例 `[3,2,1]` | 时间维循环方向错误 |
| 最后 token 错 | `last_value`, `next_value` | 链尾 value 没进入 delta |
| chunk 边界错 | `start-1`, `start`, `end-1`, `end` | boundary adv 或 boundary value 传错 |
| 余数 chunk 错 | T % chunk_size | `end = min(start + chunk_size, T)` 错 |
| batched 崩掉 | `(B,T)` 输入和 `(B,)` last_value | 固定二维索引或广播错误 |
| 性能结论弱 | workload 和计时条件 | patch-test 只能证明数值等价 |

## 3. 沿源码主路径复查

1. `labs/l34.5_gae_chunked_parallel/patch/reference/gae_chunk.py`：确认 naive 与 chunk boundary 语义。
2. `labs/l34.5_gae_chunked_parallel/patch/tests/test_patch.py`：定位失败边界。
3. `github_repo/slime/slime/utils/ppo_utils.py`：对照 batch pad、vanilla_gae、chunked_gae、slice back 和 returns。

## 4. 常见错误判断

- 只跑 long random test，没有跑手算和 terminal value。
- 只传 boundary advantage，忘记更新 boundary value。
- 把教学顺序版当成真实 GPU kernel。
- 把 CPU patch-test 通过写成真实加速结论。
- 只看 advantages，忘记生产训练还需要 returns。

## 5. 结束条件

- 能复现失败并指出具体边界。
- patch-test 通过 reference 验收。
- 报告能区分数值等价、源码映射和真实性能证据。
