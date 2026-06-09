# L40 Source Reading Card：GAE Chunked Parallel

## 主路径

1. `labs/l34.5_gae_chunked_parallel/patch/starter/gae_chunk.py`：学生要补齐的 naive 和 chunked GAE。
2. `labs/l34.5_gae_chunked_parallel/patch/reference/gae_chunk.py`：最小递推和 boundary handoff 参考实现。
3. `labs/l34.5_gae_chunked_parallel/patch/tests/test_patch.py`：七个边界测试。
4. `github_repo/slime/slime/utils/ppo_utils.py`：SLiME vanilla GAE、chunked GAE、pad/slice 和 returns。

## 阅读方法

1. 先看 `next_value` 和 `next_adv` 如何初始化和推进。
2. 再看 chunked 函数如何从右到左交接 boundary。
3. 接着看测试如何覆盖 known values、remainder、terminal 和 batched shape。
4. 最后看 SLiME 如何把 local scan、`s_prev` 和 returns 接回训练路径。

## 自检

- 我能否解释 `last_value` 对最后一个 token 的影响？
- 我能否指出 chunk `[start,end)` 结束后交给左侧的两个状态？
- 我能否说明 patch-test 和真实 GPU 加速证据的边界？
