# Source Reading Card：L36 SLiME Weight Sync Coordinator

## 主路径

1. `labs/l32_slime_rl_core/patch/reference/weight_sync.py`：本地 state_dict 合同、shape/dtype gate 和 stats。
2. `labs/l32_slime_rl_core/patch/tests/test_patch.py`：同步、mismatch、extra key 和 bytes 断言。
3. `mini_infra/slime/train.py`：generate -> train -> update_weights 的最小 loop。
4. `mini_infra/slime/ray/train_actor.py`：actor 训练后递增 `weight_version`。
5. `mini_infra/slime/ray/rollout.py`：rollout server 记录 actor_version、weight_version 和 staleness。
6. `github_repo/slime/train.py`：真实 SLiME train loop 中的初次 sync 和迭代 sync。
7. `github_repo/slime/slime/backends/megatron_utils/actor.py`：选择 updater 并调用 `weight_updater.update_weights()`。
8. `github_repo/slime/slime/backends/megatron_utils/update_weight/update_weight_from_distributed.py`：metadata + NCCL broadcast。
9. `github_repo/slime/slime/backends/sglang_utils/sglang_engine.py`：SGLang 在线更新入口和 weight version。

## 每段要得到的结论

| 文件 | 读完后要能说明 |
|---|---|
| `weight_sync.py` | 本地 copy 合同如何防止 shape/dtype 错误 |
| `test_patch.py` | 哪些行为被 patch-test 覆盖，哪些生产分支没有覆盖 |
| `mini_infra/slime/train.py` | weight sync 在 RL loop 中发生在哪一步 |
| `mini_infra/slime/ray/rollout.py` | `actor_version` 和 `weight_version` 如何形成 freshness 证据 |
| `slime/train.py` | actor train、rollout generate、offload 和 update_weights 的顺序 |
| `actor.py` | co-locate 与 disaggregate 如何选择不同 updater |
| `update_weight_from_distributed.py` | Ray 负责 metadata，NCCL 负责 tensor broadcast |
| `sglang_engine.py` | rollout engine 如何接收在线权重更新 |

## 自检

- 为什么 dtype mismatch 不应该在 coordinator 里静默 cast？
- `bytes_synced` 能帮助判断哪类性能问题？
- 真实 SLiME 为什么需要 engine lock？
- `weight_version` 没推进时，应该从哪几个入口查？
