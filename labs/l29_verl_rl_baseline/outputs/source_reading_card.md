# Source Reading Card：L31 Adaptive KL Controller

## 主路径

1. `labs/l29_verl_rl_baseline/patch/starter/kl_controller.py`：看公式注释和三个 TODO。
2. `labs/l29_verl_rl_baseline/patch/reference/kl_controller.py`：看 `self.value`、相对误差、clip 和乘法更新。
3. `labs/l29_verl_rl_baseline/patch/tests/test_patch.py`：看五个测试如何锁住初始、增大、减小、不变和 clip 边界。
4. `labs/l29_verl_rl_baseline/scripts/reward_math.py`：看最终数字提取、reward、失败原因和 self-test。
5. `labs/l29_verl_rl_baseline/scripts/run_verl_lab.py`：看 run 目录、reward self-test artifact、`metrics.jsonl` 和 report。
6. `github_repo/slime/slime/utils/ppo_utils.py`：看 KL estimator、ratio clip、`kl_coef` 进入 returns/advantages 的位置。
7. `github_repo/slime/train_async.py`：看 rollout manager、actor/critic train、weight sync 和 eval 的循环。

## 每段要得到的结论

| 文件 | 读完后要能说明 |
|---|---|
| `patch/starter/kl_controller.py` | 学生需要保存哪些状态，`update` 应如何改变 `self.value` |
| `patch/reference/kl_controller.py` | 公式如何直接落成代码，哪些配置假设由上层保证 |
| `patch/tests/test_patch.py` | patch-test 验证的行为边界，以及没有覆盖的真实训练环节 |
| `scripts/reward_math.py` | reward parser 如何处理缺失数字、逗号数字和错误答案 |
| `scripts/run_verl_lab.py` | 本地 smoke 能证明数据、reward 和 metrics 闭合，不能证明 GPU PPO 稳定 |
| `ppo_utils.py` | KL estimator 在 controller 之前，ratio clip 和 KL penalty 是两层约束 |
| `train_async.py` | 真实 RL infra 还包含 rollout、actor/critic、weight sync 和指标上报 |

## 可先跳过

- SLiME 里的 context parallel 细节，先理解 `kl_coef` 如何影响 token-level rewards。
- Ray placement group 的资源分配细节，先理解 async train loop 的顺序。
- verl 官方命令接入，L31 只要求 CPU-safe smoke 和源码对照。

## 自检

- `current_kl == target_kl` 时，为什么 controller 不改变系数？
- clip 限制的是 proportional error 还是 `kl_coef`？
- reward self-test 失败时，为什么不继续调 PPO 参数？
- `compute_approx_kl` 和 `AdaptiveKLController` 在链路中的先后关系是什么？
- ratio clip、KL penalty、reward parser 分别约束哪类风险？
