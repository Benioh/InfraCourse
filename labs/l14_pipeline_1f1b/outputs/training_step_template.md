# L15 Pipeline Parallel 1F1B 复盘模板

## Run 信息

- 日期：
- 机器 / GPU：
- 命令：
- 配置文件：
- run id：
- git commit：
- 实现来源：starter / reference / Megatron

## 并行与 batch 配置

| 项 | 值 |
|---|---|
| pipeline stages (`D`) |  |
| microbatches (`N`) |  |
| global batch size |  |
| microbatch size |  |
| forward time 模型或实测 |  |
| backward time 模型或实测 |  |
| virtual pipeline size |  |

## 预期

- 这次运行要验证的机制：
- 理论 bubble count：
- 预期 total ops：
- 成功标准：

## 观察指标

| 指标或 artifact | 数值 / 路径 | 解释 |
|---|---|---|
| `bubble_count` |  | 是否等于 `2 * (D - 1)` |
| `bubble_ratio` |  | 空泡占最长 stage 时间的比例 |
| `total_ops` |  | 是否等于 `D * N * 2` |
| `schedule_head` |  | stage 0 的 warmup 是否正确 |
| `schedule_last` |  | 最后 stage 是否很快进入 B/F 交替 |
| P2P wait |  | 真实训练时填写 |
| stage step time |  | 真实训练时填写 |

## 源码对应

| 现象 | 源码位置 | 判断 |
|---|---|---|
| warmup 数 | `patch/reference/pp_schedule.py` L15-L22 或 Megatron L2166-L2169 |  |
| steady F/B 交替 | `patch/reference/pp_schedule.py` L22-L27 或 Megatron L2268-L2338 |  |
| cooldown backward | `patch/reference/pp_schedule.py` L28-L31 或 Megatron L2340-L2364 |  |
| bubble 公式 | `patch/reference/pp_schedule.py` L35-L36 |  |

## 结论

- 本次能证明什么：
- 本次不能证明什么：
- 下一步要改的配置、代码或实验：
