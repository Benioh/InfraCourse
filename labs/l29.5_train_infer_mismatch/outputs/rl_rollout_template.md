# L32 Train-Infer Mismatch 复盘模板

## Run 信息

- 日期：
- 命令：
- 配置文件：
- git commit：
- 框架版本：
- 机器 / GPU：
- dtype：
- 数据或 workload：

## 预期

- 本次要验证的机制：
- 比较对象：
- 成功标准：
- 已知边界：

## 输入对齐

| 项目 | 证据路径或数值 | 判断 |
|---|---|---|
| rollout tokens |  |  |
| training tokens |  |  |
| rollout logprob shape |  |  |
| training logprob shape |  |  |
| loss mask shape |  |  |

## mismatch 指标

| 指标 | 数值 / 路径 | 解释 |
|---|---|---|
| K3 KL |  |  |
| ratio mean / p95 / max |  |  |
| TIS clip fraction |  |  |
| MIS mask fraction |  |  |
| veto token / seq fraction |  |  |
| batch norm factor |  |  |

## 源码对应

| 现象 | 源码位置 | 判断 |
|---|---|---|
|  |  |  |

## 结论

- 本次能证明什么：
- 不能证明什么：
- 下一步要改的配置、代码或实验：
