# L13 FSDP2 Wrap 复盘模板

## Run 信息

- 日期：
- 机器 / GPU：
- 命令：
- 配置文件：
- git commit：
- PyTorch 版本：
- world size / mesh：
- 模型规模：
- seq length / micro batch：
- `reshard_after_forward`：
- mixed precision policy：

## 预期

- 这次运行要验证的机制：
- 比较对象：
- 成功标准：

## Wrap 证据

| 指标或 artifact | 数值 / 路径 | 解释 |
|---|---|---|
| `wrap_report.json` |  |  |
| wrapped block count |  |  |
| root wrapped last |  |  |
| skipped blocks |  |  |
| mp policy summary |  |  |

## 训练或 Smoke 指标

| 指标 | 数值 | 条件 |
|---|---|---|
| loss head / tail |  |  |
| loss drop |  |  |
| peak memory GB |  |  |
| step time |  |  |
| checkpoint path |  |  |

## 源码对应

| 现象 | 源码位置 | 判断 |
|---|---|---|
|  |  |  |

## 结论

- 本次能证明什么：
- 不能证明什么：
- 下一步要改的配置、代码或实验：
