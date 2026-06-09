# L03 Memory Snapshot 泄露复盘模板

## Run 信息

- 日期：
- 机器 / GPU：
- 命令：
- 配置文件：
- git commit：
- Python / PyTorch / CUDA：
- mission 或真实任务：
- rank / worker / pid / device：

## 问题现象

- OOM step：
- 显存开始上涨的 step 区间：
- 上涨形态：单调 / 阶梯 / 偶发峰值
- `allocated` / `reserved` / `nvidia-smi` 观测：
- 是否能用最小命令复现：

## Snapshot 设置

- memory history 开启时间：
- dump 时间：
- snapshot 路径：
- 录制窗口是否覆盖泄露：
- 录制开销或对吞吐的影响：

## Top Stack 聚合

| 排名 | total bytes | stack 摘要 | 判断 | 下一步 |
|---|---:|---|---|---|
| 1 |  |  |  |  |
| 2 |  |  |  |  |
| 3 |  |  |  |  |

## 源码定位

| stack frame | 文件 / 行号 | 生命周期预期 | 发现的问题 |
|---|---|---|---|
|  |  |  |  |

## 修复与验证

- 修复动作：
- 可能代价：
- 复测命令：
- 修复后显存曲线：
- 修复后 top stack：
- 结论：

## 不能证明的事

- snapshot 没覆盖的进程：
- snapshot 没覆盖的时间窗口：
- 仍需 profiler 或 allocator 细节确认的问题：
