# L27 Serving Metrics 复盘模板

## 1. Run 信息

- 日期：
- run id：
- git commit：
- 机器 / GPU：
- 模型：
- workload：
- 并发：
- prompt 长度分布：
- 输出长度分布：
- 是否 streaming：

## 2. 命令与配置

- 服务启动命令：
- drill / benchmark 命令：
- 配置文件：
- 改动变量：
- baseline run：
- current run：

## 3. 预期

- 这次要验证的机制：
- 比较对象：
- 成功标准：
- 可能代价：

## 4. 关键指标

| 阶段 | 指标 | baseline | current | 判断 |
|---|---|---:|---:|---|
| queue | prefill queue / decode queue |  |  |  |
| prefill | TTFT / input throughput / cache hit-rate |  |  |  |
| decode | ITL / TPOT / gen throughput / running reqs |  |  |  |
| KV cache | token usage / available tokens / used tokens |  |  |  |
| KV transfer | latency / speed / size / failed counters |  |  |  |
| exporter | TYPE 行 / label 顺序 / series 数量 |  |  |  |

## 5. Artifact

| 文件 | 路径 | 说明 |
|---|---|---|
| command |  |  |
| resolved config |  |  |
| metrics |  |  |
| log |  |  |
| report |  |  |
| comparison artifact |  |  |

## 6. 源码对应

| 现象 | 源码位置 | 判断 |
|---|---|---|
|  |  |  |

## 7. 结论

- 本次能证明什么：
- 不能证明什么：
- 主要瓶颈：
- 下一步只改哪一个变量：
- 风险：
