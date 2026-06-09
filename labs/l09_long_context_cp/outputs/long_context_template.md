# L10 长上下文复盘模板

## Run 信息

- 日期：
- 机器 / GPU：
- 命令：
- git commit：
- PyTorch / Megatron / Transformer Engine 版本：
- 是否真实 GPU profile：

## Shape 与配置

| 字段 | 值 | 判断 |
|---|---:|---|
| B |  |  |
| H |  |  |
| Sq |  |  |
| Sk |  |  |
| D |  |  |
| dtype bytes |  |  |
| num_chunks |  |  |
| context_parallel_size |  |  |
| cp_comm_type |  |  |

## 显存与通信估算

| 指标 | 数值 | 判断 |
|---|---:|---|
| full score bytes |  |  |
| chunk score bytes |  |  |
| estimated peak_mem_gb_cp |  |  |
| bytes per K/V chunk |  |  |
| attn_comm_bytes |  |  |
| ring steps per rank |  |  |

## 数值验证

| 测试 | 结果 | 最大误差 |
|---|---|---:|
| num_chunks=1 vs SDPA |  |  |
| num_chunks=4 vs SDPA |  |  |
| uneven chunk |  |  |
| long K/V |  |  |
| backward finite grads |  |  |

## RoPE / YaRN

| 字段 | 值 | 判断 |
|---|---:|---|
| train context |  |  |
| target context |  |  |
| RoPE base |  |  |
| YaRN scale |  |  |
| attention temperature |  |  |
| validation metric |  |  |

## 源码对应

| 现象 | 源码位置 | 判断 |
|---|---|---|
| online softmax rescale |  |  |
| K/V chunk 切分 |  |  |
| CP group 创建 |  |  |
| TE cp kwargs |  |  |
| RoPE / YaRN 配置 |  |  |

## 结论

- 本次能证明什么：
- 本次不能证明什么：
- 若要进入真实 CP 训练，还缺哪些验证：
- 下一步动作：
