# L10 Debug Checklist：长上下文 CP 与 Online Softmax

## 1. 先确认 shape

| 检查项 | 证据 | 判断 |
|---|---|---|
| `q` shape | `(B,H,Sq,D)` | 输出长度应跟 `Sq` 一致 |
| `k/v` shape | `(B,H,Sk,D)` | `k` 和 `v` 的 `Sk`、`D` 必须一致 |
| dtype | fp32 / fp16 / bf16 | 数值容差和显存估算依赖 dtype bytes |
| mask | full / causal / padding / packed | 本关 patch 只覆盖 full attention |
| chunk | `num_chunks`、chunk_len | uneven chunk 不能写死长度 |

## 2. OOM 排查

1. 估算 full score 激活：`B * H * Sq * Sk * dtype_bytes`。
2. 估算 chunk score 激活：`B * H * Sq * chunk_len * dtype_bytes`。
3. 检查是否真的走 flash / ring 路径，避免退回 naive attention。
4. 检查 micro batch、activation checkpoint、sequence parallel 和 CP size。
5. 检查 RoPE/YaRN 配置是否只解决位置外推，未改变 attention score 显存。
6. 真实 GPU 上再用 profiler 看 peak memory、kernel、HBM 和通信时间。

## 3. Online softmax 数值偏差

| 现象 | 优先检查 |
|---|---|
| `num_chunks=1` 对不上 SDPA | scale、einsum 维度、输出 shape |
| `num_chunks=4` 对不上 SDPA | old state rescale、denom/out 同步缩放 |
| uneven chunk 失败 | 手写切片假设每块等长 |
| long K/V 误差变大 | dtype、容差、`new_max` 广播 shape |
| backward 有 nan | `running_denom` 是否为 0、第一次循环的 `-inf` 边界 |

## 4. CP ring 排查

| 检查项 | 需要确认 |
|---|---|
| CP group | `context_parallel_size`、global ranks、rank order |
| ring steps | 每个 rank 是否见过所有 K/V chunk |
| communication | bytes per chunk、总 bytes、stream、overlap |
| causal mask | chunk 的可见范围是否按 query/key 位置处理 |
| packed sequence | sample 边界、position ids、hybrid CP 配置 |
| TE version | `cp_comm_type`、hierarchical CP 和 RoPE CP 参数是否受版本支持 |

## 5. RoPE / YaRN 排查

1. 训练上下文和目标上下文是否记录。
2. RoPE base、interleaved、position ids 是否一致。
3. YaRN scale 和 temperature 只是估算还是来自真实配置。
4. 是否有长上下文 validation perplexity、needle 或任务指标。
5. serving 端 KV cache 和训练端长上下文配置是否一致。

## 6. 结论分级

| 证据强度 | 可以说明什么 |
|---|---|
| patch tests 通过 | online softmax forward 主路径正确 |
| `run_seqlen.py` 输出 | 教学版 CP ring 计划和通信字节估算可复查 |
| `eval_yarn.py` 输出 | RoPE/YaRN 简化参数估算可复查 |
| profiler 数据 | 真实 GPU 显存和 kernel/通信时间可定位 |
| 长上下文评估 | 位置外推和任务质量得到更强验证 |
