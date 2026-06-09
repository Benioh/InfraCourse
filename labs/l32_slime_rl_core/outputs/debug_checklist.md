# Debug Checklist：L36 SLiME Weight Sync Coordinator

## 1. 固定现场

- 记录命令、配置、git commit、环境、actor_gpus、rollout_gpus、sync interval 和模型规模。
- 保存 `command.sh`、`config.resolved.yaml`、`metrics.jsonl`、`rl.log`、`report.md` 和 `artifacts/slime_config_validation.json`。
- 标记运行类型：patch-test、CPU smoke、单机 SLiME dry run、H200/NCCL 真实训练。

## 2. 先查同步合同

- train 和 inference 的 key 集是否一致。
- mismatch 是 missing key、shape mismatch 还是 dtype mismatch。
- accepted tensor 是否 clone 后写入。
- `bytes_synced` 是否只统计 accepted tensors。
- inference 多余 key 是否被保留。

## 3. 再查生产链路

| 阶段 | 要看什么 | 常见动作 |
|---|---|---|
| Ray 协调 | `actor_model.update_weights()` 是否被调用，remote ref 是否完成 | 查 train loop、Ray actor 日志 |
| Megatron 准备 | TP/PP/EP gather、HF 转换、bucket size | 查 updater 日志和 bucket 数 |
| Engine lock | rollout 是否还在 generation，lock 等待多久 | 调 sync 时机或等待 generation 结束 |
| 传输 | Ray metadata、NCCL broadcast、有效带宽 | 用 bytes / seconds 估算瓶颈 |
| SGLang 应用 | `weight_version` 是否推进，cache 是否 flush | 查 engine update endpoint 和 version |

## 4. 判断 stale rollout

- rollout sample 的 policy version 是否低于 actor version。
- importance ratio 是否偏离 1，clip fraction 是否升高。
- KL、entropy、reward 和 response length 是否同时异常。
- sync interval 是否过大，或 sync 失败后仍继续生成。

## 5. 结束条件

- 问题可以用一个最小命令复现。
- 同步 stats、版本、metrics 和日志能对应同一个 run。
- 能指出问题来自本地合同、Ray 协调、Megatron 转换、传输、SGLang 更新还是训练配置。
- 结论写入 `rl_rollout_template.md`，并列出下一步只改一个变量的实验。
