# L13 Debug Checklist：FSDP2 Wrap、显存与 Checkpoint

## 1. 先固定现场

- 记录命令、配置文件、git commit、Python/PyTorch 版本、硬件、world size、device mesh、模型规模、seq length、micro batch 和随机种子。
- 保存 stdout/stderr、`config.resolved.yaml`、`wrap_report.json`、`metrics.jsonl`、checkpoint 路径和 report。
- 明确这是 CPU patch-test、CPU dryrun、GPU train smoke，还是真实 TorchTitan 训练。

## 2. 判断问题在哪一层

| 层 | 要看什么 | 可能结论 |
|---|---|---|
| 输入 | model depth、block class、skip 规则、profile、dtype | wrap 对象或配置已经偏离预期 |
| Wrap 状态 | `wrapped_blocks`、root wrap 顺序、`mp_policy_summary`、reshard | FSDP2 单元没有按预期创建 |
| Runtime | all-gather、reshard、loss、grad、peak memory、OOM | 通信/显存/数值路径出现问题 |
| Checkpoint | model state dict、optimizer state、world size、PP chunks | 恢复 layout 或分片状态不匹配 |

## 3. 沿源码主路径复查

- `labs/l12_fsdp2_llama/patch/starter/fsdp2_wrap.py`：学生需要补齐的 wrap 合同。
- `labs/l12_fsdp2_llama/patch/reference/fsdp2_wrap.py`：参考实现的 block-first/root-last 顺序。
- `labs/l12_fsdp2_llama/patch/tests/test_patch.py`：测试如何检查 policy、skip、reshard 和 root 顺序。
- `labs/l12_fsdp2_llama/scripts/run_fsdp2_smoke.py`：dryrun 和 train smoke 的 artifact 生成路径。
- `github_repo/torchtitan/torchtitan/models/llama3/parallelize.py`：TorchTitan Llama FSDP2 主路径。
- `github_repo/torchtitan/torchtitan/distributed/fsdp.py`：reshard policy 解析。
- `github_repo/torchtitan/torchtitan/components/checkpoint.py`：state dict 和训练恢复边界。

## 4. 常见错误判断

- 只看 patch 通过，就判断真实 FSDP2 显存收益成立。
- 忘记 root 最后 wrap，导致剩余参数没有进入正确 FSDP 单元。
- 重新创建 mixed precision policy，导致不同模块 dtype 证据不一致。
- 把 `reshard_after_forward=True` 当成纯收益，忽略 backward all-gather 成本。
- 只保存 model weights，没有检查 optimizer、lr scheduler、dataloader 和 train state。

## 5. 结束条件

- 问题能用一个最小 profile 复现。
- wrap report、配置、metrics 和 checkpoint 证据已经落盘。
- 源码主路径中能指出状态在哪里创建、通信和释放。
- 结论写进 `fsdp2_wrap_template.md`，并包含下一步动作。
