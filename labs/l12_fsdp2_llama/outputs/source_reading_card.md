# L13 Source Reading Card：FSDP2

## 主路径

1. `labs/l12_fsdp2_llama/patch/starter/fsdp2_wrap.py`：学生需要补齐的 wrap 合同。
2. `labs/l12_fsdp2_llama/patch/reference/fsdp2_wrap.py`：block-first/root-last 的参考实现。
3. `labs/l12_fsdp2_llama/patch/tests/test_patch.py`：policy、skip、reshard 和 smoke 验收。
4. `labs/l12_fsdp2_llama/scripts/run_fsdp2_smoke.py`：CPU dryrun 和 GPU train smoke artifact。
5. `github_repo/torchtitan/torchtitan/models/llama3/parallelize.py`：Llama FSDP2 生产主路径。
6. `github_repo/torchtitan/torchtitan/distributed/fsdp.py`：reshard policy helper。
7. `github_repo/torchtitan/torchtitan/components/checkpoint.py`：model state dict 和恢复边界。

## 阅读方法

1. 先看 patch 签名，确认输入是 model、block class、policy、reshard 和 skip。
2. 再看 reference 的遍历、skip、fully_shard 调用和 root wrap。
3. 用 tests 确认 patch-test 覆盖的行为边界。
4. 看 smoke 脚本，把 wrap report、loss 和 peak memory 放进 artifact 视角。
5. 最后读 TorchTitan，确认生产路径里多出的 mesh、PP、offload、checkpoint 和 dtype 复杂度。

## 自检

- 我能否解释 FSDP2 与 DDP 的状态保存差异？
- 我能否说清 block-first/root-last 的原因？
- 我能否说明 `reshard_after_forward` 的收益和代价？
- 我能否区分 CPU dryrun、GPU smoke 和真实训练日志的证据强度？
