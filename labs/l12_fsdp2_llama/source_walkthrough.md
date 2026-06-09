# L13 源码带读：FSDP2 Wrap 与 TorchTitan 主路径

这份带读按“patch 最小合同 → smoke 证据 → TorchTitan 生产路径”的顺序走。先读 [lecture.md](lecture.md)，再按下面的锚点看源码。不要从文件顶部一路滚到底。

## 0. 源码地图

```text
labs/l12_fsdp2_llama/patch/starter/fsdp2_wrap.py
  -> labs/l12_fsdp2_llama/patch/reference/fsdp2_wrap.py
  -> labs/l12_fsdp2_llama/patch/tests/test_patch.py
  -> labs/l12_fsdp2_llama/scripts/run_fsdp2_smoke.py

github_repo/torchtitan/torchtitan/models/llama3/parallelize.py
  -> github_repo/torchtitan/torchtitan/distributed/fsdp.py
  -> github_repo/torchtitan/torchtitan/components/checkpoint.py
```

patch 负责缩小机制，TorchTitan 负责展示生产系统如何把 FSDP2 放进 TP、CP、activation checkpoint、compile、checkpoint 和 trainer 里。

## 1. Patch starter：先看接口和 TODO

文件：`labs/l12_fsdp2_llama/patch/starter/fsdp2_wrap.py`

重点看：

- L22-L26: `WrapReport` 定义 patch 和 drill 需要落盘的证据字段。
- L29-L37: `_summarize_policy` 把 mixed precision policy 转成可序列化摘要。
- L40-L58: `wrap_transformer_blocks_fsdp2` 的签名和 TODO，列出 block-first/root-last 的合同。

读完后要得到的结论：学生要实现的是 FSDP2 wrap 编排函数，不涉及完整训练循环。

可以先跳过：真实 FSDP2 的 process group 初始化、DTensor 内部实现和 checkpoint 细节。

## 2. Patch reference：看状态如何变化

文件：`labs/l12_fsdp2_llama/patch/reference/fsdp2_wrap.py`

重点看：

- L31-L36: `_resolve_fully_shard` 支持测试注入 spy，也支持生产导入 PyTorch API。
- L48-L60: 遍历 `named_modules()`，筛出 block，应用 skip，再调用 `fully_shard`。
- L62-L71: root 最后 wrap，并返回 `WrapReport`。

读完后要得到的结论：block wrap 的顺序就是 report 的顺序，root wrap 是最后一个 side effect。

可以先跳过：PyTorch FSDP2 内部如何把 Parameter 变成 DTensor；本关只读 API 边界。

## 3. Patch tests：确认验收边界

文件：`labs/l12_fsdp2_llama/patch/tests/test_patch.py`

重点看：

- L53-L58: `_spy_fully_shard` 只记录模块类名和 kwargs，用来检查调用顺序。
- L61-L69: `test_wrap_marks_each_block` 验证 block 名称和数量。
- L72-L80: `test_wrap_marks_root_last` 验证 root 是最后一次 wrap。
- L83-L93: `test_mp_policy_propagates` 验证 policy 对象原样透传。
- L135-L160: CUDA 可用时才跑 forward/backward smoke。

读完后要得到的结论：CPU 测试主要验证编排语义；GPU 测试才开始触碰真实 FSDP2 runtime。

可以先跳过：pytest marker、临时目录和 skip 细节。

## 4. Smoke 脚本：看 artifact 怎样证明一次 dryrun

文件：`labs/l12_fsdp2_llama/scripts/run_fsdp2_smoke.py`

重点看：

- L38-L46: `_impl` 选择 starter 或 reference 实现。
- L49-L61: `_build_model` 定义教学版 Llama block 的 attention、norm 和 MLP。
- L136-L166: `wrap_only` 模式调用 wrap，并写出 `wrap_report.json` 和 `metrics.jsonl`。
- L188-L199: GPU `train_smoke` 模式创建 `MixedPrecisionPolicy` 并调用真实 `fully_shard`。
- L209-L230: 训练循环记录 loss 和 peak memory。

读完后要得到的结论：dryrun 和 train smoke 产物不同，不能用 dryrun 证明真实显存收益。

可以先跳过：shell profile 组织和真实集群 torchrun 参数。

## 5. TorchTitan parallelize：FSDP2 放在并行组合之后

文件：`github_repo/torchtitan/torchtitan/models/llama3/parallelize.py`

重点看：

- L35-L47: `parallelize_llama` 的入口说明：它把 TP、activation checkpoint、compile 和 data parallelism 应用到 Llama。
- L92-L102: 选择 data parallel mesh，并把 mixed precision dtype、PP 状态、CPU offload 和 reshard policy 传给 `apply_fsdp`。
- L141-L146: `apply_fsdp` 创建 `MixedPrecisionPolicy` 和基础 FSDP config。
- L151-L153: 根据 pipeline 状态解析 `reshard_after_forward`。
- L187-L194: 对每个 transformer block 调用 `fully_shard`，最后 wrap root model。

读完后要得到的结论：TorchTitan 的生产主路径和本关 patch 同构，但它还要处理 mesh、tied weights、pipeline 和 offload。

可以先跳过：日志、pyrefly 注释、具体 dtype map 和 tied weights 分支的全部细节。

## 6. TorchTitan FSDP helper：理解 reshard policy

文件：`github_repo/torchtitan/torchtitan/distributed/fsdp.py`

重点看：

- L28-L39: `get_fsdp_reshard_after_forward_policy` 的输入和返回值。
- L40-L48: `"always"`、`"never"`、`"default"` 如何映射成布尔值。
- L46-L48: pipeline parallel 启用时，默认不在 forward 后 reshard，避免每个 microbatch 重复 all-gather。

读完后要得到的结论：reshard 不是固定开关，它要结合 pipeline schedule 和 microbatch 条件判断。

可以先跳过：`disable_fsdp_gradient_division` 的 token loss 细节；它属于后续训练循环缩放问题。

## 7. TorchTitan checkpoint：看恢复证据的边界

文件：`github_repo/torchtitan/torchtitan/components/checkpoint.py`

重点看：

- L62-L71: `ModelWrapper` 通过 `get_model_state_dict` 收集模型 state dict。
- L76-L85: `load_state_dict` 用 `set_model_state_dict` 写回模型，并刷新缓存 state dict。
- L122-L139: checkpoint manager 的 docstring 说明 pipeline optimizer state 会出现 key 和 layout 问题。

读完后要得到的结论：FSDP2 训练恢复不能只看 model weights；optimizer、lr scheduler、pipeline chunk 和 state dict layout 都会影响恢复。

可以先跳过：异步保存线程、HF safetensors 转换和 purge 逻辑。

## 读完后的自检问题

1. patch reference 的哪一段保证 root 是最后 wrap？
2. CPU dryrun 和 GPU train smoke 分别能证明什么？
3. TorchTitan 在调用 `apply_fsdp` 前已经处理了哪些并行或训练技术？
4. `default` reshard policy 为什么会受 pipeline parallel 影响？
5. checkpoint 恢复失败时，为什么不能只检查模型参数文件？
