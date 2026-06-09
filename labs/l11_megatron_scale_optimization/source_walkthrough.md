# L12 源码带读：Bucketed Manual DDP

这份带读按主路径组织。读源码时先建立调用链，再回头看生产分支。

## 0. 源码地图

```text
labs/l11_megatron_scale_optimization/patch/starter/bucketed_ddp.py
labs/l11_megatron_scale_optimization/patch/reference/bucketed_ddp.py
labs/l11_megatron_scale_optimization/patch/tests/conftest.py
labs/l11_megatron_scale_optimization/patch/tests/worker_cases.py
github_repo/Megatron-LM/megatron/core/distributed/distributed_data_parallel.py
github_repo/Megatron-LM/megatron/core/distributed/param_and_grad_buffer.py
github_repo/torchtitan/torchtitan/distributed/parallel_dims.py
labs/l11_megatron_scale_optimization/scripts/run_scale_stub.py
```

## 1. Patch starter

文件：`labs/l11_megatron_scale_optimization/patch/starter/bucketed_ddp.py`

重点看：`BucketedManualDDP.__init__`、`_create_buckets`、`synchronize_grads`。

建议阅读顺序：

- L23-L39: 入口保存 module、process group、world size 和 bucket size。
- L41-L60: `_create_buckets` 的 TODO 描述按字节分桶的合同。
- L62-L74: `synchronize_grads` 的 TODO 描述 flatten、all-reduce、平均和 copy-back。

读完后写一句结论：starter 把需要实现的状态和通信步骤限定在三个函数里。

## 2. Patch reference

文件：`labs/l11_megatron_scale_optimization/patch/reference/bucketed_ddp.py`

重点看：参考实现怎样处理边界。

建议阅读顺序：

- L12-L25: 构造函数保存状态并立即创建 buckets。
- L27-L43: `_create_buckets` 跳过 frozen param，超出 bucket size 时开新桶。
- L45-L56: `synchronize_grads` 对每个 bucket 执行 flatten、all-reduce、平均和 copy-back。

读完后写一句结论：reference 的数学语义是每个 rank 在 optimizer step 前获得同一组 averaged grads。

## 3. Patch tests

文件：`labs/l11_megatron_scale_optimization/patch/tests/conftest.py`

重点看：多进程 gloo harness。

建议阅读顺序：

- L28-L30: `_get_impl` 根据 `IMPL` 选择 starter 或 reference。
- L33-L45: `_worker` 设置分布式环境、初始化 gloo、导入实现并执行 per-rank case。
- L49-L66: `parallel_run` 启动子进程、等待退出并把失败反馈给 pytest。

文件：`labs/l11_megatron_scale_optimization/patch/tests/worker_cases.py`

建议阅读顺序：

- L26-L38: `grads_match_pytorch_ddp` 构造 PyTorch DDP reference。
- L40-L55: 同一 case 构造 `BucketedManualDDP` 并比较每个参数的 grad。
- L57-L69: `works_with_huge_param` 覆盖单个参数大于 bucket size 的边界。
- L109-L121: `world_size_1_is_noop` 覆盖本地路径。

读完后写一句结论：测试先证明数值等价，再覆盖单 rank、frozen param、超大 param 和很多小 param。

## 4. Megatron DDP

文件：`github_repo/Megatron-LM/megatron/core/distributed/distributed_data_parallel.py`

重点看：生产 DDP 如何把本关扩展成 buffer、bucket group 和 overlap。

建议阅读顺序：

- L23-L38: 类说明连续 grad buffer、bucket 和 all-reduce/reduce-scatter overlap。
- L67-L73: 默认 bucket size 和 `overlap_grad_reduce` 的关系。
- L111-L128: 收集 trainable params，并按 dtype/grad dtype/expert parallel 分组。

读完后写一句结论：Megatron 先确定数据并行通信组和 trainable params，再把参数组织进可通信的 buffer。

## 5. Megatron ParamAndGradBuffer

文件：`github_repo/Megatron-LM/megatron/core/distributed/param_and_grad_buffer.py`

重点看：buffer 和 bucket group 的状态字段。

建议阅读顺序：

- L51-L57: `BufferType` 区分 PARAM 和 GRAD buffer。
- L60-L69: `shard_buffer` 按 data parallel world size 切分连续 buffer。
- L72-L84: `_ParamAndGradBucket` 说明 bucket 记录 params、param_data、grad_data、offset 和 scaling factor。
- L159-L171: `_ParamAndGradBucketGroup` 说明 ready 状态和异步通信职责。

读完后写一句结论：生产实现把本关临时 flatten 的 tensor 固定成可复用的连续 buffer，并用 ready hook 驱动通信。

## 6. TorchTitan ParallelDims

文件：`github_repo/torchtitan/torchtitan/distributed/parallel_dims.py`

重点看：扩展配置怎样落到 device mesh。

建议阅读顺序：

- L21-L33: `ParallelDims` 保存 DP、CP、TP、PP、EP 和 world size。
- L51-L71: `_validate` 检查并行度乘积是否匹配 world size。
- L84-L96: `build_mesh` 的 docstring 说明 mesh 维度如何被后续模块使用。

读完后写一句结论：scale optimization 需要先让并行维度自洽，再讨论具体通信优化。

## 7. Drill 脚本

文件：`labs/l11_megatron_scale_optimization/scripts/run_scale_stub.py`

重点看：估算表怎样形成可复查 artifact。

建议阅读顺序：

- L29-L37: `estimate` 把 TP、PP、recompute 和 GPU 数映射到 memory、tokens/sec 和 MFU。
- L64-L79: `main` 写出多个 scenario 和 `metrics.jsonl`。
- L85-L121: `report.md` 记录目标、配置、预测、结果和下一步。

读完后写一句结论：drill 不替代真实 benchmark，但它训练同一份扩展评审的证据格式。

## 读完后的自检问题

1. 你能否从 backward 后的本地 grad 画到 optimizer step 前的 averaged grad？
2. 哪些代码负责 bucket 构造，哪些代码负责跨 rank 平均？
3. Megatron 的 buffer 和 bucket group 相比 patch reference 多了哪些生产状态？
4. 如果真实系统 MFU 低，你会先看哪三个配置或 artifact？
