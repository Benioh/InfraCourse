# L12 Source Reading Card：Bucketed Manual DDP

## 主路径

1. `labs/l11_megatron_scale_optimization/patch/starter/bucketed_ddp.py`：学生需要补齐的 bucketed grad sync。
2. `labs/l11_megatron_scale_optimization/patch/reference/bucketed_ddp.py`：参考实现中的分桶、flatten、all-reduce 和 copy-back。
3. `labs/l11_megatron_scale_optimization/patch/tests/conftest.py`：多进程 gloo 测试 harness。
4. `labs/l11_megatron_scale_optimization/patch/tests/worker_cases.py`：PyTorch DDP 对照和边界 case。
5. `github_repo/Megatron-LM/megatron/core/distributed/distributed_data_parallel.py`：生产 DDP 的 bucket、buffer、process group 和 overlap。
6. `github_repo/Megatron-LM/megatron/core/distributed/param_and_grad_buffer.py`：连续 param/grad buffer、bucket group 和 reduce-scatter 相关状态。
7. `github_repo/torchtitan/torchtitan/distributed/parallel_dims.py`：并行度校验和 device mesh。
8. `labs/l11_megatron_scale_optimization/scripts/run_scale_stub.py`：scale drill 的 artifact 生成路径。

## 阅读方法

1. 先看 `BucketedManualDDP` 的输入对象和状态字段。
2. 再看 bucket 构造如何跳过 frozen param、处理超大 param。
3. 继续看 `synchronize_grads` 怎样保持 PyTorch DDP 的平均梯度语义。
4. 用测试确认边界，再去 Megatron 源码里找同构状态。
5. 最后看 TorchTitan mesh 和 drill artifact，把局部通信问题放回扩展评审。

## 自检

- 我能否说清 bucket 合并减少的是哪一类开销？
- 我能否指出 reference 和 Megatron DDP 的同构关系？
- 我能否解释 distributed optimizer 为什么会影响 checkpoint 迁移？
- 我能否用 `scale_table.json` 写出一个带比较对象的扩展判断？
