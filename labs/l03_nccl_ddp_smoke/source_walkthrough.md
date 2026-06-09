# L04 源码带读：ManualDDP 的通信主路径

这份带读只看本讲需要的主路径。目标是把“local grad -> all_reduce sum -> average grad -> smoke artifact”连起来。真实 PyTorch DDP 的 reducer、bucket、hook 和 NCCL 内部实现先跳过，等后续课程再读。

## 1. 源码地图

| 文件 | 读什么 | 结论 |
|---|---|---|
| `labs/l03_nccl_ddp_smoke/patch/starter/manual_ddp.py` | TODO 和接口边界 | 学生需要实现构造函数和 `synchronize_grads()` |
| `labs/l03_nccl_ddp_smoke/patch/reference/manual_ddp.py` | 最小参考实现 | no-op、skip、all-reduce、平均四个步骤构成合同 |
| `labs/l03_nccl_ddp_smoke/patch/tests/conftest.py` | 多进程测试 harness | spawn worker、设置环境变量、初始化 gloo group |
| `labs/l03_nccl_ddp_smoke/patch/tests/worker_cases.py` | 每个测试的数值断言 | 对齐 PyTorch DDP、抓 summed grad、覆盖 skip 边界 |
| `labs/l03_nccl_ddp_smoke/scripts/ddp_hello.py` | 最小 process group smoke | rank+1 tensor 的 all-reduce sum 和 barrier |
| `labs/l03_nccl_ddp_smoke/scripts/run_smoke.py` | smoke launcher 和 artifact | torchrun、fallback、metrics、report |
| `mini_infra/distributed/collectives.py` | MiniInfra collective snapshot | 记录 rank/world_size 和期望 all-reduce 结果 |
| `mini_infra/megatron/training/initialize.py` | 后续并行状态入口 | world size 会继续连接 TP/PP/DP 状态 |

## 2. 阅读顺序

### Step 1：先看 starter 的 TODO 边界

文件：`labs/l03_nccl_ddp_smoke/patch/starter/manual_ddp.py`

重点行：

- L22-L27：类注释说明这个版本不挂 hook，不做 overlap。
- L29-L40：构造函数需要保留原模型、process group 和 world size。
- L42-L59：`synchronize_grads()` 的输入是已经写好的 `param.grad`。

读完要得到的结论：

`ManualDDP` 不是完整训练框架。它只在 backward 后改写 `p.grad`，不管理 forward、loss、optimizer 或 checkpoint。

可以先跳过：

- `torch` import 是否被使用这类细节。
- coalesced all-reduce 作为可选实现，本讲参考路径先用逐参数 all-reduce。

### Step 2：读 reference，把合同压缩成 7 行

文件：`labs/l03_nccl_ddp_smoke/patch/reference/manual_ddp.py`

重点行：

- L11-L15：构造函数保存 `module`、`process_group`、`world_size`。
- L17-L19：单 rank 或 distributed 未初始化时直接返回。
- L20-L24：遍历参数，跳过无效 grad，执行 all-reduce sum，再除以 `world_size`。

读完要得到的结论：

参考实现没有隐藏魔法。DDP 最小语义就是“有效 grad 求和后平均”。这一点要先理解，再去看真实 DDP 的 bucket 和 hook。

可以先跳过：

- 类型注解和导入顺序。
- 真实 PyTorch DDP 源码里的 reducer 细节。

### Step 3：读测试 harness，理解 rank 是怎样来的

文件：`labs/l03_nccl_ddp_smoke/patch/tests/conftest.py`

重点行：

- L33-L40：worker 设置 `MASTER_ADDR`、`MASTER_PORT`、`RANK`、`WORLD_SIZE`，再初始化 gloo process group。
- L41-L44：加载学生实现，并调用对应 worker case。
- L56-L74：`parallel_run` spawn 多进程，等待退出，处理 timeout 和异常。

读完要得到的结论：

patch test 真正启动了多个 Python 进程。`world_size` 不是一个普通循环变量，它决定 process group 的参与者数量，也决定梯度平均的除数。

可以先跳过：

- `_free_port()` 的 socket 实现。
- pytest queue 报错格式。

### Step 4：读 worker cases，定位每个测试抓的 bug

文件：`labs/l03_nccl_ddp_smoke/patch/tests/worker_cases.py`

重点行：

- L14-L45：和 PyTorch DDP 对齐，输入按 rank 轻微变化。
- L48-L81：相同输入下平均梯度应等于单进程 baseline，用来抓忘记除法。
- L84-L97：world size 为 1 时同步函数不改变 grad。
- L100-L117：frozen 参数保持 `grad is None`。
- L119-L131：partial grad 为 `None` 时同步函数不能报错。

读完要得到的结论：

5 个测试不是重复检查。它们分别覆盖：与真实 DDP 对齐、平均语义、no-op、frozen 参数和 partial grad。

可以先跳过：

- MLP 结构的具体 hidden size。
- `torch.manual_seed()` 的具体数字。

### Step 5：读 `ddp_hello.py`，看最小 collective artifact

文件：`labs/l03_nccl_ddp_smoke/scripts/ddp_hello.py`

重点行：

- L24-L35：导入 torch/distributed，读取 `WORLD_SIZE`、`RANK`、`LOCAL_RANK`，选择 backend。
- L42-L48：初始化 process group，对 `rank + 1` tensor 执行 all-reduce sum，再 barrier。
- L52-L65：写入 payload 字段并销毁 process group。
- L68-L79：rank0 写出 JSON，同时每个进程打印 payload。

读完要得到的结论：

这个脚本验证的是 process group、all-reduce 和 barrier 的最小通信链路。它不验证 `ManualDDP` 梯度，也不代表 NCCL 性能。

可以先跳过：

- argparse 细节。
- rank0 写文件的目录创建逻辑。

### Step 6：读 `run_smoke.py`，判断 artifact 证据强度

文件：`labs/l03_nccl_ddp_smoke/scripts/run_smoke.py`

重点行：

- L28-L39：fallback payload 明确标出 `fallback_used=True`。
- L42-L59：用 `torch.distributed.run` 启动两个进程，失败或缺 artifact 时走 fallback。
- L68-L81：准备 run dir，写配置，并保证 `ddp_hello.json` 存在。
- L83-L95：写 `metrics.jsonl` 和 `train.log`。
- L96-L124：报告写明 fallback 边界和迁移判断。

读完要得到的结论：

smoke 的输出要分级解释。`fallback_used=False` 才能作为真实 process group 跑通的证据；`fallback_used=True` 只能证明 validation artifact 生成链路可用。

可以先跳过：

- `scripts.runtime_utils` 内部实现。
- report 模板的 Markdown 排版。

### Step 7：读 MiniInfra 的 collective snapshot

文件：`mini_infra/distributed/collectives.py`

重点行：

- L9-L13：从环境变量读取 rank、local rank、world size，并计算期望求和结果。
- L14-L23：把通信边界和 `CUDA_VISIBLE_DEVICES` 写进 payload。

读完要得到的结论：

MiniInfra 用更轻的方式保存分布式上下文证据。它不执行真实 all-reduce，但能帮助学生在没有 GPU 或 torchrun 的环境下检查环境变量语义。

可以先跳过：

- CLI `--json` 的输出分支。

### Step 8：看 Megatron 初始化的前置影子

文件：`mini_infra/megatron/training/initialize.py`

重点行：

- L8-L15：`DistributedState` 记录 world size、TP、PP、DP 和 sequence parallel。
- L17-L28：初始化函数从 TP/PP 参数推导并行状态。

读完要得到的结论：

本讲的 rank/world size 后续会进入更复杂的并行状态。L04 先把 data parallel 的平均梯度语义打稳，后面才能讨论 TP/PP/Megatron 的 group 划分。

可以先跳过：

- `MiniMegatronArgs` 的完整字段定义。
- sequence parallel 的通信细节。

## 3. 读完后的自检问题

1. `ManualDDP.__init__` 为什么不能复制参数？
2. `synchronize_grads()` 为什么要同时判断 `world_size == 1` 和 `dist.is_initialized()`？
3. `all_reduce(SUM)` 的输出是什么？平均发生在哪一行？
4. patch tests 中哪个测试最容易抓住忘记除以 `world_size` 的错误？
5. `fallback_used=True` 的 smoke artifact 能证明什么，不能证明什么？
6. 真实 PyTorch DDP 的 bucket 和 hook 主要解决语义问题还是性能调度问题？
