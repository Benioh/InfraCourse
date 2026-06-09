# L06 源码带读：Tensor Parallel Linear

这份带读按“starter TODO -> reference 语义 -> tests 证据 -> MiniInfra/Megatron 对照 -> toy/smoke 边界”的顺序组织。目标是看清 Column/Row Linear 的切分和通信位置，不先追 Megatron 的全部工程分支。

## 1. 源码地图

| 文件 | 读什么 | 结论 |
|---|---|---|
| `patch/starter/tp_linear.py` | 三个 primitive 和两个 layer 的 TODO | 学生要补齐 forward/backward 通信和参数分片 |
| `patch/reference/tp_linear.py` | 最小正确实现 | copy/reduce/gather 的双向规则和 Column/Row forward |
| `patch/tests/conftest.py` | 2-rank gloo harness | patch-test 真正启动多进程 |
| `patch/tests/worker_cases.py` | 数值断言 | 用 canonical 单卡权重比较 TP 输出和梯度 |
| `mini_infra/megatron/core/tensor_parallel/layers.py` | 同构概念骨架 | Column 切输出，Row 切输入 |
| Megatron `mappings.py` | 真实 autograd Function | copy/reduce/gather 的规则与本关一致 |
| Megatron `layers.py` | 真实 Column/Row layer | 生产实现增加 sequence parallel、async、state dict 等分支 |
| `scripts/*.py` | toy 和 smoke | 帮助理解，不替代 patch-test |

## 2. 阅读顺序

### Step 1：先读 starter 的三个 primitive

文件：`labs/l05_distributed_primitives/patch/starter/tp_linear.py`

重点行：

- L30-L42：`_CopyToParallelRegion`，forward identity，backward all-reduce。
- L45-L57：`_ReduceFromParallelRegion`，forward all-reduce，backward identity。
- L60-L80：`_GatherAlongLastDim`，forward all-gather + cat，backward split。

读完要得到的结论：

TP Linear 的正确性不只在 module forward。通信 primitive 的 backward 决定梯度是否回到正确分片。

### Step 2：读 starter 的两个 layer TODO

文件：`labs/l05_distributed_primitives/patch/starter/tp_linear.py`

重点行：

- L88-L142：Column 的参数分片、本地 linear、可选 gather。
- L145-L213：Row 的输入分片、partial output reduce、bias 陷阱。

读完要得到的结论：

Column 切 `out_features`，Row 切 `in_features`。Row 的 bias 是完整 replicated 向量，并且只能在 all-reduce 后加一次。

### Step 3：读 reference，确认最小正确路径

文件：`labs/l05_distributed_primitives/patch/reference/tp_linear.py`

重点行：

- L20-L31：copy primitive 的 backward all-reduce。
- L34-L45：reduce primitive 的 forward all-reduce 和 backward identity。
- L48-L67：gather primitive 的 forward gather 和 backward split。
- L96-L113：Column 参数和 forward。
- L143-L171：Row 参数、broadcast bias、forward reduce 后加 bias。

读完要得到的结论：

reference 实现很短，复杂度主要来自“哪一步通信”和“反向路径是否匹配”。真实框架的额外分支是工程优化，不改变这条数学主线。

### Step 4：读测试，理解证据从哪里来

文件：`labs/l05_distributed_primitives/patch/tests/conftest.py`

重点行：

- L54-L65：worker 设置 rendezvous 环境变量、初始化 gloo group、调用 worker case。
- L77-L92：`parallel_run` spawn 多个 worker。
- L93-L108：处理 timeout、非零退出和 rank0 状态。

文件：`labs/l05_distributed_primitives/patch/tests/worker_cases.py`

重点行：

- L20-L49：Column forward 对齐单卡。
- L52-L79：Row forward 对齐单卡。
- L95-L125：Column backward 对齐单卡梯度。
- L128-L161：Row backward 对齐单卡梯度。
- L235-L270：Row bias 只加一次。

读完要得到的结论：

测试会把 canonical 单卡权重切片复制到 TP layer。失败时要看是切片方向错、collective 位置错、梯度 primitive 错，还是 Row bias 重复加。

### Step 5：读 MiniInfra 和 Megatron 对照

文件：`mini_infra/megatron/core/tensor_parallel/layers.py`

重点行：

- L13-L49：MiniInfra Column 保存输出维分片范围和通信类型。
- L52-L88：MiniInfra Row 保存输入维分片范围和 reduce 通信类型。

文件：`github_repo/Megatron-LM/megatron/core/tensor_parallel/mappings.py`

重点行：

- L197-L214：copy 的 forward identity，backward reduce。
- L217-L233：reduce 的 forward reduce，backward identity。
- L256-L273：gather 的 forward gather，backward split。

读完要得到的结论：

本关 primitive 与 Megatron 的 mapping 语义一致。生产版本增加 buffer、sequence parallel、reduce-scatter 等路径，但基本双向规则相同。

### Step 6：看真实 Megatron layer 的生产复杂度

文件：`github_repo/Megatron-LM/megatron/core/tensor_parallel/layers.py`

重点行：

- L1020-L1031：Column forward 决定是否 copy 到 tensor model parallel region。
- L1069-L1090：Column 根据配置 gather output 或保留分片。
- L1200-L1209：Row 计算 input partition。
- L1325-L1352：Row 本地 linear 后 reduce，再处理 bias。

读完要得到的结论：

Megatron 的生产代码多了 sequence parallel、expert 通信、CPU offload、skip bias add、sharded state dict 等分支。L06 只要求先看懂主路径。

### Step 7：读 toy/smoke 边界

文件：

- `labs/l05_distributed_primitives/scripts/collectives_demo.py`
- `labs/l05_distributed_primitives/scripts/toy_column_parallel_linear.py`
- `labs/l05_distributed_primitives/scripts/run_lab.py`

重点行：

- `collectives_demo.py` L42-L60：all-reduce、all-gather、broadcast 的最小演示。
- `toy_column_parallel_linear.py` L14-L27：Column forward 的单进程切分校验。
- `run_lab.py` L52-L79：写入 collective、DDP、TP toy 和 pipeline toy 指标。

读完要得到的结论：

toy 和 smoke 帮你定位系统概念，patch-test 才验收 starter 的完整 forward/backward 语义。

## 3. 读完后的自检问题

1. `nn.Linear` 的 weight shape 是什么？
2. Column 和 Row 分别切哪个维度？
3. `_CopyToParallelRegion` 的 backward 为什么要 all-reduce？
4. `_GatherAlongLastDim` 的 backward 为什么是 split？
5. Row bias 为什么不能传进本地 `F.linear`？
6. CPU/gloo patch-test 通过后，还缺什么证据才能讨论 GPU/NCCL 性能？
