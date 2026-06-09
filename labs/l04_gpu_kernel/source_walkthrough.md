# L05 源码带读：Triton 行 Softmax

这份带读按“数学骨架 -> Triton reference -> 测试合同 -> roofline 估算 -> smoke 边界”的顺序读。不要先读真实框架里复杂的 fused attention kernel；本讲先把单行 softmax 的最小路径闭合。

## 1. 源码地图

| 文件 | 读什么 | 结论 |
|---|---|---|
| `labs/l04_gpu_kernel/patch/starter/triton_softmax.py` | TODO、kernel 参数、wrapper 边界 | 学生要补齐 kernel 和 Python wrapper |
| `labs/l04_gpu_kernel/patch/reference/triton_softmax.py` | 行 kernel 和 wrapper 的最小实现 | 每个 program 处理一行，mask 保护 padding |
| `labs/l04_gpu_kernel/patch/tests/test_patch.py` | pytest 行为合同 | 输出对齐 PyTorch softmax，row sum 接近 1 |
| `mini_infra/gpu/triton_softmax.py` | stable softmax 和 online softmax | CPU 侧解释减 max 与 running sum 重缩放 |
| `mini_infra/gpu/memory_model.py` | bytes、occupancy、roofline 估算 | softmax 性能先看 HBM 读写预算 |
| `labs/l04_gpu_kernel/scripts/bench_softmax.py` | bench CLI | 调用 MiniInfra 模拟，不验收 starter patch |
| `labs/l04_gpu_kernel/scripts/run_smoke.py` | smoke CLI | 走通用 half-mission runner，默认 validation-only |

## 2. 阅读顺序

### Step 1：先看 starter，明确要补哪两层

文件：`labs/l04_gpu_kernel/patch/starter/triton_softmax.py`

重点行：

- L27-L35：Triton kernel 的签名，包含 input/output 指针、stride、`n_cols` 和 `BLOCK_SIZE`。
- L37-L49：kernel TODO，按 row id、offset、mask、load、max、exp、sum、store 展开。
- L52-L57：wrapper 先检查 Triton、CUDA tensor 和 2D 输入。
- L58-L70：wrapper TODO，计算 shape、`BLOCK_SIZE`、output，并按 `(n_rows,)` 启动。

读完要得到的结论：

本关实现分成两层。kernel 负责行内 softmax；wrapper 负责输入检查、block 选择、output 分配和 launch。不能在 wrapper 中用 PyTorch softmax 兜底。

可以先跳过：

- ImportError 分支的文字。
- `HAS_TRITON` 的测试环境细节。

### Step 2：读 reference kernel，看 mask 和稳定计算

文件：`labs/l04_gpu_kernel/patch/reference/triton_softmax.py`

重点行：

- L26-L33：取 `row_idx`、列 offsets、mask，并用 `other=-inf` masked load。
- L34-L37：求 row max、numerator、denominator 和 output。
- L38-L42：masked store，避免 padding 列写越界。
- L45-L60：wrapper 检查 CUDA/2D，选择 `BLOCK_SIZE`，启动 grid。

读完要得到的结论：

reference 的核心没有多余抽象。每个 program 只处理一行；padding 列在 max 中被 `-inf` 忽略；输出只写回真实列。

可以先跳过：

- Triton JIT 编译缓存细节。
- `num_warps` 和 `num_stages` 的生产调优。

### Step 3：读 tests，把验收点和错误类型对应起来

文件：`labs/l04_gpu_kernel/patch/tests/test_patch.py`

重点行：

- L18-L20：通过 `IMPL` 环境变量加载 starter 或 reference。
- L26-L28：没有 CUDA 时 skip。
- L31-L50：fp32/fp16 分别与 `F.softmax` 比较 max diff。
- L53-L59：每行输出和接近 1。
- L62-L77：短行和长行两种 shape。

读完要得到的结论：

测试只验证 forward 行 softmax。skip 是环境状态，不是通过。若失败，要从 max diff、row sum、shape 和 dtype 判断是数值、mask 还是 launch 问题。

可以先跳过：

- pytest marker 的配置细节。
- `sys.path` 注入 patch 目录的机制。

### Step 4：读 MiniInfra stable softmax 与 online softmax

文件：`mini_infra/gpu/triton_softmax.py`

重点行：

- L31-L40：`stable_softmax` 一次性减 max 后计算概率。
- L43-L50：`online_softmax` docstring 解释 running sum 为什么要重缩放。
- L52-L63：分块更新 running max 和 running sum，最后输出概率。
- L71-L77：`mask_needed` 解释 tail block 为什么需要 mask。
- L81-L99：`simulate_softmax_kernel` 组合稳定性误差和 roofline 估算。

读完要得到的结论：

MiniInfra 的 CPU 代码不跑 Triton kernel。它用于解释数学：减 max 防溢出；online softmax 在 max 变化时必须重缩放旧 sum；非整除 block 需要 mask。

可以先跳过：

- 注释中提到的真实框架路径。
- `typing` 和 JSON 输出格式。

### Step 5：读 memory model，理解性能预算来自哪里

文件：`mini_infra/gpu/memory_model.py`

重点行：

- L28-L48：`GPUProfile` 记录 HBM、L2、SMEM、register 和 warp 上限。
- L51-L76：课程内置的 GPU profile。
- L83-L92：`softmax_bytes` 估算 fused 与 eager 的读写字节。
- L95-L100：`occupancy_estimate` 用 block size 和 warps 粗估 occupancy。
- L103-L132：`roofline_softmax` 输出 bytes、bandwidth、estimated_ms 和 speedup 估算。

读完要得到的结论：

这个模型不是 benchmark。它用规格和启发式给出量级预算，帮助你把 softmax 的性能讨论落到 bytes、bandwidth、occupancy 和 shape 上。

可以先跳过：

- 具体启发式系数的来源。
- 不同 GPU 的精确硬件规格争议。

### Step 6：读 bench 与 smoke 的边界

文件：`labs/l04_gpu_kernel/scripts/bench_softmax.py`

重点行：

- L12-L16：CLI 参数包括 `seq`、`block` 和 `batch`。
- L17-L23：调用 `simulate_softmax_kernel` 并打印 JSON。

文件：`labs/l04_gpu_kernel/scripts/run_smoke.py`

重点行：

- L6-L10：把项目根目录加入 `sys.path`，导入通用 runner。
- L13-L20：解析 `run-id` 和 `mode`，调用 `run_half_mission("l04_gpu_kernel", ...)`。

读完要得到的结论：

bench 是 CPU 侧教学模拟，smoke 是课程通用 artifact 链路。二者都不能替代 `make patch-test M=l04_gpu_kernel` 在 GPU/Triton 环境里通过。

可以先跳过：

- `half_mission_runner` 内部模板。
- argparse 的默认值处理。

## 3. 读完后的自检问题

1. `BLOCK_SIZE` 和 `n_cols` 有什么区别？
2. mask 外为什么填 `-inf`，而不是 0？
3. 为什么 load 和 store 都需要 mask？
4. reference 中哪几行完成稳定 softmax 的 `max -> exp -> sum -> divide`？
5. patch tests 中 skip、pass、failure 分别代表什么？
6. `bench_softmax.py` 的输出为什么不能当作 GPU kernel 验收？
