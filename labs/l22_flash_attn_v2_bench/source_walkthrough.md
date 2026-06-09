# 源码带读：L23 PyTorch SDPA 与 FlashAttention benchmark

这份带读按调用链组织。先读教学 patch，确认公式和指标合同；再读 MiniInfra 的 online softmax，理解分块归一化；最后读真实项目里的 SDPA wrapper，观察 layout、cache、mask 和 backend 约束。

## 0. 源码地图

```text
labs/l22_flash_attn_v2_bench/patch/starter/flash_bench.py
labs/l22_flash_attn_v2_bench/patch/reference/flash_bench.py
labs/l22_flash_attn_v2_bench/patch/tests/test_patch.py
labs/l22_flash_attn_v2_bench/scripts/run_bench.py
mini_infra/gpu/triton_softmax.py
github_repo/torchtitan/torchtitan/models/common/attention.py
github_repo/sglang/python/sglang/srt/layers/attention/torch_native_backend.py
github_repo/vllm/vllm/v1/attention/backends/cpu_attn.py
```

## 1. Patch starter：先看学生要补齐的合同

文件：`labs/l22_flash_attn_v2_bench/patch/starter/flash_bench.py`

重点看：

- L11-L17：`eager_attention()` 的 TODO 列出了公式、score 矩阵、causal mask、softmax 和输出。
- L20-L23：`flash_attention()` 只要求调用 SDPA。
- L26-L45：`bench_attention()` 要把正确性、性能、同步、显存和返回字段串起来。

读完要得到的结论：

`starter` 不是让学生调一个库就结束。它要求先写出可解释的 eager baseline，再用 SDPA 对照，最后把比较结果结构化。

可以先跳过：

- import 顺序和异常文本。
- GPU 特定指标的细节，先把 CPU 正确性闭环跑通。

## 2. Patch reference：看最小实现如何分层

文件：`labs/l22_flash_attn_v2_bench/patch/reference/flash_bench.py`

重点看：

- L12-L20：eager baseline 显式创建 `scores`，causal 时构造上三角 mask。
- L23-L24：SDPA 调用只传 `is_causal`，保留和 eager baseline 对齐的语义。
- L37-L45：CUDA 计时前后同步。
- L57-L66：正确性路径使用 fp32 和短序列。
- L68-L84：性能路径使用完整 shape 和目标 dtype，并记录峰值显存。
- L86-L93：返回字段就是下游 artifact 和测试依赖的合同。

读完要得到的结论：

reference 把“正确性”和“性能”分开。这个分层很关键，因为 fp32 小 shape 适合对齐数值，bf16/fp16 大 shape 才适合观察 kernel 差异。

可以先跳过：

- `_alloc()` 的一行封装。
- 具体 speedup 阈值，阈值由测试和 config 决定。

## 3. Patch tests：看验收边界

文件：`labs/l22_flash_attn_v2_bench/patch/tests/test_patch.py`

重点看：

- L17-L18：测试通过 `IMPL` 选择 starter 或 reference。
- L21-L27：输出 shape 必须等于输入 q 的 shape。
- L30-L40：causal mask 用 one-hot value 检查第一个 token 是否看到了未来位置。
- L43-L51：fp32 下 SDPA 和 eager baseline 要在容差内一致。
- L54-L68：benchmark 返回字段必须完整。
- L71-L93：CUDA 存在时再检查 speedup 和峰值显存方向。

读完要得到的结论：

测试是最小合同，不是完整性能报告。它会抓公式、mask、API 和指标字段，但不会证明每种真实模型配置都走 flash backend。

可以先跳过：

- pytest 标记和 skip 细节。

## 4. run_bench：看 drill 怎样写 artifact

文件：`labs/l22_flash_attn_v2_bench/scripts/run_bench.py`

重点看：

- L33-L45：`_impl()` 支持用 `IMPL=starter/reference` 明确选择实现。
- L48-L51：配置里的 dtype 字符串被映射成 torch dtype。
- L54-L68：读取 config、创建 run_dir、记录命令和 resolved config。
- L78-L97：逐个 shape 调用 `bench_attention()` 并写入 `metrics.jsonl`。
- L99-L105：把关键字段写成 `bench.csv`。
- L107-L120：按 config 的 acceptance 生成 `bench_summary.json`。

读完要得到的结论：

drill 的目标是留下可复查证据。一次 benchmark 至少要能恢复实现版本、配置、shape、指标和验收结果。

可以先跳过：

- `runtime_utils` 的具体写文件实现。
- torch 缺失时的 fallback 文件。

## 5. MiniInfra online softmax：看分块归一化的数学骨架

文件：`mini_infra/gpu/triton_softmax.py`

重点看：

- L31-L40：`stable_softmax()` 一次性计算参考结果。
- L43-L55：`online_softmax()` 初始化 running max 和 running sum，准备分块更新。
- L56-L63：每个 block 更新最大值、重缩放旧分母、累加新分母，最后生成输出。
- L66-L68：`max_abs_error()` 给数值对照一个明确指标。

读完要得到的结论：

FlashAttention 能分块处理 softmax，核心原因是 softmax 的最大值和分母可以在线更新。这个文件只证明数学骨架，不证明 GPU 性能。

可以先跳过：

- `simulate_softmax_kernel()` 的 roofline 估算。
- 顶部教学注释里的历史课程编号，它不影响 L23 的主路径。

## 6. Torchtitan SDPA wrapper：看 backend 约束和 layout

文件：`github_repo/torchtitan/torchtitan/models/common/attention.py`

重点看：

- L249-L256：类注释说明 wrapper 用 SDPA，并在调用前后处理 layout。
- L271-L276：默认 backend 列表包含 cudnn、flash 和 math。
- L289-L296：从 `[B, T, H, D]` 转成 `[B, H, T, D]`，在 `sdpa_kernel()` 约束下调用 SDPA，再转回。

读完要得到的结论：

真实训练代码不会只写一行 SDPA。它要处理 layout，并且可能显式排序或限制 backend。

可以先跳过：

- FlexAttention 分支。
- document mask 和 packed sequence 的后续逻辑。

## 7. SGLang torch native backend：看 serving decode 的 cache 语义

文件：`github_repo/sglang/python/sglang/srt/layers/attention/torch_native_backend.py`

重点看：

- L27-L41：extend 路径的输入包含 query、output、KV cache、请求到 token 的映射和长度。
- L89-L99：从 token 映射中取 key/value cache，并对齐 dtype。
- L101-L113：对单个请求调用 SDPA，再把输出写回对应 slice。
- L163-L183：decode 路径同样从 KV cache 取历史 key/value，再调用 SDPA。
- L214-L236：forward extend 负责把 layer 和 forward batch 转成 SDPA 所需参数。

读完要得到的结论：

serving 代码的 attention 不是孤立的 `q/k/v`。它还要从 KV cache、request pool、seq length 和 attention type 中恢复每个请求的有效上下文。

可以先跳过：

- cross attention 和 encoder-only 的完整分支。
- 具体 model runner 初始化。

## 8. vLLM CPU attention：看 mask 和 GQA 参数怎样进入 SDPA

文件：`github_repo/vllm/vllm/v1/attention/backends/cpu_attn.py`

重点看：

- L412-L417：query、key、value 被移动维度，并根据 attention type 决定 causal。
- L419-L437：按序列 slice 调用 SDPA，传入 `attn_mask`、`dropout_p`、`is_causal`、`scale` 和 `enable_gqa`。

读完要得到的结论：

真实后端会把 mask、dropout、scale 和 GQA 显式传给 SDPA。排查 fallback 或数值差异时，这些参数要和 patch 中的最小调用分开看。

可以先跳过：

- ALiBi 和 sliding window mask 的构造细节。

## 读完后的自检问题

1. eager baseline 中 `[B, H, T, T]` 的矩阵在哪一行产生？
2. `flash_attention()` 只调用 SDPA 时，为什么还不能直接断言走了 flash backend？
3. reference 为什么把 fp32 正确性和目标 dtype 性能分开？
4. `run_bench.py` 的哪些 artifact 能支撑一次性能结论？
5. 真实项目里的 SDPA wrapper 比 patch 多了哪些生产参数？
