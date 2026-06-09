# L15 源码带读：Pipeline Parallel 1F1B

这份带读只走 non-interleaved 1F1B 主路径。第一次读时先跳过 interleaved schedule、multi-module pipeline、cuda graph、fine-grained activation offload 和具体模型 forward 细节。目标是看清：schedule 选择在哪里发生，warmup 数怎么算，forward/backward 在稳定阶段怎样交替，cooldown 怎样排空。

## 1. 源码地图

```text
labs/l14_pipeline_1f1b/patch/starter/pp_schedule.py
  -> 学生要补齐的 timeline 生成器

labs/l14_pipeline_1f1b/patch/reference/pp_schedule.py
  -> warmup / steady / cooldown 的最小参考实现

labs/l14_pipeline_1f1b/patch/tests/test_patch.py
  -> L15 patch 的行为合同

labs/l14_pipeline_1f1b/scripts/run_pp_smoke.py
  -> mock 时间模型和 artifact 落盘

mini_infra/megatron/core/pipeline_parallel/schedules.py
  -> 教学版 forward/backward 事件流

github_repo/Megatron-LM/megatron/core/pipeline_parallel/schedules.py
  -> Megatron non-interleaved 1F1B 真实主路径
```

## 2. 阅读步骤一：先看 patch starter 的合同

文件：`labs/l14_pipeline_1f1b/patch/starter/pp_schedule.py`

先读 `make_1f1b_schedule` 的签名和 TODO。它定义了本关的最小输入输出：输入是 stage 数和 microbatch 数，输出是每个 stage 的局部 timeline。

重点：

- L6-L7：函数返回 `list[list[tuple[str, int]]]`，外层按 stage 分组。
- L8-L13：TODO 把 warmup、steady、cooldown 三段写成明确步骤。
- L17-L20：`bubble_count` 只负责标准 non-interleaved 1F1B 的首尾空泡公式。

读完要能回答：patch 为什么不需要真实 tensor，也能验证 1F1B 的调度顺序？

## 3. 阅读步骤二：对照 reference 的三段式

文件：`labs/l14_pipeline_1f1b/patch/reference/pp_schedule.py`

这份 reference 是 L15 最小机制的答案。它没有 Megatron 的通信和梯度收尾，只保留调度顺序。

重点：

- L6-L12：输入校验，特别是 `num_microbatches >= num_stages`。
- L13-L18：对每个 stage 建立 timeline，并初始化 forward/backward 游标。
- L19-L22：warmup 只追加 forward。
- L22-L27：steady 每轮追加一个 forward 和一个 backward。
- L28-L31：cooldown 只追加 backward，并把 timeline 放入 schedule。
- L35-L36：bubble 公式是 `2 * (num_stages - 1)`。

读完要能画出 stage 0 和最后一个 stage 的 timeline 差异。

## 4. 阅读步骤三：用测试反推不变量

文件：`labs/l14_pipeline_1f1b/patch/tests/test_patch.py`

测试用来钉住最小合同，不覆盖 Megatron 全部行为。

重点：

- L20-L27：每个 stage 的开头必须有正确数量的 warmup forward。
- L29-L38：steady 区间必须按 F/B 交替。
- L50-L57：每个 microbatch 在每个 stage 上恰好出现一次 forward 和一次 backward。
- L70-L77：同一 microbatch 的 backward 位置必须晚于 forward。
- L80-L90：bubble 公式和非法 microbatch 输入都要显式验证。

读完要能判断：如果某个实现连续输出两个 forward，哪条测试会先失败？

## 5. 阅读步骤四：看 drill 怎样把 schedule 变成证据

文件：`labs/l14_pipeline_1f1b/scripts/run_pp_smoke.py`

这一步看系统层证据。脚本从配置读 stage 数、microbatch 数和 mock op 时间，运行 schedule generator，再把结果写到 run 目录。

重点：

- L53-L61：加载实现、配置和 run 目录，并写入 resolved config。
- L70-L76：读取 `D`、`N`、forward/backward time，并生成 schedule 和 bubble。
- L78-L85：用最长 stage 时间和理想时间估算 bubble ratio。
- L87-L97：把 acceptance 条件和实际结果对齐。
- L99-L111：向 `metrics.jsonl` 写入可检索的 PP 指标。
- L113-L124：向 `pp_summary.json` 写入 schedule 头部和尾部样本。

读完要能说明：drill 的 `accept=true` 能证明什么，不能证明什么。

## 6. 阅读步骤五：用 MiniInfra 建立事件流直觉

文件：`mini_infra/megatron/core/pipeline_parallel/schedules.py`

MiniInfra 不是 Megatron 的完整复制，它用于展示 PP size 如何改变 forward-backward 函数选择，以及 forward/backward 事件如何按 stage 和 microbatch 排列。

重点：

- L14-L19：PP size 大于 1 时选择 pipeline 版本，否则走 no pipeline。
- L31-L34：non-interleaved 教学函数返回 `PipelineEvent` 列表。
- L35-L44：forward 事件按 microbatch 和 stage 向后推进。
- L45-L57：backward 事件按 microbatch 和 stage 反向推进。
- L60-L61：bubble ratio 用 stage 数和 microbatch 数估算。

读完要能说明：MiniInfra 的事件流和 patch timeline 分别省略了哪些真实系统细节。

## 7. 阅读步骤六：进入 Megatron schedule selector

文件：`github_repo/Megatron-LM/megatron/core/pipeline_parallel/schedules.py`

先读函数选择，不要直接跳进两千多行的 schedule 实现。

重点：

- L48-L54：`get_forward_backward_func` 的目标是根据并行配置返回训练 step 函数。
- L147-L154：PP size 大于 1 时进入 pipeline；有 virtual PP size 时选择 interleaved，否则选择 non-interleaved。
- L157-L168：`deallocate_output_tensor` 说明真实 schedule 会在发送 activation 后释放 `.data`，降低激活驻留。

读完要能回答：L15 patch 对应的是 Megatron 的哪个分支？

## 8. 阅读步骤七：Megatron non-interleaved 1F1B 主路径

文件：`github_repo/Megatron-LM/megatron/core/pipeline_parallel/schedules.py`

进入 `forward_backward_pipelining_without_interleaving` 后，只抓五段：初始化、warmup、steady、cooldown、finalize。

重点：

- L2035-L2055：函数入口说明这是带 pipeline stage 通信的 non-interleaved 1F1B。
- L2166-L2169：计算 warmup microbatch 数和 remaining 数。
- L2223-L2233：warmup 循环准备 activation checkpoint 策略。
- L2234-L2245：warmup 接收 forward tensor，并调用 `forward_step`。
- L2246-L2252：warmup 发送 forward 输出。
- L2260-L2269：进入 steady 前先接收第一个 forward tensor，然后循环 remaining microbatch。
- L2280-L2296：steady 的 forward step 使用 `i + num_warmup_microbatches` 作为当前 microbatch。
- L2301-L2308：训练模式下发送 forward 并接收 backward gradient。
- L2310-L2318：保存新 forward 的 tensor，再弹出最早的一对 tensor 做 backward。
- L2320-L2328：必要时打开 grad sync，并执行 backward。
- L2330-L2338：发送 backward gradient，非最后一轮同时接收下一个 forward。
- L2340-L2351：cooldown 最后一轮可能重新打开 grad sync。
- L2353-L2364：cooldown 接收剩余 backward gradient 并继续向前发送。
- L2380-L2388：最后做 data parallel、sequence parallel 和 embedding 相关的 gradient finalize。

读完要能把 patch 的 `warmup / steady / cooldown` 三段，分别指到 Megatron 的具体循环。

## 9. 可以先跳过的分支

- interleaved schedule：等 non-interleaved 读通后再看 virtual pipeline stage。
- multi-module pipeline：面向 VLM 等多模块 pipeline，本讲只保留单模型主线。
- cuda graph 和 activation offload：它们影响性能和内存管理，不改变 1F1B 的基本依赖。
- 具体模型 forward_step：本讲关心 schedule 怎样驱动它，不展开模型层内部。

## 10. 自检问题

1. stage 0 和最后一个 stage 的 warmup 数为什么不同？
2. `num_microbatches_remaining` 在 Megatron 里对应 patch 的哪个 steady 循环？
3. 为什么 steady 阶段需要保存新 forward 的 tensor，同时弹出最早的 tensor 做 backward？
4. `deallocate_output_tensor` 解决的是调度顺序问题，还是激活内存问题？
5. drill 的 `bubble_ratio` 偏高时，先检查 schedule 还是先检查真实网络带宽？
