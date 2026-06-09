# L26：Serving Eval · OpenAI-Compatible GSM8K Harness

Serving 系统做完调度、KV、量化或 spec decode 优化后，必须回答一个很普通但很硬的问题：服务还答对吗？只报告 tokens/s、TTFT 或 ITL，会漏掉输出格式漂移、数值错误、超长续写、timeout、stop 配置错误和 prompt 泄漏。L26 用一个最小 GSM8K-style harness，把质量评测写成可执行、可复查、可落盘的工程链路。

## 1. 本讲目标

- 写出一个最小 evaluation harness：切分 shots 和 eval items、构造 few-shot prompt、调用 completion client、聚合指标。
- 区分 `exact_match`、`first_number_match` 和 completion rate 的含义与边界。
- 解释 deterministic serving eval 为什么要固定 temperature、stop sequence、max tokens、模型版本和 prompt 集。
- 读懂本讲 patch、run_eval、SGLang simple eval、vLLM/SGLang OpenAI-compatible endpoint 的主路径。
- 用 artifact 复盘评测异常：格式错、数值错、timeout、空答、endpoint 配置错和 few-shot leakage。

## 2. 问题背景：Serving 优化需要质量闭环

推理服务优化经常先看性能：吞吐、TTFT、ITL、GPU memory、cache hit、acceptance 或 batch utilization。这些指标只说明系统跑得怎么样，不能说明答案是否还正确。量化可能让数值题错在最后一步；spec decode 的配置错误可能改变输出稳定性；缺少 stop sequence 时，模型会继续生成下一道题；timeout 被静默记成空答时，分数下降的根因会被藏起来。

评测闭环要把一次服务运行变成可复查证据。最小证据包括：评测配置、输入样本、few-shot prompt 规则、生成参数、模型输出、打分函数、每样本结果、聚合指标、运行命令和服务版本。L26 的 patch 用 stub client 做 CPU-safe 验证，目标是把 harness 的行为合同写清楚；真实质量结论还要接入真实服务。

## 3. Harness 的输入和输出

本讲的 patch 入口是：

```python
def run_evaluation(
    items: list[dict],
    client: Any,
    n_shots: int = 8,
    max_tokens: int = 256,
) -> dict:
    ...
```

`items` 是按顺序排列的题目，每个元素至少包含 `question` 和 `answer`。前 `n_shots` 个样本作为 few-shot examples，后面的样本才进入评测。`client` 只需要实现 `complete(prompt, max_tokens=..., stop=...)`，因此它可以是本地 stub，也可以包装 vLLM/SGLang 的 OpenAI-compatible completion endpoint。

返回值至少包含：

| 字段 | 含义 |
|---|---|
| `exact_match` | prediction 和 reference 经过 strip/lower 后完全一致的比例 |
| `first_number_match` | prediction 和 reference 的第一个数字在容差内相等的比例 |
| `n_total` | 实际评测样本数，不包含 shots |
| `samples` | 每条样本的 question、reference、prediction 和两个命中标记 |

`n_total` 是关键字段。评测报告不能只写分数，还要证明跑完了多少题。若 timeout 导致样本缺失，完成率必须暴露出来。

## 4. 打分函数：exact 和 first-number 解释不同错误

`extract_first_number(text)` 要从字符串中抽取第一个带符号的小数，允许 `$`、`,` 和 `%` 这类格式噪声。测试覆盖 `"Answer: 42."`、`"loss=-1.5"`、`"$1,234"` 和 `"3%"`。课堂 patch 使用 first number，是为了让最小函数可手算；真实 GSM8K 常用 final answer delimiter 或 last number extraction，原因是完整推理文本中间可能出现很多数字。

`score_exact_match(prediction, reference)` 只做 strip 和 lower。它对格式非常敏感，适合检查模型是否严格按要求输出。`score_first_number_match(prediction, reference)` 把答案中的数字抽出来比较，适合隔离“数值对但格式不同”的场景。两者一起看，才能区分三类问题：

| 现象 | 可能解释 |
|---|---|
| exact 高，numeric 高 | 输出格式和数值都稳定 |
| exact 低，numeric 高 | 格式或附加文本问题 |
| exact 低，numeric 低 | 数值错误、prompt 错误、服务错误或 timeout |

数字指标不能替代任务指标。对于多选、代码、事实问答或结构化 JSON，应该使用对应任务的 parser 和 grader。

## 5. Prompt、stop 和 deterministic 设置

Few-shot prompt 的顺序要稳定。`format_few_shot_prompt` 先放 system instruction，再按原顺序写入 `(question, answer)` shots，最后追加待评测 question，并以 `Answer:` 结束。测试要求 Q1、Q2、Q3 顺序不变，prompt 以 `Answer:` 结尾。

Stop sequence 是评测 harness 的一部分。对 completion API，常见 stop 是 `["Question:"]` 或同类题目分隔符，目的是阻止模型继续生成下一题。缺少 stop 时，模型可能输出“Question: ... Answer: ...”，第一数字或 exact match 都会被污染。

Deterministic eval 通常要设置 `temperature=0.0`，并固定 model、endpoint、max_tokens、few-shot 样本、题目顺序和代码版本。若任务需要 self-consistency 或 majority voting，要在报告里写清采样次数和聚合规则。不能把随机采样的一次结果和 greedy baseline 直接比较。

## 6. run_eval 脚本和 artifact

`scripts/run_eval.py` 把 patch harness 接到可复查的运行目录。它读取 YAML config，写入 `command.sh`、`config.resolved.yaml` 和 `prediction.yaml`，根据 mode 选择 `StubClient` 或 `OpenAIClient`，生成 toy GSM8K-style 数据，然后调用 `run_evaluation`。

成功时，它会写：

- `metrics.jsonl`：每条样本一行，包含 prediction、reference、exact 和 first-number 命中。
- `artifacts/eval_summary.json`：聚合指标、样本数、耗时和 accept 状态。
- `report.md`：配置摘要、指标和下一步。

本地 stub 评测的价值是快、稳定、可在 CPU 上跑。真实服务评测要切到 OpenAI-compatible endpoint，并记录服务启动命令、模型、端口、采样参数、timeout 和硬件。vLLM 的 `/v1/completions` 会把 request 转为 sampling params 并送进 engine；SGLang 的 HTTP server 也暴露 `/v1/completions` 和 `/v1/chat/completions`。Harness 要和 endpoint 类型匹配。

## 7. 真实 eval 对照

SGLang 的 simple eval 代码展示了完整评测系统常见的结构：Sampler 负责对 OpenAI-compatible API 发请求；GSM8K eval 负责构造 few-shot prompt、抽取答案、并行跑样本、聚合 score；统一 run_eval 根据任务名选择 eval 对象，并把 score、latency 和 output throughput 写入指标。

和 L26 patch 相比，真实 eval 多了这些复杂度：

- 数据下载和缓存。
- 多线程或异步并发。
- retry、timeout、rate limit 和空响应处理。
- chat/completion 两种 API。
- token usage、latency 和 throughput 统计。
- 多任务 parser、grader 和聚合方式。
- 报告上传或统一指标收集。

读真实源码时不要把所有分支一次追完。先找四个点：prompt 怎么构造、client 怎么调用、答案怎么解析、指标怎么聚合。

## 8. Debug 路线

Exact match 低、first-number 高时，先看输出样本。常见原因是多了单位、句号、解释文本或大小写差异。此时要决定任务是否真的要求严格格式；如果要求 JSON 或单词答案，就不能只看 numeric metric。

Exact 和 first-number 都低时，先查 prompt、shot 顺序、temperature、stop、endpoint、模型和 timeout。若 stub 全对而真实服务低，问题大概率在服务端、tokenizer/chat template、生成参数或模型能力。若 stub 也失败，先回到 patch 函数和 tests。

评测样本数少于 `n_eval` 时，要把它当成完成率问题，而不是普通错误答案。报告里要区分 skipped、timeout、exception、empty response 和 wrong answer。生产服务评测中，完成率本身就是质量指标。

## Lab 验收边界

本讲 patch 命令：

```bash
make patch-test M=l25_serve_eval_lm_eval
```

patch 验收的是五个函数：

1. `extract_first_number`：抽取第一个格式化数字。
2. `score_exact_match`：strip/lower 后比较。
3. `score_first_number_match`：数字抽取后按容差比较。
4. `format_few_shot_prompt`：保持 shots 顺序并以 `Answer:` 结尾。
5. `run_evaluation`：切分 shots/eval items，调用 client，记录每样本和聚合指标。

测试通过后，还要跑本地 stub eval，检查 artifact 是否完整。真实模型评测要使用同结构配置补充完成率、质量指标和服务端证据。

## 9. 小结

L26 的核心是把 serving 质量评测做成工程闭环。一个分数不够，必须能追到样本、prompt、生成参数、prediction、打分函数、完成率、命令和配置。前面几讲的服务优化都应接到这个闭环上：速度和显存收益只有在质量边界可复查时才有上线价值。
