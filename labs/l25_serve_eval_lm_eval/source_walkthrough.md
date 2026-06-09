# 源码带读：L26 Serving Eval Harness

这份带读按“patch 合同 -> 本地 run_eval -> SGLang eval -> OpenAI-compatible endpoint”的顺序组织。目标是先闭合一次评测请求从样本到 artifact 的主路径，完整评测框架的任务 registry 和并发细节放到后面再看。

## 0. 源码地图

```text
labs/l25_serve_eval_lm_eval/patch/starter/eval_harness.py
labs/l25_serve_eval_lm_eval/patch/reference/eval_harness.py
labs/l25_serve_eval_lm_eval/patch/tests/test_patch.py
labs/l25_serve_eval_lm_eval/scripts/run_eval.py
github_repo/sglang/python/sglang/test/simple_eval_gsm8k.py
github_repo/sglang/python/sglang/test/simple_eval_common.py
github_repo/sglang/python/sglang/test/run_eval.py
github_repo/vllm/vllm/entrypoints/openai/completion/serving.py
github_repo/sglang/python/sglang/srt/entrypoints/http_server.py
```

## 1. Patch starter：确认五个待实现函数

文件：`labs/l25_serve_eval_lm_eval/patch/starter/eval_harness.py`

重点看：数字抽取、两个 scorer、prompt builder 和 `run_evaluation`。

建议阅读顺序：

- L9-L13：`extract_first_number` 要处理符号、小数、`$`、`,` 和 `%`。
- L16-L23：两个 scorer 分别处理 exact match 和 numeric match。
- L26-L33：prompt builder 要保持 shots 顺序，并以待评测 question 的 `Answer:` 结束。
- L36-L46：`run_evaluation` 要切分 shots/eval items，调用 client，记录样本和聚合指标。

可以先跳过：typing import 和文件头说明。

## 2. Patch reference：看最小正确 harness

文件：`labs/l25_serve_eval_lm_eval/patch/reference/eval_harness.py`

重点看：reference 怎样把局部函数串成完整评测。

建议阅读顺序：

- L8-L14：正则定义和空字符串边界。
- L15-L22：匹配后清理 `$`、`,`、`%`，再转成 float。
- L25-L34：exact scorer 和 first-number scorer 的边界。
- L37-L49：few-shot prompt 的系统提示、shots 顺序和最终 `Answer:`。
- L52-L64：`run_evaluation` 校验样本数，切分 shots 和 eval items，初始化计数器。
- L65-L78：每条样本构造 prompt、调用 client、计算两个指标并保存样本记录。
- L79-L87：聚合 exact、first-number、`n_total` 和 samples。

读完后的结论：reference 是可替换 client 的最小 harness；真实服务复杂度在 client 和 runner 里。

## 3. Patch tests：测试覆盖哪些合同

文件：`labs/l25_serve_eval_lm_eval/patch/tests/test_patch.py`

重点看：测试如何用 stub client 固定输出。

建议阅读顺序：

- L16-L17：`IMPL` 决定测试 starter 或 reference。
- L20-L31：数字抽取覆盖普通整数、小数、无数字、负数、千分位和百分号。
- L34-L45：exact scorer 和 first-number scorer 的容差边界。
- L48-L57：few-shot prompt 必须保持 Q1、Q2、Q3 顺序，并以 `Answer:` 结尾。
- L60-L78：stub client 始终返回 42，完整评测应得到 4 条 eval samples 和 100% numeric match。
- L81-L98：stub client 返回 41/42/43/42，numeric match 应为 0.5。

可以先跳过：pytest import 和 Path 处理。

## 4. run_eval：把 harness 接到 artifact

文件：`labs/l25_serve_eval_lm_eval/scripts/run_eval.py`

重点看：配置读取、client 选择、toy dataset、acceptance gate 和 artifact 写入。

建议阅读顺序：

- L38-L46：`IMPL` 可选择 starter 或 reference，导入失败时 fallback 到 reference。
- L49-L60：构造 toy GSM8K-style 数据，字段与 patch items 对齐。
- L63-L72：`StubClient` 从 prompt 最后一题中解析加法并返回答案。
- L75-L100：`OpenAIClient` 构造 `/completions` 请求，固定 temperature，并读取 `choices[0].text`。
- L103-L117：读取 config，准备 run directory，写命令和 resolved config。
- L119-L129：保存 prerequisite serve command，并根据 `n_eval+n_shots` 构造 items。
- L131-L144：按 mode 选择 stub 或 OpenAI client，调用 harness，异常时写 fallback。
- L146-L155：根据完成率和指标阈值判断 accept。
- L157-L176：写每样本 `metrics.jsonl` 和聚合 `eval_summary.json`。
- L177-L204：写 markdown report 并打印 run directory。

读完后的结论：run_eval 是证据链入口，patch 只是其中的评分和循环核心。

## 5. SGLang GSM8K eval：真实评测结构

文件：`github_repo/sglang/python/sglang/test/simple_eval_gsm8k.py`

重点看：few-shot prompt、答案抽取、样本切分和聚合。

建议阅读顺序：

- L21-L29：构造单题 prompt 和 few-shot examples。
- L32-L40：从答案字符串抽取数字，失败时返回 INVALID。
- L43-L60：`GSM8KEval` 读取数据，生成 few-shot prompt。
- L62-L65：评测数据跳过 few-shot examples，避免数据泄漏。
- L67-L75：每条样本构造 prompt message。
- L77-L83：调用 sampler，抽取 answer 并和 correct answer 比较。
- L85-L99：保存 HTML/convo 证据并聚合结果。

可以先跳过：下载工具和 HTML 模板细节。

## 6. SGLang sampler 和统一 run_eval

文件：`github_repo/sglang/python/sglang/test/simple_eval_common.py`

重点看：completion sampler 如何调用 OpenAI-compatible endpoint。

建议阅读顺序：

- L181-L190：`CompletionSampler` 保存 base_url、model、temperature、top_p、max_tokens 和 stop。
- L208-L225：把 message list 拼成 raw prompt，并调用 `client.completions.create()`。
- L226-L241：记录 completion tokens，错误时返回空字符串或重试后空答。

文件：`github_repo/sglang/python/sglang/test/run_eval.py`

建议阅读顺序：

- L64-L85：根据 API mode 选择 completion 或 chat sampler。
- L95-L106：准备环境变量和 base_url。
- L172-L180：`eval_name=gsm8k` 时构造 `GSM8KEval`。
- L184-L195：运行一次 eval，打印 score、latency 和 output throughput。
- L197-L205：把 score 和 latency 写入统一 metric。

读完后的结论：真实 eval 会把 client、任务、并发和指标收集分离，本讲 patch 只保留核心合同。

## 7. vLLM 与 SGLang endpoint：为什么 harness 用 OpenAI-compatible client

文件：`github_repo/vllm/vllm/entrypoints/openai/completion/serving.py`

建议阅读顺序：

- L107-L119：`create_completion` 是 OpenAI Completion API 入口。
- L144-L155：根据 prompt length 和配置计算 max_tokens。
- L156-L165：request 转为 sampling params。
- L190-L201：把 engine input 和 sampling params 送入 engine client。
- L224-L249：非 streaming 路径收集结果并转换成 completion response。

文件：`github_repo/sglang/python/sglang/srt/entrypoints/http_server.py`

建议阅读顺序：

- L1479-L1487：SGLang 暴露 `/v1/completions`，并转给 completion handler。
- L1490-L1497：SGLang 也暴露 `/v1/chat/completions`。

读完后的结论：harness 只需要 OpenAI-compatible completion 形状，就能连接不同 serving backend。

## 读完后的自检问题

1. `n_total` 为什么不能包含 few-shot examples？
2. exact match 和 first-number match 同时记录时，能区分哪些错误类型？
3. 缺少 stop sequence 时，为什么模型可能继续生成下一题？
4. `run_eval.py` 哪些文件构成证据链？
5. stub eval 通过后，真实服务仍可能在哪些位置失败？
