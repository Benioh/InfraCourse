# L26 Source Reading Card

## 主路径

1. `labs/l25_serve_eval_lm_eval/patch/starter/eval_harness.py`：五个待实现函数，覆盖数字抽取、打分、prompt 和完整评测。
2. `labs/l25_serve_eval_lm_eval/patch/reference/eval_harness.py`：最小正确 harness，展示 shots/eval 切分和每样本记录。
3. `labs/l25_serve_eval_lm_eval/patch/tests/test_patch.py`：测试数字格式、exact/numeric scorer、prompt 顺序和 stub 聚合。
4. `labs/l25_serve_eval_lm_eval/scripts/run_eval.py`：读取配置、选择 client、调用 harness、写 metrics 和 report。
5. `github_repo/sglang/python/sglang/test/simple_eval_gsm8k.py`：真实 GSM8K eval 的 prompt、answer parser、样本切分和聚合。
6. `github_repo/sglang/python/sglang/test/simple_eval_common.py`：OpenAI-compatible sampler、temperature、stop、retry 和 token usage。
7. `github_repo/sglang/python/sglang/test/run_eval.py`：根据任务名选择 eval，记录 score、latency 和 throughput。
8. `github_repo/vllm/vllm/entrypoints/openai/completion/serving.py`：vLLM completion request 到 sampling params 和 response 的主路径。
9. `github_repo/sglang/python/sglang/srt/entrypoints/http_server.py`：SGLang 的 `/v1/completions` 和 `/v1/chat/completions` endpoint。

## 必须记住的字段关系

- `shots = items[:n_shots]`。
- `eval_items = items[n_shots:]`。
- `n_total = len(eval_items)`。
- `samples` 必须保留 question、reference、prediction、exact 和 numeric 命中。
- `exact_match` 用于严格格式。
- `first_number_match` 用于隔离数值正确但格式不同的样本。
- `completion rate` 用于暴露 timeout、异常和漏题。

## 读源码时先跳过

- 多任务 eval registry 的全部分支。
- HTML report 模板细节。
- 多模态、MMLU、HumanEval 等任务特定 parser。
- streaming response 的完整处理。

## 自检问题

1. 为什么 few-shot examples 不能计入 `n_total`？
2. `stop=["Question:"]` 在 completion eval 里防止什么问题？
3. exact 低而 first-number 高时，报告应如何解释？
4. `run_eval.py` 写了哪些 artifact？
5. 真实 endpoint 比 stub 多了哪些失败点？
