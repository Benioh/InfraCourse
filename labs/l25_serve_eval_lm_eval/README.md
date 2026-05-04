# L08.8 · 评测闭环：vLLM serve + lm-eval-harness 风格 GSM8K 100-shot

> 本关只做一件事：**实现一个最小评测 harness**——把模型套在
> OpenAI-compatible endpoint（vLLM / SGLang）后面，跑 GSM8K 100 题，
> 输出 exact_match + first_number_match 两个 metric。

之前课程没有任何评测 lab。L08.8 把这个空白补上：你写完后可以 5 行命令评测任何
HuggingFace causal LM 在数学/事实/格式上的表现。

## 闭环

```bash
cat labs/l25_serve_eval_lm_eval/patch/task.md
$EDITOR labs/l25_serve_eval_lm_eval/patch/starter/eval_harness.py
make patch-test M=l25_serve_eval_lm_eval

# 真实评测：先启 vLLM，再打分
bash labs/l25_serve_eval_lm_eval/scripts/run_eval.sh
```

## 测试覆盖

| 测试 | 验证 |
|---|---|
| `test_extract_first_number` | 从 "Answer: $42." 抽出 42 |
| `test_extract_first_number_signed_decimal` | 处理负号 / 小数 / 千分位 |
| `test_score_exact_match` | 字符串 strip+lower 后相等 |
| `test_score_first_number_match` | 数字相等（容忍格式差） |
| `test_format_few_shot_prompt` | n-shot prompt 顺序 + 分隔符 |
| `test_run_evaluation_with_stub_client` | 用 stub LLM client 完整跑通 100 题 |

## Eval

`scripts/run_eval.py`：

1. 加载 GSM8K 100 题（自动下载或本地 jsonl）
2. 构造 8-shot prompt（前 8 题作 few-shot 例子）
3. 通过 OpenAI client 打到 vLLM/SGLang/任意兼容 endpoint
4. 算 `exact_match` 和 `first_number_match`
5. 输出 metrics.jsonl + report.md

acceptance：
- 完成率 100%（不能因为 timeout 漏题）
- exact_match 与 first_number_match 都打印
- p50 / p95 latency 也记录

## Configs

| 配置 | 用途 |
|---|---|
| `configs/local_stub.yaml` | 不需要真实服务，本地 stub LLM 跑通 |
| `configs/local_vllm_qwen.yaml` | 假定本地 `vllm serve Qwen/Qwen2.5-0.5B-Instruct --port 8000` |
| `configs/h200_llama3.yaml` | 真实 8×H200 上 Llama-3-8B 评测 |

## 进入下一关

通过后进入 [L09 SGLang PD 观测](../l26_sglang_pd_observability/README.md)。
