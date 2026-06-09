# L26 Patch · 最小 Serving Evaluation Harness

## 你要交付什么

实现一个最小 GSM8K-style evaluation harness。它接收题目列表和一个 `client.complete()` 风格的模型客户端，构造 few-shot prompt，调用 completion，计算 strict 和 numeric 两类指标，并保留每条样本记录。

```python
def run_evaluation(
    items: list[dict],
    client: Any,
    n_shots: int = 8,
    max_tokens: int = 256,
) -> dict:
    ...
```

补丁规模目标：60-100 行。

## 函数合同

1. `extract_first_number(text)`：抽取第一个带符号整数或小数，支持 `$1,234`、`-1.5`、`3%`，返回 `float | None`。
2. `score_exact_match(prediction, reference)`：对两边做 `strip().lower()` 后比较。
3. `score_first_number_match(prediction, reference, atol)`：分别抽取数字，任一边没有数字时返回 `False`，否则按容差比较。
4. `format_few_shot_prompt(question, shots, system)`：先写 system，再按顺序写 shots，最后追加当前 question，并以 `Answer:` 结尾。
5. `run_evaluation(items, client, n_shots, max_tokens)`：前 `n_shots` 条作为 shots，其余样本进入评测；每条样本调用 `client.complete(prompt, max_tokens=max_tokens, stop=["Question:"])`。

## 返回格式

```python
{
    "exact_match": float,
    "first_number_match": float,
    "n_total": int,
    "samples": [
        {
            "question": str,
            "reference": str,
            "prediction": str,
            "exact_match": bool,
            "first_number_match": bool,
        },
        ...
    ],
}
```

`n_total` 只统计 eval items，不包含 few-shot examples。

## 怎么验证

```bash
make patch-test M=l25_serve_eval_lm_eval
```

7 个测试：

| 测试 | 验证 |
|---|---|
| `test_extract_first_number` | 普通整数、小数和无数字 |
| `test_extract_first_number_signed_decimal` | 负数、千分位、百分号 |
| `test_score_exact_match` | strip/lower 后比较 |
| `test_score_first_number_match` | numeric match 和容差 |
| `test_format_few_shot_prompt_order` | few-shot 顺序和 `Answer:` 结尾 |
| `test_run_evaluation_with_stub_client` | stub client 完整跑通 |
| `test_run_evaluation_partial_correct` | 部分正确时聚合指标正确 |

## 写完之后你能做什么

- 把任意 OpenAI-compatible completion endpoint 接到一个可复查的质量评测流程。
- 用 exact 和 numeric 两类指标区分格式问题和数值问题。
- 用 `samples`、`n_total` 和 artifact 定位 timeout、stop 缺失、prompt 泄漏和 endpoint 配置错误。
