# L08.8 Patch · 最小评测 harness

## 你要交付什么

```python
def extract_first_number(text: str) -> float | None: ...
def score_exact_match(prediction: str, reference: str) -> bool: ...
def score_first_number_match(prediction: str, reference: str, atol: float = 1e-6) -> bool: ...
def format_few_shot_prompt(
    question: str, shots: list[tuple[str, str]], system: str = "Solve the math problem."
) -> str: ...
def run_evaluation(
    items: list[dict],            # [{"question", "answer"}]
    client,                       # 任意有 .complete(prompt) -> str 的对象（OpenAI / stub）
    n_shots: int = 8,
    max_tokens: int = 256,
) -> dict:                        # {"exact_match", "first_number_match", "n_total", "samples": [...]}
```

补丁规模目标：80–120 行。

## 不变量

1. `extract_first_number` 必须处理负号、小数点、`$`、千分位逗号、`%`
2. `extract_first_number("no number here") == None`
3. `score_exact_match` 大小写无关、首尾空白无关
4. `score_first_number_match` 用 `extract_first_number` 抽数字再比较，容忍 `atol`
5. few-shot prompt 顺序：system → shot[0] question → shot[0] answer → … → 真实 question →（不能跟 answer，留给模型生成）
6. `run_evaluation` 必须返回每条样本的 prediction 与是否命中

## 怎么验证

```bash
make patch-test M=l25_serve_eval_lm_eval
```

## 写完之后你能做什么

- 把任何 vLLM / SGLang serve 接到 GSM8K / MMLU / HumanEval 评测
- 给训练 ckpt 做 sanity check：训完不掉点才放线上
- 给 capstone 提供"模型服务可用 + 数学正确率"双指标
