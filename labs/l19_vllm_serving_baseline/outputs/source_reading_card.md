# L20 Source Reading Card

## 主路径

1. `labs/l19_vllm_serving_baseline/patch/reference/typical_p.py`：typical-p logits filter。
2. `labs/l19_vllm_serving_baseline/patch/tests/test_patch.py`：5 个行为边界。
3. `mini_infra/vllm/entrypoints/openai/api_server.py`：OpenAI facade 到 engine。
4. `mini_infra/vllm/v1/engine/llm_engine.py`：add_request、step、RequestOutput。
5. `github_repo/vllm/vllm/sampling_params.py`：temperature、top_p、top_k、min_p 的参数边界。
6. `github_repo/vllm/vllm/v1/sample/sampler.py`：真实 sampler 的处理顺序。
7. `labs/l19_vllm_serving_baseline/scripts/run_vllm_lab.py`：serving smoke artifact。
8. `labs/l19_vllm_serving_baseline/scripts/parse_vllm_log.py`：日志指标解析。

## 自检

- 我能否解释 typical-p 为什么按 `abs(-log(p) - entropy)` 排序？
- 我能否指出 vLLM sampler 在哪里应用 temperature 和 top-k/top-p？
- 我能否判断一份 report 是真实 served metrics 还是 validation_only？
