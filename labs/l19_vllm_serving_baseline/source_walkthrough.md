# 源码带读：L20 vLLM Serving Baseline

按“patch 机制 -> MiniInfra 请求路径 -> vLLM sampler -> smoke artifact”的顺序读。

## 0. 源码地图

```text
labs/l19_vllm_serving_baseline/patch/reference/typical_p.py
labs/l19_vllm_serving_baseline/patch/tests/test_patch.py
mini_infra/vllm/entrypoints/openai/api_server.py
mini_infra/vllm/v1/engine/llm_engine.py
github_repo/vllm/vllm/sampling_params.py
github_repo/vllm/vllm/v1/sample/sampler.py
labs/l19_vllm_serving_baseline/scripts/run_vllm_lab.py
labs/l19_vllm_serving_baseline/scripts/parse_vllm_log.py
```

## 1. Patch reference

文件：`labs/l19_vllm_serving_baseline/patch/reference/typical_p.py`

- L8-L14：`typical_p >= 1.0` 时直接返回 logits。
- L16-L19：计算概率、surprisal、entropy 和 distance。
- L21-L29：按 distance 排序，累计概率，并保证第一个 token 保留。
- L31-L34：把删除 mask scatter 回原 vocab 顺序，再 masked fill。

## 2. Patch tests

文件：`labs/l19_vllm_serving_baseline/patch/tests/test_patch.py`

- L22-L27：阈值为 1 时不改变 logits。
- L30-L37：极小阈值只保留一个 token。
- L40-L50：filter value 只作用在被删除位置。
- L53-L61：计算 distance，后续用于验证保留集合。
- L63-L71：保留 token 的最大 distance 不超过被删 token 的最小 distance。
- L74-L83：batch 每行至少保留一个 token。

## 3. MiniInfra serving path

文件：`mini_infra/vllm/entrypoints/openai/api_server.py`

- L8-L14：OpenAI facade 持有 engine 和 model 名。
- L15-L23：messages 拼成 prompt，进入 engine，并 step 到完成。
- L24-L35：返回 OpenAI-like chat completion 和 debug 字段。

文件：`mini_infra/vllm/v1/engine/llm_engine.py`

- L20-L23：engine 创建 KV manager、scheduler 和 outputs。
- L25-L28：`add_request` 把 request 交给 scheduler。
- L30-L42：`step` 调 scheduler，并为 scheduled decode 追加 token。
- L43-L51：finished request 写入最终 RequestOutput。

## 4. vLLM sampler

文件：`github_repo/vllm/vllm/sampling_params.py`

- L201-L214：temperature、top_p、top_k、min_p 的参数定义。
- L425-L430：greedy 模式下 top_p/top_k/min_p 被重置。
- L466-L488：验证 temperature、top_p、top_k、min_p 的合法范围。

文件：`github_repo/vllm/vllm/v1/sample/sampler.py`

- L21-L30：Sampler docstring 开始描述处理顺序。
- L31-L42：logits processors 和 penalties 在采样前应用。
- L43-L52：随机采样路径包含 temperature、processors、top-k/top-p。
- L90-L97：forward 将 logits 转 float32 并调用 sampler。
- L261-L269：sample 阶段应用 temperature 和 argmax-invariant processors。
- L271-L277：调用 top-k/top-p sampler。
- L279-L288：根据 greedy/random 路径选择返回 token。

## 5. Smoke scripts

文件：`labs/l19_vllm_serving_baseline/scripts/run_vllm_lab.py`

- L37-L50：读取 config，创建 run dir，写命令快照和 resolved config。
- L51-L57：检测 vLLM 包、端口和 serve 命令，并写 serve.log。
- L60-L78：端口打开时发送 OpenAI-compatible 请求并测量耗时。
- L79-L90：写 serving metrics row，区分 served 和 validation_only。
- L91-L104：report 写目标、环境和配置。

文件：`labs/l19_vllm_serving_baseline/scripts/parse_vllm_log.py`

- L13-L33：定义 TTFT、ITL、吞吐等日志 pattern。
- L45-L61：解析可用性、端口、命令和行数。
- L62-L82：抽取指标并生成 workload fingerprint 和 status。

## 自检问题

1. typical-p 和 top-p 的排序依据有什么不同？
2. 为什么 scatter 回原 vocab 顺序是必要步骤？
3. smoke 的 `validation_only` 能证明什么，不能证明什么？
