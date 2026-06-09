# L41：MM-Tiny-Omni Capstone

L41 是最后一讲。它不再只看单个函数或单个服务，而是要求把一个小型多模态项目拆成可训练、可服务、可对齐、可交付的证据链。Bronze 先用 CPU patch 锁住 projector、WER 和 CLIP score；Silver/Gold 再要求 Stage A/B/C 和 final artifacts。

## 1. 本讲目标

- 解释 MM-Tiny-Omni 的教学边界：image/audio/text 到文本输出。
- 实现 `MultimodalProjector`、`wer_score` 和 `clip_score`。
- 说明 Stage A 训练、Stage B serving、Stage C RL 各自的输入、输出和指标。
- 读懂 final artifacts 聚合器如何扫描 runs 并生成 evidence package。
- 判断 validation-only、缺 run 和真实性能证据之间的差异。

## 2. Bronze 组件

`MultimodalProjector` 把 CLIP 和 Whisper 的连续特征投到 LLM hidden dim。图像输入形状是 `(B, num_image_tokens, vit_dim)`，音频输入形状是 `(B, T, whisper_dim)`。输出 dict 包含 `image_emb` 和 `audio_emb`，缺失模态返回 `None`。图像路径需要 learnable positional embedding，因为 patch 顺序本身携带空间信息。

`wer_score` 是 audio reward 的最小实现。它把 hypothesis 和 reference 按空格切词，用 word-level Levenshtein DP 算 edit distance，再除以 reference 词数。reference 为空时要有稳定边界。

`clip_score` 是 image reward 的最小实现。它把 image embedding 和 text embedding 归一化后做余弦相似度。相同向量应接近 1，正交向量应接近 0。

## 3. Stage A：训练

Stage A 的目标是让 LLM 接收 image/audio soft tokens。建议冻结 CLIP 和 Whisper encoder，训练 projector，并只解冻 LLM 少量后层。训练证据至少包括数据样本、collator 输出 shape、loss 曲线、checkpoint 路径、显存峰值和失败边界。

embedding injection 的关键顺序是：文本 token 先走 LLM embedding；image/audio feature 经过 projector；再把 projector 输出写回对应占位位置；最后用 `inputs_embeds` 进入 LLM。这里最常见的错误是 modal span 错位和 attention mask 不一致。

## 4. Stage B：服务化

Stage B 把 Stage A checkpoint 包成 OpenAI Chat Completions 兼容 endpoint。服务端要解析 messages 中的 image/audio，运行对应 encoder，复用训练时的 projector 和 injection 逻辑，再交给 LLM prefill/decode。

服务证据至少要拆 TTFT、encoder time、prefill time、decode latency、cache hit、错误样本和请求类型。image-only、audio-only、text-only 和混合请求要分别验收。

## 5. Stage C：RL 对齐

Stage C 使用任务条件 reward。图像任务使用 CLIP image-text score；音频任务使用 `1 - WER`。RL 证据要记录 reward、KL、ratio、rollout freshness、weight sync、before/after 样本和失败样本。

这里不能把所有 reward 混成一个黑盒分数。CLIP score 不适用于纯音频转写；WER 不适用于图像 caption。按任务路由 reward，才能解释 RL 指标。

## 6. Final Artifacts

`run_capstone_aggregator.py` 扫描 `runs/` 中已有 run，读取 command、config、metrics、report、grade 和 artifacts。它会生成 `evidence_index.json`、stage reports、final README、reproducibility commands、debug report、risk register 和 architecture diagram。

聚合器的边界很重要。它可以暴露证据缺口，但不能把 validation-only run 变成真实性能结论。最终报告要明确哪些行是格式验证，哪些行来自真实训练、服务或 RL 运行。

## 7. 排障顺序

Capstone 出问题时按证据链排查：

1. Bronze patch 是否通过，projector/WER/CLIP 是否满足数值边界。
2. Stage A 的 batch schema、modal span、mask 和 checkpoint 是否完整。
3. Stage B 的 media parsing、encoder、prefill 和 decode 是否分段记录。
4. Stage C 的 reward 路由、KL、rollout freshness 和 weight sync 是否有证据。
5. final artifacts 是否能从每一行回到 command/config/metrics/report。

## Lab 验收边界

```bash
IMPL=reference make patch-test M=l35_multimodal_capstone
python labs/l35_multimodal_capstone/scripts/run_capstone_aggregator.py --run-id l41_validation
```

patch 验收 Bronze 组件。聚合器验收证据包格式。真实 Capstone 还需要 Stage A/B/C 的实际运行结果。

---

## 补充：多模态模型接入的工程要点

### 多模态 Token 对齐

多模态模型的核心工程难点在于"异构 token"对齐：

| Token 类型 | 来源 | Shape | 进入 LLM 的方式 |
|---|---|---|---|
| 文本 token | Tokenizer | `[B, T_text]` → embedding → `[B, T_text, H]` | 直接 embedding lookup |
| 图像 token | CLIP ViT | `[B, N_patches, ViT_dim]` → projector → `[B, N_patches, H]` | 替换占位位置 |
| 音频 token | Whisper | `[B, T_audio, Whisper_dim]` → projector → `[B, T_audio, H]` | 替换占位位置 |

**工程陷阱**：
1. **Modal span 错位**：占位 token 数量必须和 projector 输出的 token 数量一致。ViT 的 patch 数是固定的（如 224/14=16, 16×16=256 patches），但变分辨率输入会改变这个数。
2. **Attention mask 一致性**：图像/音频 token 必须出现在 attention mask 为 1 的位置。
3. **Position IDs**：多模态 token 是否共享文本的 position 编号？不同框架策略不同。
4. **Batch 中混合模态**：有些样本只有文本，有些有图像+文本。需要正确处理缺失模态（projector 不调用，对应位置用 padding）。

### Projector 设计选择

| 类型 | 结构 | 优点 | 缺点 |
|---|---|---|---|
| Linear | 一层 `nn.Linear(vit_dim, llm_dim)` | 简单，训练快 | 表达力有限 |
| MLP | 2-3 层 MLP + GELU | 更好的特征映射 | 更多参数 |
| Perceiver | Cross-attention 到 learnable queries | 可控输出 token 数 | 训练更复杂 |
| Q-Former (BLIP-2) | Query transformer | 强大的视觉-语言对齐 | 参数多，预训练成本高 |

### 训练策略

多模态训练通常分阶段冻结/解冻：

```
Stage 0: 预训练 encoder（已有，如 CLIP/Whisper）— 冻结
Stage 1: 训练 projector（连接器）— 解冻 projector，冻结 encoder + LLM
Stage 2: 微调 LLM 部分层 — 解冻 LLM 后几层 + projector
Stage 3: 全量微调（可选）— 全部解冻
```

每个阶段的学习率、数据量和 epoch 数都不同。Stage 1 通常只需要少量数据（image-caption pairs），Stage 2 需要指令数据。

### Serving 时的额外复杂度

- Encoder 前向比 LLM embedding lookup 慢很多（ViT 本身就是一个模型）
- KV cache 中图像/音频 token 的 KV 占据固定位置
- 混合请求 batch：text-only 和 multimodal 请求在同一 batch 中需要不同处理路径
- TTFT 分解：encoder_time + projector_time + prefill_time
