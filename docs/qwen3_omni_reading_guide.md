# Qwen3-Omni / S2 Pro 阅读指引：从 Capstone 简化版回到生产架构

> 本文配套 [Capstone L35](../labs/l35_multimodal_capstone/README.md) 与 Zhaochen Yang 的
> [Codec、RVQ、Dual AR、Thinker-Talker——深入 Qwen3-Omni 与 S2 Pro 的 Omni 模型推理流程](../github_repo/Awesome-ML-SYS-Tutorial/transformers/omni/readme.md)。
>
> Capstone 的 **MM-Tiny-Omni** 是一个故意 stripped down 的教学版。本文回答两个问题：
> 1. 我们裁掉了什么、为什么裁？
> 2. 想把它升级到生产 Qwen3-Omni 形态需要补什么？

## 1. 架构对照表

| 组件 | 生产 Qwen3-Omni / S2 Pro | Capstone MM-Tiny-Omni | 取舍理由 |
|---|---|---|---|
| **图像 encoder** | ViT-L 多分辨率 + 动态分块 | CLIP-ViT-B/32 单一分辨率 | 裁掉动态分块复杂度；保留"image features → projector → LLM"主线 |
| **音频 encoder** | Whisper-large 风格自研 + RVQ codec | Whisper-tiny + MLP projector，**无 codec** | RVQ 带来一整套 codebook + EMA 训练逻辑，对教学来说权重过高 |
| **LLM 主干** | Qwen3 base | Qwen2.5-0.5B | 模型小，CPU 也能跑通端到端 |
| **输出头** | **Dual AR**：Thinker（文本） + Talker（speech token） | **仅 Thinker**：只输出文本 | Talker 涉及 codec 反向解码 + 流式合成，单独做一节都嫌挤 |
| **流式 TTS** | Talker AR + codec decoder + vocoder | ❌ 无 | 等同于不做 speech-out |
| **接口** | 多模态聊天 + speech-out streaming | OpenAI Chat Completions（text only） | curl 兼容性优先 |

## 2. 推理流程对照

### Qwen3-Omni 完整推理流程（Zhaochen 原文 §3）

```
input audio ──► Whisper encoder ──► RVQ codec ──► audio tokens ──┐
                                                                 ├──► Thinker LLM ──► text tokens ──┐
input image ──► ViT encoder ──► image tokens ────────────────────┘                                  │
                                                                                                    ▼
                                                                            Talker LLM ──► speech tokens ──► codec decoder ──► waveform
```

### Capstone MM-Tiny-Omni 简化流程

```
input audio ──► Whisper-tiny encoder ──► MLP projector ──► soft tokens ──┐
                                                                         ├──► Qwen2.5-0.5B LLM ──► text response
input image ──► CLIP-ViT-B/32 ─────────► MLP projector ──► soft tokens ──┘
```

**关键差异**：
1. **没有 RVQ**：audio features 直接走 MLP 拉到 LLM hidden dim，不经过离散化。代价是模型必须把连续 audio embedding 当成"假 token"来处理；好处是省掉了 codec 训练。
2. **没有 Talker**：只有 Thinker 输出文本。S2 Pro 的核心创新（Dual AR 解耦推理与语音生成）在 Capstone 里看不到。
3. **没有 streaming TTS**：Capstone 的 OpenAI endpoint 是文本流式，不是音频流式。

## 3. 想拿真正的 Capstone Gold+，补什么

如果你 Gold 已通过，想往 production grade 推：

### Step 1: 加 RVQ codec（中等难度，约 1 周）

参考 Zhaochen 原文 §2.1 RVQ 部分。要做：
- 训练一个小 codec：Whisper encoder output → 8-codebook RVQ → 离散 audio token
- LLM tokenizer 词表里 reserve 1024–4096 个 audio token id
- Stage A 训练时，把 audio token 当成正常文本 token 一起喂

**为什么 codec 重要**：离散 token 让 LLM 可以用同一个词表统一处理 text/image/audio，
不再依赖"projector 把 embedding 拉到 hidden dim"这种 hacky 桥接。

### Step 2: 加 Talker AR（高难度，约 2 周）

参考 Zhaochen 原文 §3 Thinker-Talker 部分。要做：
- 在 LLM 后面接一个独立的 AR head（Talker），输入是 Thinker 的 hidden states
- Talker 输出 codec audio token，再经过 codec decoder 还原 waveform
- 训练时用 chunk-wise teacher forcing，推理时用 chunk-wise streaming

**为什么 Dual AR 比单 AR 强**：Thinker 可以专注 reasoning，Talker 可以并行生成 audio
（不用等 Thinker 完整生成完 text 才能开始合成 audio）。S2 Pro 的延迟优势主要来自这。

### Step 3: 流式 endpoint（中等，约 3 天）

把 Stage B 的 OpenAI Chat Completions 升级到流式 audio 输出：
- SGLang scheduler 加 audio token output stream
- Talker AR 边生成边送进 codec decoder
- 用 WebRTC / WebSocket 推回客户端

## 4. 测试方案差异

Capstone 的 7 个 patch test 只覆盖 projector + WER + CLIP score。生产 Qwen3-Omni 还要：

| 维度 | Capstone | 生产 |
|---|---|---|
| Audio quality | WER（识别准确度） | WER + MOS（自然度） + Speaker Similarity |
| Image grounding | CLIP score | CLIP + 人工对照 + VQA accuracy |
| Latency | TTFT < 500ms（text only） | TTFT < 300ms + first audio chunk < 800ms |
| Streaming | text streaming | audio streaming + text streaming 同步 |
| RL reward | CLIP score + WER reward | + 人工偏好 reward + audio MOS reward |

## 5. 配套阅读顺序（建议）

1. 先做完 Capstone Gold（确保 text-only omni 能 curl 跑通）。
2. 读 Zhaochen 原文 §1–§2（codec 与 RVQ 的概念，结合本指南 §3 Step 1）。
3. 读 §3（Dual AR 与 Thinker-Talker，对应本指南 §3 Step 2）。
4. 读 §4（部署与流式优化，对应本指南 §3 Step 3）。
5. 在自己的 fork 里实现 Step 1 → 2 → 3 中的任意一个，作为 Capstone "Platinum" 等级。

## 6. 一句话总结

> Capstone 的 MM-Tiny-Omni 是 Qwen3-Omni 的 **architectural skeleton**：保留了
> "多模态 encoder + projector + LLM + OpenAI endpoint" 的 minimum loop，
> 但裁掉了 RVQ codec 与 Dual AR Talker 这两个让生产 omni 模型真正强大的组件。
> 学完 Capstone 你不会自动会做 S2 Pro，但你会**清楚地知道差距在哪**——这本身就是
> 区分"会调包"和"懂系统"的分水岭。
