# L12 Capstone Patch · MM-Tiny-Omni 集成组件

> ★★ 这是本课的毕业项目入口。前 19 个 patch 是器件，本关把它们装成一个真正的多模态 omni 模型。

## 项目蓝图：MM-Tiny-Omni（3 阶段）

把 Qwen2.5-0.5B 改造成同时吃 **图像 + 语音 + 文本** 的 omni 模型，并以 OpenAI 兼容
endpoint 部署 + RL 对齐。

```
Stage A · 架构改造 + Projector 训练（Megatron 路径）
   - 接 CLIP-ViT-B/32 image encoder（冻结）
   - 接 Whisper-tiny encoder（冻结）
   - 训练 image_projector + audio_projector + LLM 最后 4 层
   - 数据：~50k image-caption（COCO）+ ~10k speech-transcription（LibriSpeech）

Stage B · 推理服务化（SGLang 路径）
   - 多模态 scheduler 扩展：(image, audio, text) → token sequence
   - 复用 L08 RadixCache 给图像 token 加 prefix cache
   - 暴露 OpenAI Chat Completions 兼容 endpoint

Stage C · RL 对齐（SLiME 路径）
   - CLIP score reward（图像 grounding）+ WER reward（语音转写）
   - 用 L10 KL controller + L11 weight sync 跑 PPO 50 步
```

## 你要交付什么（Patch 部分）

本关 patch 实现 **3 个集成组件**——每个分别对应一个 stage 的核心：

```python
# Stage A: 多模态 projector
class MultimodalProjector(nn.Module):
    def __init__(self, vit_dim, whisper_dim, llm_dim, num_image_tokens):
        # image_proj: Linear(vit_dim → llm_dim)
        # audio_proj: Linear(whisper_dim → llm_dim)
        # image_pos_emb: (num_image_tokens, llm_dim) learnable
    def forward(self, image_features=None, audio_features=None) -> dict: ...

# Stage C: 两个 reward 函数
def wer_score(hypothesis: str, reference: str) -> float:
    """word error rate, 0=perfect, 1=完全错。基于 Levenshtein 编辑距离。"""

def clip_score(image_embed: Tensor, text_embed: Tensor) -> float:
    """归一化余弦相似度。两 embed shape (D,)。返回 [-1, 1]。"""
```

**禁止** 用 `jiwer` / `clip` 现成库做 wer / cosine。
**允许** torch / numpy 基础算子。

补丁规模目标：80–150 行。

## 接口契约

```python
# Stage A
proj = MultimodalProjector(vit_dim=512, whisper_dim=384, llm_dim=896, num_image_tokens=49)
out = proj(image_features=torch.randn(2, 49, 512))
# out["image_emb"].shape == (2, 49, 896)
out2 = proj(audio_features=torch.randn(2, 30, 384))
# out2["audio_emb"].shape == (2, 30, 896)

# Stage C
wer_score("hello world", "hello world")        # 0.0
wer_score("hello world", "hello there")         # 0.5（1 个词错 / 2 个词）
clip_score(torch.randn(512), torch.randn(512)) # ∈ [-1, 1]
```

## 不变量

1. Projector 输出 dim = `llm_dim`（不论输入维度）。
2. Projector 同时支持 image-only / audio-only / 两者都有。
3. wer_score(x, x) = 0；wer_score(x, "完全无关词") ≤ 1。
4. clip_score 归一化（两端不依赖 magnitude），范围在 [-1, 1]。
5. 所有函数支持 batch（即使 starter 只展示单样本）。

## 怎么验证（Patch Track）

```bash
make patch-test M=l35_multimodal_capstone
```

7 个测试：

| 测试 | 验证 |
|---|---|
| `test_projector_image_only` | image features → (B, num_image_tokens, llm_dim) |
| `test_projector_audio_only` | audio features → (B, T, llm_dim) |
| `test_projector_both` | 两者并存时 dict 包含两 key |
| `test_wer_perfect_match` | 相同 → 0.0 |
| `test_wer_known_value` | 标准 case 数值正确 |
| `test_clip_score_identical` | 相同向量 → 1.0 |
| `test_clip_score_orthogonal` | 正交 → ≈ 0 |

## 完整 Capstone 交付（写完 patch 后做）

仅 patch-test 通过 = Bronze Capstone。要拿 Silver/Gold 还需要完成 3 阶段交付，
详见 `labs/l35_multimodal_capstone/CAPSTONE_PLAN.md`（已给完整步骤、数据集
下载、训练命令、评测脚本）。

## 写完之后你能做什么

- 写一份 5–8 页 Capstone 报告，附 3 个真改过的 framework patch 截图。
- 在面试讲清"我把 Qwen2.5-0.5B 改造成了多模态 omni 模型，
  代码里复用了我自己写的 TP/MoE/RadixCache/KL controller/weight sync"。
- 把 mm-tiny-omni 部署成 OpenAI 兼容服务，简历里贴 endpoint 链接。
