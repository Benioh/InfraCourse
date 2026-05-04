# L12 ★★ Capstone · MM-Tiny-Omni（图像 + 语音 + 文本 omni 模型）

> 把前 19 个 patch 串起来，做一个真能 curl 调通的多模态 omni 模型。

## 三档评分

- **Bronze**：`make patch-test M=l35_multimodal_capstone` 7/7 PASS（实现 projector + WER + CLIP score）
- **Silver**：完成 Stage A + Stage B；CLIP > 0.20，WER < 0.5
- **Gold**：完成 Stage A/B/C 全部三阶段；TTFT < 500ms，post-RL +5pp

详见 [CAPSTONE_PLAN.md](./CAPSTONE_PLAN.md) 完整三阶段交付。

## 闭环

```bash
# 1. 拿 Bronze（patch）
cat labs/l35_multimodal_capstone/patch/task.md
$EDITOR labs/l35_multimodal_capstone/patch/starter/mm_omni.py
make patch-test M=l35_multimodal_capstone

# 2. 拿 Silver/Gold（看 CAPSTONE_PLAN.md）
cat labs/l35_multimodal_capstone/CAPSTONE_PLAN.md
# 三阶段：架构改造 → 推理服务化 → RL 对齐
```

## Patch 测试覆盖（7 个）

| 测试 | 验证 |
|---|---|
| `test_projector_image_only` | image features → (B, N, llm_dim) |
| `test_projector_audio_only` | audio features → (B, T, llm_dim) |
| `test_projector_both` | 两模态共存时 dict 包含两 key |
| `test_wer_perfect_match` | 相同句子 → 0 |
| `test_wer_known_value` | 标准 case 数值正确 |
| `test_clip_score_identical` | 相同向量 → 1 |
| `test_clip_score_orthogonal` | 正交 → 0 |

## 全课总结

恭喜你跑到这里。如果 Capstone Gold 通过，你的简历可以直接写：

> 我把 Qwen2.5-0.5B 改造成了同时吃图像 + 语音的 omni 模型，
> 用我自己实现的 TP（L02）/ ckpt policy（L03）/ LR scheduler（L04）/
> ring attention（L04.5）/ bucketed DDP（L05）/ MoE router（L05.5）/
> multimodal collator（L06）/ MinHash dedup（L06.3）/ typical_p sampler（L07）/
> RadixCache（L08）/ AWQ quantization（L08.5）/ spec decode（L08.7）/
> Prometheus metrics（L09）/ adaptive KL（L10）/ async rollout（L10.5）/
> weight sync（L11）—— 全部 16 个真补丁串起来部署成 OpenAI 兼容 endpoint。

这就是面试官想看的。
