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

## v2 增量在 Capstone 里怎么用

5 个新 lab 不是孤立外挂，应当作为 Capstone 三阶段的工程基石融进 Gold 评分：

| 阶段 | 必装 patch | 解决的问题 |
|---|---|---|
| Stage A 训练前数据 | **L09.7 Multi-turn Chat Mask**（l28.5） | 多轮 SFT 数据的 loss mask 不能搞错，否则 reward 全是噪声 |
| Stage A 调试 | **L01.3 Memory Snapshot**（l02.5） | Stage A 跑 80 步突然 OOM 时按 stack 定位；不会的人只能盲改 batch size |
| Stage B 推理 | **L10.7 CUDA Graph + Savor**（l30.5） | OpenAI endpoint 的 decode loop 必须 capture/replay；否则 TTFT 拉不到 < 500ms |
| Stage C RL 训练 | **L10.3 Train-Infer Mismatch**（l29.5） | 不加 TIS/MIS 修正，RL 跑到 300 步会因为 mismatch 崩溃，Gold 的 +5pp 拿不到 |
| Stage C 权重同步 | **L11.3 IPC Weight Sync**（l32.5） | actor → rollout 的权重推送必须用 handle tuple，不是 dict copy；否则单次 sync 几十秒 |

Stage C 的 **Gold +5pp** 评估前，必须证明 K3 KL 在训练期间不发散（用 L29.5 实现的 `compute_k3_kl` 出图）；
任何 OOM 的 incident report 必须附 L02.5 的 stack 聚合截图；
任何 Stage B 的 latency 报告必须区分 capture vs replay 的 step time。

## 全课总结

恭喜你跑到这里。如果 Capstone Gold 通过，你的简历可以直接写：

> 我把 Qwen2.5-0.5B 改造成了同时吃图像 + 语音的 omni 模型，
> 用我自己实现的 TP（L02）/ ckpt policy（L03）/ LR scheduler（L04）/
> ring attention（L04.5）/ bucketed DDP（L05）/ MoE router（L05.5）/
> multimodal collator（L06）/ MinHash dedup（L06.3）/ typical_p sampler（L07）/
> RadixCache（L08）/ AWQ quantization（L08.5）/ spec decode（L08.7）/
> Prometheus metrics（L09）/ adaptive KL（L10）/ async rollout（L10.5）/
> weight sync（L11）—— 全部 16 个真补丁串起来部署成 OpenAI 兼容 endpoint。
>
> **而且我懂"基础设施会以哪些方式骗你"**：在 Capstone 里我用 K3 KL 监控并用 TIS/MIS（L10.3）
> 防住了训推不一致导致的训练崩溃；用 IPC handle tuple weight sync（L11.3）做到 actor → rollout
> 的同步只花几百毫秒；用 CUDA Graph + Memory Savor（L10.7）让 SGLang endpoint 的 decode TTFT
> 稳定在 sub-500ms；用 fixed-base chat template tokenization（L09.7）保证 multi-turn loss mask
> 没有一个 token 错位；OOM 现场用 memory snapshot stack 聚合（L01.3）3 分钟定位泄露源。

这就是面试官想看的。
