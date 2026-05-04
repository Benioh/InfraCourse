# MM-Tiny-Omni · Capstone 三阶段交付计划

> 写完 patch 拿 Bronze；做完三阶段交付拿 Silver / Gold。

## Bronze（仅 patch）

`make patch-test M=l35_multimodal_capstone` 7/7 PASS = Bronze。
此时你已经有了：MultimodalProjector + WER + CLIP score 三个 lego 件。

下面三阶段是真做项目。

---

## Stage A · 架构改造 + Projector 训练（Megatron 路径）

### 目标
把 Qwen2.5-0.5B 改成能吃 image + audio 的多模态模型。

### 步骤

1. **拉取数据集**（半天）
   ```bash
   # COCO captions（图像）
   huggingface-cli download HuggingFaceM4/COCO --local-dir data/coco
   # LibriSpeech-clean（音频）
   wget https://openslr.elda.org/resources/12/train-clean-100.tar.gz
   tar -xzf train-clean-100.tar.gz -C data/librispeech/
   ```

2. **预处理**（半天）
   - COCO：subset 50k pairs，调用 CLIP-ViT-B/32 image encoder 离线提取 patch features
     存成 `(num_patches, 512)` 的 npy
   - LibriSpeech：调用 Whisper-tiny encoder 离线提取 mel encoder 输出
     存成 `(T, 384)` 的 npy
   - 用 L06 你写的 `multimodal_collate` 拼 batch

3. **构建模型**
   ```python
   import torch.nn as nn
   from transformers import AutoModelForCausalLM
   from starter.mm_omni import MultimodalProjector  # 你的 patch

   llm = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
   for p in llm.parameters():
       p.requires_grad_(False)
   # 解冻最后 4 层
   for layer in llm.model.layers[-4:]:
       for p in layer.parameters():
           p.requires_grad_(True)

   projector = MultimodalProjector(
       vit_dim=512, whisper_dim=384, llm_dim=llm.config.hidden_size,
       num_image_tokens=49,  # CLIP-ViT-B/32 输出 7×7=49 patches
   )

   class MMOmni(nn.Module):
       def __init__(self):
           super().__init__()
           self.llm = llm
           self.projector = projector
       def forward(self, input_ids, modal_type_ids, image_features=None, audio_features=None, attention_mask=None, labels=None):
           # text embedding
           text_emb = self.llm.model.embed_tokens(input_ids)
           # project image / audio
           proj_out = self.projector(image_features=image_features, audio_features=audio_features)
           # inject embedding (按 modal_type_ids 替换 IMAGE_TOKEN/AUDIO_TOKEN 位置)
           if proj_out["image_emb"] is not None:
               # 找出 modal=1 的位置, 替换
               img_mask = modal_type_ids == 1
               text_emb[img_mask] = proj_out["image_emb"].view(-1, text_emb.shape[-1])
           if proj_out["audio_emb"] is not None:
               aud_mask = modal_type_ids == 2
               text_emb[aud_mask] = proj_out["audio_emb"].view(-1, text_emb.shape[-1])
           # forward
           out = self.llm(inputs_embeds=text_emb, attention_mask=attention_mask, labels=labels)
           return out
   ```

4. **训练**
   - 用你 L04 写的 `CosineWithRestartsLR` 调 lr
   - 用你 L05 写的 `BucketedManualDDP` 做 DDP
   - 用你 L03 写的 `selective_checkpoint_wrap(model, attention_only_policy)` 给最后 4 层 LLM 加 ckpt
   - 配置：
     ```
     batch=4 image + 4 audio (oversample audio 5×)
     lr=2e-4 max, 1e-5 min
     epoch=2
     gradient accumulation=4
     ```
   - 1×H100 训练时间：~4 小时

5. **评估**
   - 内部 eval：取 200 张未见 COCO 图 + 200 段未见 LibriSpeech，跑模型生成
   - 算 CLIP score（image task）+ WER（audio task）
   - **Silver 标准**：CLIP score > 0.20，WER < 0.5
   - **Gold 标准**：CLIP score > 0.25，WER < 0.3

### 交付
- `checkpoints/mm-tiny-omni-stageA/` (model + tokenizer)
- `runs/stageA/loss_curve.png`
- `runs/stageA/eval.json`（含 CLIP score 和 WER）

---

## Stage B · 推理服务化（SGLang 路径）

### 目标
把 Stage A 的 ckpt 部署成 OpenAI Chat Completions 兼容 endpoint。

### 步骤

1. **写 inference engine**：
   ```python
   class MMOmniEngine:
       def __init__(self, ckpt_path):
           self.model = load_mm_omni(ckpt_path)
           self.cache = RadixCache(max_tokens=10000)  # 你的 L08 patch
       
       def chat_completion(self, messages):
           # 解析 OpenAI 格式 messages, 提取 image / audio URL
           # 下载并 encode image / audio
           # 用 multimodal_collate（你的 L06 patch）build inputs
           # forward + generate
           # 返回 OpenAI 格式 response
   ```

2. **HTTP server**（FastAPI）
   ```python
   from fastapi import FastAPI
   app = FastAPI()
   engine = MMOmniEngine("checkpoints/mm-tiny-omni-stageA")
   
   @app.post("/v1/chat/completions")
   async def chat(req: dict):
       return engine.chat_completion(req["messages"])
   ```

3. **测试**：
   ```bash
   curl http://localhost:8000/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"messages": [{"role": "user", "content": [
       {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
       {"type": "text", "text": "What is in this image?"}
     ]}]}'
   ```

4. **加 Prometheus metrics**（用你 L09 patch）：
   - prefill_queue_depth gauge
   - decode_active_seqs gauge
   - prefix_cache_hit_rate (用 record_event hit/miss)

### 交付
- `app/mm_omni_server.py`
- `runs/stageB/curl_test.sh`（验证 image / audio / mixed 三种请求）
- `runs/stageB/metrics_screenshot.png`（Grafana 面板）

**Silver 标准**：endpoint 能响应 image-only / audio-only / 混合三种请求。
**Gold 标准**：TTFT < 500ms（H100），prefix_cache_hit_rate > 0.3 在重复请求 workload 下。

---

## Stage C · RL 对齐（SLiME 路径）

### 目标
用 PPO 微调 Stage A 模型对 image grounding 和 audio transcription 任务对齐。

### 步骤

1. **rollout pipeline**（用你 L10.5 RolloutPool）：
   ```python
   pool = RolloutPool(generate_fn=engine.generate_async, max_concurrency=8)
   prompts = sample_from_dataset(32)  # mix image / audio
   responses = await pool.rollout(prompts)
   ```

2. **reward computation**（用你 L12 patch + 外部 CLIP）：
   ```python
   from transformers import CLIPModel
   clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
   
   for sample in batch:
       if sample.modality == "image":
           reward = clip_score(clip.encode_image(sample.image),
                               clip.encode_text(generated_text))
       elif sample.modality == "audio":
           reward = 1.0 - wer_score(generated_text, sample.transcript)
   ```

3. **PPO loop**（用你 L10 KL controller）：
   ```python
   kl_ctrl = AdaptiveKLController(0.2, target_kl=0.05, horizon=1000)
   for step in range(50):
       prompts = sample_batch(32)
       responses = await pool.rollout(prompts)
       rewards = compute_rewards(responses)
       kl = compute_kl(policy, ref_policy)
       loss = -reward + kl_ctrl.get_coef() * kl + entropy_bonus
       loss.backward(); optimizer.step()
       kl_ctrl.update(kl)
       if step % 5 == 0:
           weight_sync()  # 你的 L11 patch
   ```

4. **eval**：
   - 与 Stage A 模型对比同一组 prompt
   - **Silver 标准**：reward 上升、KL 稳定（不发散）
   - **Gold 标准**：post-RL CLIP score / WER 比 Stage A 提升 ≥ 5pp

### 交付
- `runs/stageC/reward_curve.png`、`kl_curve.png`
- `runs/stageC/before_after_samples.md`（10 个 prompt 的 pre/post 对比）

---

## 最终 Capstone 报告

写一份 5-8 页 markdown 报告，至少包含：

1. **三阶段数字总结**（loss / acc / TTFT / TPS / reward / KL）
2. **架构图**（手画或 draw.io）
3. **3 个真改过的框架代码 diff 截图**：
   - 来自前 19 关任意 patch（推荐：L02 TP / L05.5 MoE / L08 RadixCache）
4. **"如果给我一个月会做什么"**（面试 follow-up）

提交到 `final_artifacts/capstone/REPORT.md`。

---

## 评分总览

| 项目 | Bronze | Silver | Gold |
|---|---|---|---|
| Stage A | projector loss 收敛 | CLIP > 0.20，WER < 0.5 | CLIP > 0.25，WER < 0.3 |
| Stage B | endpoint 能返 caption | 三模态混合输入支持 | TTFT < 500ms，prefix hit > 0.3 |
| Stage C | reward 跑通无 NaN | reward 上升、KL 稳定 | post-RL +5pp 提升 |

每档对应 `final_artifacts/capstone/{bronze,silver,gold}/checklist.md`，autograder 检查文件存在 + 数字门槛。
