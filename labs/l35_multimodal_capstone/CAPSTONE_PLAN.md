# L41 Capstone Plan

## Bronze：CPU Patch

Bronze 只验收三个组件：

- `MultimodalProjector`
- `wer_score`
- `clip_score`

通过标准：

```bash
IMPL=reference make patch-test M=l35_multimodal_capstone
```

## Silver：Stage A + Stage B

Stage A 训练目标：

- 冻结 CLIP / Whisper encoder。
- 训练 image/audio projector。
- 只解冻 LLM 的少量后层用于适配。
- 保存训练配置、loss 曲线、样本输入、checkpoint 路径和失败边界。

Stage B 服务目标：

- 暴露 OpenAI Chat Completions 兼容 endpoint。
- 支持 image-only、audio-only、text-only 和混合请求。
- 记录 TTFT、decode latency、encoder time、prefill time、cache hit 和错误样本。

## Gold：Stage C RL + Final Delivery

Stage C 目标：

- 图像任务使用 CLIP image-text score。
- 音频任务使用 `1 - WER`。
- 记录 KL、reward、rollout freshness、weight sync 和 before/after 样本。

Final Delivery 目标：

- 运行 `scripts/run_capstone_aggregator.py`。
- 检查 `artifacts/final_artifacts/evidence_index.json`。
- 补齐 `final_readme.md`、`risk_register.md`、`debug_report.md` 和 `reproducibility_commands.sh`。

## 评分边界

Bronze 证明组件接口正确。Silver 证明模型可训练、可服务。Gold 证明 RL 和最终证据能支持迁移判断。validation-only run 只能作为格式验证，不能作为真实性能结论。
