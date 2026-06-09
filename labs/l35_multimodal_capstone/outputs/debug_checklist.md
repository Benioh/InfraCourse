# L41 Debug Checklist：MM-Tiny-Omni Capstone

## 1. Bronze

- projector image/audio shape 是否正确。
- image-only、audio-only、双模态缺失处理是否正确。
- WER 的分母是否是 reference word 数。
- CLIP score 是否先 normalize。

## 2. Stage A

- batch schema、modal span、attention mask 是否一致。
- encoder 是否按计划冻结。
- checkpoint、loss 曲线、显存峰值和失败样本是否落盘。

## 3. Stage B

- image/audio/text/mixed 请求是否分别验收。
- TTFT 是否拆成 encoder、prefill、decode。
- endpoint 的 command、config、metrics、report 是否齐全。

## 4. Stage C

- reward 是否按任务路由。
- KL、ratio、rollout freshness 和 weight sync 是否记录。
- before/after 样本是否能解释 reward 变化。

## 5. Final Artifacts

- `evidence_index.json` 是否能回到 run。
- `risk_register.md` 是否列出缺 run 和 validation-only。
- `reproducibility_commands.sh` 是否能复现关键路径。
