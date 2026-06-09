# L41 Debug Tickets

| Ticket | Stage | Symptom | First Check |
|---|---|---|---|
| `mm_projector_shape_mismatch` | Bronze / Stage A | projector 输出 shape 与 LLM hidden dim 或占位 token 数不一致 | 对照 patch tests 的 image-only、audio-only 和双模态 case |
| `mm_wer_denominator_wrong` | Stage C | audio reward 偏高或偏低，已知 WER 样例不匹配 | 检查 denominator 是否为 reference 词数 |
| `mm_clip_norm_missing` | Stage C | image reward 被 embedding norm 主导 | 检查 cosine 前是否分别归一化 image/text embedding |
| `mm_serving_ttft_unattributed` | Stage B | TTFT 变慢但无法定位阶段 | 拆分 queue、tokenize、encoder、prefill 和 decode 首步 |
| `mm_evidence_index_gap` | Capstone | final artifacts 中某个结论缺 run 证据 | 检查 evidence_index、command、config、metrics 和 report path |
