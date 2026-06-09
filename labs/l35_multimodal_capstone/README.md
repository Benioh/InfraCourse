# L41 · MM-Tiny-Omni Capstone：Projector、Reward 与 Evidence Package

<!-- LECTURE_FIRST_START -->

L41 是本轮课程的收口。它把多模态 projector、WER/CLIP reward、训练/服务/RL 证据和最终交付包放到同一个 Capstone 里验收。Bronze 验收 CPU patch，Silver/Gold 验收三阶段项目证据。

## 学习路线

1. 读 [system_map.md](system_map.md)：确认 L41 的三阶段边界和证据链。
2. 读 [lecture.md](lecture.md)：从 Bronze 组件、Stage A/B/C 到 final artifacts 聚合。
3. 读 [source_walkthrough.md](source_walkthrough.md)：按 patch、测试、聚合器和 delivery helper 读源码。
4. 读 [CAPSTONE_PLAN.md](CAPSTONE_PLAN.md)：了解 Silver/Gold 的项目交付要求。
5. 做 quiz：确认 projector shape、WER、CLIP score、服务化和证据边界。
6. 做 patch：实现 `MultimodalProjector`、`wer_score` 和 `clip_score`。
7. 跑聚合器：生成 final artifacts，检查 evidence index 和 risk register。
8. 使用 [outputs/data_pipeline_template.md](outputs/data_pipeline_template.md) 记录交付复盘。

## 本讲定位

| 问题 | 本讲回答 |
|---|---|
| 所属主线 | 第 5 章：Capstone |
| 解决什么问题 | 多模态模型从组件、训练、服务、RL 到交付证据的闭环 |
| 连接哪些源码 | `patch/reference/mm_omni.py`, `patch/tests/test_patch.py`, `scripts/run_capstone_aggregator.py`, `mini_infra/reports/build_delivery.py` |
| lab 检验什么 | Bronze：projector、WER、CLIP score 的形状和数值合同；Capstone：final artifacts 证据聚合 |

## Patch 闭环

```bash
cat labs/l35_multimodal_capstone/patch/task.md
$EDITOR labs/l35_multimodal_capstone/patch/starter/mm_omni.py
IMPL=reference make patch-test M=l35_multimodal_capstone
```

聚合器：

```bash
python labs/l35_multimodal_capstone/scripts/run_capstone_aggregator.py --run-id l41_validation
```

## 课后产物

| 产物 | 用途 |
|---|---|
| [outputs/debug_checklist.md](outputs/debug_checklist.md) | 排查 projector、reward、serving、RL 和证据缺口 |
| [outputs/source_reading_card.md](outputs/source_reading_card.md) | 复习 Bronze patch 与聚合器源码主路径 |
| [outputs/data_pipeline_template.md](outputs/data_pipeline_template.md) | 记录 Capstone 三阶段交付证据 |

<!-- LECTURE_FIRST_END -->
