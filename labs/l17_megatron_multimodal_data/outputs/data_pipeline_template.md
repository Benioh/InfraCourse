# L18 Data Pipeline 复盘模板

## Run 信息

- 日期：
- 命令：
- git commit：
- Python / PyTorch：
- 数据路径：
- batch size：
- `image_num_patches`：
- `max_audio_frames`：

## 预期

- 本次要验证的层：
- 输入样本组合：
- 成功标准：

## 关键张量

| 字段 | shape / 值 | 判断 |
|---|---|---|
| `input_ids` |  |  |
| `attention_mask.sum(dim=1)` |  |  |
| `modal_type_ids` 分布 |  |  |
| `pixel_values` |  |  |
| `image_batch_indices` |  |  |
| `audio_features` |  |  |
| `audio_mask.sum(dim=1)` |  |  |
| `audio_batch_indices` |  |  |

## Artifact

| artifact | 路径 | 结论 |
|---|---|---|
| manifest validation |  |  |
| shard inspection |  |  |
| loader smoke |  |  |
| metrics.jsonl |  |  |

## 源码对应

| 现象 | 源码位置 | 判断 |
|---|---|---|
|  |  |  |

## 结论

- 本次能证明什么：
- 还不能证明什么：
- 下一步要改的配置、代码或实验：
