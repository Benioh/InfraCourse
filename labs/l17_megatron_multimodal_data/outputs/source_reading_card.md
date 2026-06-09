# L18 Source Reading Card

## 主路径

1. `labs/l17_megatron_multimodal_data/patch/starter/multimodal_collate.py`：学生补齐的最小 batch 合同。
2. `labs/l17_megatron_multimodal_data/patch/reference/multimodal_collate.py`：reference 的序列构造、padding、side tensor 收集。
3. `labs/l17_megatron_multimodal_data/patch/tests/test_patch.py`：7 个行为边界。
4. `labs/l17_megatron_multimodal_data/scripts/build_manifest.py`：toy manifest 的三类任务样本。
5. `labs/l17_megatron_multimodal_data/scripts/build_webdataset_shards.py`：把 manifest 行打进 tar shard。
6. `labs/l17_megatron_multimodal_data/scripts/run_multimodal_data_lab.py`：串起下载、manifest、shard、validation、loader smoke。
7. `mini_infra/data/manifest.py`：MiniInfra manifest/schema check 的简化入口。
8. `github_repo/Megatron-LM/examples/multimodal/dataset_helpers.py`：Megatron multimodal TaskEncoder 的 sample/batch 合同。
9. `github_repo/Megatron-LM/examples/multimodal/dataloader_provider.py`：Energon dataloader、rank gating 和 dataloader state restore。

## 阅读方法

1. 先看 patch reference，确认 batch 合同的字段和 shape。
2. 再看 patch tests，理解每个测试守住哪条不变量。
3. 然后看 lab scripts，确认 manifest、shard 和 smoke artifact 怎么产生。
4. 最后看 Megatron 源码，把教学版的 `pixel_values`、tokens、indices 对应到真实 TaskEncoder 的 imgs、tokens、labels、num_tiles。

## 自检

- 我能否从一个 mixed batch 手算 `input_ids.shape`？
- 我能否解释 `modal_type_ids=0` 为什么同时覆盖文本和 padding？
- 我能否指出真实 Megatron 源码在哪一步 stack images、pad tokens、保存 dataloader state？
