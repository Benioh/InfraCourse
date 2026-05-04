"""
L06 Patch · 多模态 collator (image + audio + text)

填空规则：
- TODO(student) 必须自己写
- 不许 import transformers / torchvision.transforms
- 允许 torch.cat / F.pad / list 操作

完成度自检：
    make patch-test M=l17_megatron_multimodal_data
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F


PAD_TOKEN_ID = 0
IMAGE_TOKEN_ID = 1
AUDIO_TOKEN_ID = 2

MODAL_TEXT = 0
MODAL_IMAGE = 1
MODAL_AUDIO = 2


def multimodal_collate(
    batch: List[Dict[str, Any]],
    image_num_patches: int = 16,
    max_audio_frames: int = 64,
) -> Dict[str, Optional[torch.Tensor]]:
    """把变长 multimodal 样本打包成 batch tensor."""
    B = len(batch)

    # ============ STEP 1：每个样本构造 (input_ids, modal_type_ids) ============
    # 顺序：[image_tokens] + [audio_tokens] + [text_ids]
    # TODO(student):
    #   per_item_ids = []
    #   per_item_modal = []
    #   for item in batch:
    #       ids, modal = [], []
    #       if item.get("image") is not None:
    #           ids.extend([IMAGE_TOKEN_ID] * image_num_patches)
    #           modal.extend([MODAL_IMAGE] * image_num_patches)
    #       if item.get("audio_mel") is not None:
    #           # 音频 token 数 = 截断后 mel 帧数（不能超 max_audio_frames）
    #           audio_t = min(item["audio_mel"].shape[0], max_audio_frames)
    #           ids.extend([AUDIO_TOKEN_ID] * audio_t)
    #           modal.extend([MODAL_AUDIO] * audio_t)
    #       ids.extend(item["text_ids"])
    #       modal.extend([MODAL_TEXT] * len(item["text_ids"]))
    #       per_item_ids.append(ids)
    #       per_item_modal.append(modal)

    # ============ STEP 2：pad 到 batch 内最大长度 S ============
    # TODO(student):
    #   S = max(len(ids) for ids in per_item_ids)
    #   input_ids = torch.full((B, S), PAD_TOKEN_ID, dtype=torch.long)
    #   attention_mask = torch.zeros(B, S, dtype=torch.bool)
    #   modal_type_ids = torch.zeros(B, S, dtype=torch.int8)  # MODAL_TEXT=0
    #   for i, (ids, modal) in enumerate(zip(per_item_ids, per_item_modal)):
    #       L = len(ids)
    #       input_ids[i, :L] = torch.tensor(ids, dtype=torch.long)
    #       attention_mask[i, :L] = True
    #       modal_type_ids[i, :L] = torch.tensor(modal, dtype=torch.int8)

    # ============ STEP 3：收集 image / audio tensor ============
    # 只收集有图 / 有音频的 batch idx；shapes 与对应 batch_indices 一一对应。
    # TODO(student):
    #   pixel_values = None
    #   image_batch_indices = None
    #   image_list, image_idx_list = [], []
    #   for i, item in enumerate(batch):
    #       if item.get("image") is not None:
    #           image_list.append(item["image"])
    #           image_idx_list.append(i)
    #   if image_list:
    #       pixel_values = torch.stack(image_list, dim=0)
    #       image_batch_indices = torch.tensor(image_idx_list, dtype=torch.long)

    # TODO(student): 类似地收集 audio——
    #   每个 mel 截断到 max_audio_frames 行后再 pad 到 max_audio_frames（用 F.pad）；
    #   audio_mask 在前 truncated_T 行 = True。
    #   audio_features = torch.stack(...)  # (B'', max_audio_frames, 80)
    #   audio_mask = torch.stack(...)      # (B'', max_audio_frames)
    #   audio_batch_indices = torch.tensor(...)

    # ============ STEP 4：返回 ============
    # return {
    #     "input_ids": input_ids,
    #     "attention_mask": attention_mask,
    #     "modal_type_ids": modal_type_ids,
    #     "pixel_values": pixel_values,
    #     "image_batch_indices": image_batch_indices,
    #     "audio_features": audio_features,
    #     "audio_mask": audio_mask,
    #     "audio_batch_indices": audio_batch_indices,
    # }
    raise NotImplementedError("L06 Patch: implement multimodal_collate")
