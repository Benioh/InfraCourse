"""Reference solution for L06 Patch · multimodal collator."""

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
    B = len(batch)

    per_item_ids: List[List[int]] = []
    per_item_modal: List[List[int]] = []
    for item in batch:
        ids: List[int] = []
        modal: List[int] = []
        if item.get("image") is not None:
            ids.extend([IMAGE_TOKEN_ID] * image_num_patches)
            modal.extend([MODAL_IMAGE] * image_num_patches)
        if item.get("audio_mel") is not None:
            audio_t = min(int(item["audio_mel"].shape[0]), max_audio_frames)
            ids.extend([AUDIO_TOKEN_ID] * audio_t)
            modal.extend([MODAL_AUDIO] * audio_t)
        ids.extend(item["text_ids"])
        modal.extend([MODAL_TEXT] * len(item["text_ids"]))
        per_item_ids.append(ids)
        per_item_modal.append(modal)

    S = max(len(x) for x in per_item_ids)
    input_ids = torch.full((B, S), PAD_TOKEN_ID, dtype=torch.long)
    attention_mask = torch.zeros(B, S, dtype=torch.bool)
    modal_type_ids = torch.zeros(B, S, dtype=torch.int8)
    for i, (ids, modal) in enumerate(zip(per_item_ids, per_item_modal)):
        L = len(ids)
        input_ids[i, :L] = torch.tensor(ids, dtype=torch.long)
        attention_mask[i, :L] = True
        modal_type_ids[i, :L] = torch.tensor(modal, dtype=torch.int8)

    # Image collation
    pixel_values: Optional[torch.Tensor] = None
    image_batch_indices: Optional[torch.Tensor] = None
    image_list, image_idx = [], []
    for i, item in enumerate(batch):
        if item.get("image") is not None:
            image_list.append(item["image"])
            image_idx.append(i)
    if image_list:
        pixel_values = torch.stack(image_list, dim=0)
        image_batch_indices = torch.tensor(image_idx, dtype=torch.long)

    # Audio collation
    audio_features: Optional[torch.Tensor] = None
    audio_mask: Optional[torch.Tensor] = None
    audio_batch_indices: Optional[torch.Tensor] = None
    audio_list_padded, audio_mask_list, audio_idx = [], [], []
    for i, item in enumerate(batch):
        mel = item.get("audio_mel")
        if mel is None:
            continue
        T = mel.shape[0]
        T_eff = min(T, max_audio_frames)
        truncated = mel[:T_eff]  # (T_eff, 80)
        # pad along seq dim to max_audio_frames
        pad_amount = max_audio_frames - T_eff
        if pad_amount > 0:
            padded = F.pad(truncated, (0, 0, 0, pad_amount), value=0.0)
        else:
            padded = truncated
        # mask
        m = torch.zeros(max_audio_frames, dtype=torch.bool)
        m[:T_eff] = True
        audio_list_padded.append(padded)
        audio_mask_list.append(m)
        audio_idx.append(i)
    if audio_list_padded:
        audio_features = torch.stack(audio_list_padded, dim=0)
        audio_mask = torch.stack(audio_mask_list, dim=0)
        audio_batch_indices = torch.tensor(audio_idx, dtype=torch.long)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "modal_type_ids": modal_type_ids,
        "pixel_values": pixel_values,
        "image_batch_indices": image_batch_indices,
        "audio_features": audio_features,
        "audio_mask": audio_mask,
        "audio_batch_indices": audio_batch_indices,
    }
