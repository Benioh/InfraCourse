"""L18 Patch tests · CPU only."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pytest
import torch

PATCH_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PATCH_DIR))


def _impl():
    name = os.environ.get("IMPL", "starter")
    return importlib.import_module(f"{name}.multimodal_collate")


def _make_text(ids):
    return {"text_ids": ids, "image": None, "audio_mel": None}


def _make_text_image(ids, c=3, h=8, w=8):
    return {"text_ids": ids, "image": torch.randn(c, h, w), "audio_mel": None}


def _make_text_audio(ids, T):
    return {"text_ids": ids, "image": None, "audio_mel": torch.randn(T, 80)}


def _make_text_image_audio(ids, T, c=3, h=8, w=8):
    return {"text_ids": ids, "image": torch.randn(c, h, w), "audio_mel": torch.randn(T, 80)}


def test_text_only_batch():
    impl = _impl()
    batch = [_make_text([10, 11, 12]), _make_text([20, 21])]
    out = impl.multimodal_collate(batch)

    assert out["input_ids"].shape == (2, 3)
    assert out["attention_mask"].dtype == torch.bool
    assert out["pixel_values"] is None
    assert out["image_batch_indices"] is None
    assert out["audio_features"] is None
    # text-only batch has all modal_type=MODAL_TEXT (0)
    assert torch.all(out["modal_type_ids"] == 0)


def test_with_image_batch():
    impl = _impl()
    batch = [_make_text_image([10, 11], c=3, h=4, w=4)]
    out = impl.multimodal_collate(batch, image_num_patches=16)

    # Expected sequence: [IMG] x 16 + [10, 11] = 18 tokens
    assert out["input_ids"].shape == (1, 18)
    # First 16 positions are IMAGE_TOKEN_ID
    assert torch.all(out["input_ids"][0, :16] == impl.IMAGE_TOKEN_ID)
    # modal_type_ids in first 16 = MODAL_IMAGE (1)
    assert torch.all(out["modal_type_ids"][0, :16] == 1)
    # Last 2 positions = text
    assert torch.all(out["modal_type_ids"][0, 16:] == 0)
    # pixel_values present
    assert out["pixel_values"] is not None
    assert out["pixel_values"].shape == (1, 3, 4, 4)
    assert out["image_batch_indices"].tolist() == [0]


def test_with_audio_batch():
    impl = _impl()
    T = 10
    batch = [_make_text_audio([5, 6], T=T)]
    out = impl.multimodal_collate(batch, max_audio_frames=64)

    # Expected sequence: [AUDIO] x 10 + [5, 6] = 12 tokens
    assert out["input_ids"].shape == (1, 12)
    assert torch.all(out["input_ids"][0, :T] == impl.AUDIO_TOKEN_ID)
    assert torch.all(out["modal_type_ids"][0, :T] == 2)  # MODAL_AUDIO
    # Audio padded to max_audio_frames
    assert out["audio_features"].shape == (1, 64, 80)
    assert out["audio_mask"][0, :T].all()
    assert not out["audio_mask"][0, T:].any()
    assert out["audio_batch_indices"].tolist() == [0]


def test_mixed_modality_batch():
    impl = _impl()
    batch = [
        _make_text([1, 2, 3]),                     # text only
        _make_text_image([4, 5], c=3, h=4, w=4),    # text + image
        _make_text_audio([6], T=8),                 # text + audio
        _make_text_image_audio([7, 8, 9], T=12, c=3, h=4, w=4),  # all three
    ]
    out = impl.multimodal_collate(batch, image_num_patches=4, max_audio_frames=32)

    # Sample 0 length = 3 (text only)
    # Sample 1 length = 4 (image) + 2 (text) = 6
    # Sample 2 length = 8 (audio) + 1 (text) = 9
    # Sample 3 length = 4 (image) + 12 (audio) + 3 (text) = 19
    # Padded S = 19
    assert out["input_ids"].shape[0] == 4
    assert out["input_ids"].shape[1] == 19

    # Two items have image: index 1 and 3
    assert out["image_batch_indices"].tolist() == [1, 3]
    assert out["pixel_values"].shape == (2, 3, 4, 4)

    # Two items have audio: index 2 and 3
    assert out["audio_batch_indices"].tolist() == [2, 3]
    assert out["audio_features"].shape == (2, 32, 80)


def test_padding_blocks_attention():
    impl = _impl()
    batch = [_make_text([1, 2]), _make_text([3, 4, 5, 6])]
    out = impl.multimodal_collate(batch)

    # Sample 0 padded from 2 → 4
    assert out["attention_mask"][0, :2].all()
    assert not out["attention_mask"][0, 2:].any()
    # Padding tokens should be PAD_TOKEN_ID
    assert torch.all(out["input_ids"][0, 2:] == impl.PAD_TOKEN_ID)
    # Modal type for padding = MODAL_TEXT (0) (unchanged from default)
    assert torch.all(out["modal_type_ids"][0, 2:] == 0)


def test_audio_truncation():
    impl = _impl()
    # text ids must be >= 3 (0=PAD, 1=IMAGE, 2=AUDIO are reserved)
    batch = [_make_text_audio([10, 11], T=200)]
    out = impl.multimodal_collate(batch, max_audio_frames=64)

    # Audio tokens in input_ids should be exactly 64 (truncated)
    audio_token_count = int((out["input_ids"][0] == impl.AUDIO_TOKEN_ID).sum().item())
    assert audio_token_count == 64

    # audio_features pad to 64; audio_mask all True (since truncation = full)
    assert out["audio_features"].shape == (1, 64, 80)
    assert out["audio_mask"].all()


def test_image_batch_indices_correct():
    impl = _impl()
    # text ids >= 3 (avoid collision with IMAGE_TOKEN_ID=1 / AUDIO_TOKEN_ID=2)
    batch = [
        _make_text_image([10], c=3, h=4, w=4),
        _make_text([20]),
        _make_text_image([3], c=3, h=4, w=4),
    ]
    out = impl.multimodal_collate(batch, image_num_patches=2)

    assert out["image_batch_indices"].tolist() == [0, 2]
    assert out["pixel_values"].shape == (2, 3, 4, 4)
    # The image at index 0 of pixel_values must equal batch[0]['image']
    assert torch.equal(out["pixel_values"][0], batch[0]["image"])
    assert torch.equal(out["pixel_values"][1], batch[2]["image"])
