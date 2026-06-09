"""L41 Capstone Patch tests · CPU OK."""

from __future__ import annotations

import importlib
import math
import os
import sys
from pathlib import Path

import pytest
import torch

PATCH_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PATCH_DIR))


def _impl():
    name = os.environ.get("IMPL", "starter")
    return importlib.import_module(f"{name}.mm_omni")


# --------------- Stage A: Projector ---------------


def test_projector_image_only():
    impl = _impl()
    proj = impl.MultimodalProjector(vit_dim=512, whisper_dim=384, llm_dim=896, num_image_tokens=49)
    x = torch.randn(2, 49, 512)
    out = proj(image_features=x)
    assert out["image_emb"].shape == (2, 49, 896)
    assert out["audio_emb"] is None


def test_projector_audio_only():
    impl = _impl()
    proj = impl.MultimodalProjector(vit_dim=512, whisper_dim=384, llm_dim=896, num_image_tokens=49)
    x = torch.randn(2, 30, 384)  # variable-length audio
    out = proj(audio_features=x)
    assert out["audio_emb"].shape == (2, 30, 896)
    assert out["image_emb"] is None


def test_projector_both():
    impl = _impl()
    proj = impl.MultimodalProjector(vit_dim=512, whisper_dim=384, llm_dim=896, num_image_tokens=49)
    img = torch.randn(2, 49, 512)
    aud = torch.randn(2, 30, 384)
    out = proj(image_features=img, audio_features=aud)
    assert out["image_emb"].shape == (2, 49, 896)
    assert out["audio_emb"].shape == (2, 30, 896)


# --------------- Stage C: WER ---------------


def test_wer_perfect_match():
    impl = _impl()
    assert impl.wer_score("hello world", "hello world") == pytest.approx(0.0)


def test_wer_known_value():
    impl = _impl()
    # "hello world" vs "hello there" → 1 substitution / 2 reference words = 0.5
    assert impl.wer_score("hello world", "hello there") == pytest.approx(0.5)
    # "the cat" vs "a dog" → 2 subs / 2 words = 1.0
    assert impl.wer_score("the cat", "a dog") == pytest.approx(1.0)
    # "" vs "hello" → 1 insertion / 1 word = 1.0
    assert impl.wer_score("", "hello") == pytest.approx(1.0)


# --------------- Stage C: CLIP score ---------------


def test_clip_score_identical():
    impl = _impl()
    v = torch.randn(512)
    s = impl.clip_score(v, v)
    assert s == pytest.approx(1.0, abs=1e-5)


def test_clip_score_orthogonal():
    impl = _impl()
    a = torch.zeros(8)
    b = torch.zeros(8)
    a[0] = 1.0
    b[1] = 1.0
    s = impl.clip_score(a, b)
    assert s == pytest.approx(0.0, abs=1e-5)
