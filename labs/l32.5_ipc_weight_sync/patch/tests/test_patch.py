"""L32.5 Patch tests · CPU OK."""

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
    return importlib.import_module(f"{name}.ipc_weight_sync")


def test_serialize_returns_handle_not_data():
    impl = _impl()
    pool = impl.IPCStoragePool()
    tensor = torch.randn(1024, 1024)  # 4 MB at fp32
    blob = impl.serialize_handle(tensor, pool)
    tensor_bytes = tensor.numel() * tensor.element_size()
    # blob should be metadata-sized, not data-sized.
    assert len(blob) < 1024, f"serialize_handle leaked data: {len(blob)} bytes for {tensor_bytes}-byte tensor"


def test_deserialize_shares_storage():
    impl = _impl()
    pool = impl.IPCStoragePool()
    tensor = torch.randn(8, 16)
    blob = impl.serialize_handle(tensor, pool)
    rebuilt = impl.deserialize_handle(blob, pool)
    # share storage = same data_ptr (since pool.get returns the same object)
    assert rebuilt.data_ptr() == tensor.data_ptr(), "deserialize must share storage with source"


def test_handle_round_trip_preserves_values():
    impl = _impl()
    pool = impl.IPCStoragePool()
    torch.manual_seed(0)
    tensor = torch.randn(32, 64)
    blob = impl.serialize_handle(tensor, pool)
    rebuilt = impl.deserialize_handle(blob, pool)
    assert torch.equal(rebuilt, tensor)


def test_gather_only_rank_0_has_full_list():
    impl = _impl()
    pool = impl.IPCStoragePool()
    group: dict = {}
    blobs = []
    for r in range(4):
        t = torch.randn(4, 4) + r
        b = impl.serialize_handle(t, pool)
        blobs.append(b)
        result = impl.gather_handles_to_rank0(b, rank=r, world_size=4, group=group)
        if r == 0:
            # rank 0 was first; group only has its own entry, must have all 4 by end
            pass
        else:
            assert result is None, f"rank {r} should return None"
    # Now call rank 0 again to gather (group is now full)
    final = impl.gather_handles_to_rank0(blobs[0], rank=0, world_size=4, group=group)
    assert final is not None
    assert len(final) == 4


def test_local_serialized_tensor_get_by_rank():
    impl = _impl()
    pool = impl.IPCStoragePool()
    tensors = [torch.full((3,), float(i)) for i in range(4)]
    blobs = [impl.serialize_handle(t, pool) for t in tensors]
    lst = impl.LocalSerializedTensor(values=blobs)
    for r in range(4):
        got = lst.get(r, pool)
        assert torch.equal(got, tensors[r])


def test_update_weights_replaces_inference_state():
    impl = _impl()
    pool = impl.IPCStoragePool()
    src_a = torch.randn(8, 16)
    src_b = torch.randn(4, 4)
    blob_a = impl.serialize_handle(src_a, pool)
    blob_b = impl.serialize_handle(src_b, pool)
    # Single TP rank: each LocalSerializedTensor has one value
    lst_a = impl.LocalSerializedTensor(values=[blob_a])
    lst_b = impl.LocalSerializedTensor(values=[blob_b])

    inference_state = {
        "layer.weight": torch.zeros(8, 16),
        "layer.bias": torch.zeros(4, 4),
    }
    impl.update_weights_from_tensor(
        named_handles=[("layer.weight", lst_a), ("layer.bias", lst_b)],
        inference_state=inference_state,
        tp_rank=0,
        pool=pool,
        flush_cache=False,
    )
    assert torch.equal(inference_state["layer.weight"], src_a)
    assert torch.equal(inference_state["layer.bias"], src_b)


def test_flush_cache_only_on_last_tensor():
    impl = _impl()
    pool = impl.IPCStoragePool()
    src_a = torch.randn(2, 2)
    src_b = torch.randn(2, 2)
    blob_a = impl.serialize_handle(src_a, pool)
    blob_b = impl.serialize_handle(src_b, pool)
    lst_a = impl.LocalSerializedTensor(values=[blob_a])
    lst_b = impl.LocalSerializedTensor(values=[blob_b])

    state: dict = {}
    # flush_cache=False: pool retains both entries
    impl.update_weights_from_tensor(
        [("a", lst_a), ("b", lst_b)],
        inference_state=state, tp_rank=0, pool=pool, flush_cache=False,
    )
    assert pool.size() == 2

    # flush_cache=True with two tensors: pool clears at end
    pool2 = impl.IPCStoragePool()
    blob_a2 = impl.serialize_handle(src_a, pool2)
    blob_b2 = impl.serialize_handle(src_b, pool2)
    lst_a2 = impl.LocalSerializedTensor(values=[blob_a2])
    lst_b2 = impl.LocalSerializedTensor(values=[blob_b2])
    state2: dict = {}
    impl.update_weights_from_tensor(
        [("a", lst_a2), ("b", lst_b2)],
        inference_state=state2, tp_rank=0, pool=pool2, flush_cache=True,
    )
    assert pool2.size() == 0, "pool must be cleared on last-tensor flush_cache"
