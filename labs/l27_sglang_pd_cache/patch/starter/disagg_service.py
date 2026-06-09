"""L28 Patch · SGLang-shaped prefill/decode disaggregation service."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Worker:
    worker_id: str
    role: str
    load_tokens: int = 0
    active_requests: int = 0


@dataclass
class KVTransfer:
    request_id: str
    prefill_worker: str
    decode_worker: str
    token_count: int


@dataclass
class DisaggRoute:
    request_id: str
    prefill_worker: str
    decode_worker: str
    prompt_tokens: int
    cached_prefix_tokens: int
    prefill_tokens: int
    kv_transfer_tokens: int


class DisaggregationService:
    """Tiny prefill/decode disaggregation service with explicit KV transfer."""

    def __init__(
        self,
        prefill_workers: list[str] | None = None,
        decode_workers: list[str] | None = None,
    ) -> None:
        self.prefill_workers = [
            Worker(worker_id, "prefill")
            for worker_id in (prefill_workers or ["prefill-0"])
        ]
        self.decode_workers = [
            Worker(worker_id, "decode")
            for worker_id in (decode_workers or ["decode-0"])
        ]
        self.transfers: list[KVTransfer] = []
        self.routes: dict[str, DisaggRoute] = {}

    def route_request(
        self,
        request_id: str,
        prompt_token_count: int,
        cached_prefix_tokens: int = 0,
    ) -> DisaggRoute:
        """Assign one request to prefill/decode workers and record the KV handoff."""
        # TODO(student): validate request_id and token counts.
        # TODO(student): uncached prompt tokens are the prefill work:
        #   prefill_tokens = prompt_token_count - cached_prefix_tokens.
        # TODO(student): choose least-loaded prefill and decode workers.
        # TODO(student): update worker loads / active request counters.
        # TODO(student): call transfer_kv(..., token_count=prompt_token_count).
        # TODO(student): store and return a DisaggRoute.
        raise NotImplementedError("L28: implement route_request")

    def transfer_kv(
        self,
        request_id: str,
        prefill_worker: str,
        decode_worker: str,
        token_count: int,
    ) -> KVTransfer:
        # TODO(student): validate request_id/workers and positive token_count.
        # TODO(student): append a KVTransfer and return it.
        raise NotImplementedError("L28: implement transfer_kv")

    def metrics(self) -> dict:
        # TODO(student): return transfer count, transferred tokens, cached-prefix tokens,
        # prefill tokens, active requests, tokens by worker, and current worker loads.
        raise NotImplementedError("L28: implement metrics")

    def transfers_for_request(self, request_id: str) -> list[KVTransfer]:
        # TODO(student): return transfers matching request_id in insertion order.
        raise NotImplementedError("L28: implement transfers_for_request")

    def complete_request(self, request_id: str) -> None:
        # TODO(student): remove the active route and subtract its worker loads.
        # Unknown request_id should be a no-op, matching idempotent completion paths.
        raise NotImplementedError("L28: implement complete_request")
