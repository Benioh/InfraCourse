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

    def _select_least_loaded(self, workers: list[Worker]) -> Worker:
        return min(
            workers,
            key=lambda worker: (
                worker.load_tokens,
                worker.active_requests,
                worker.worker_id,
            ),
        )

    def _worker_by_id(self, worker_id: str) -> Worker:
        for worker in [*self.prefill_workers, *self.decode_workers]:
            if worker.worker_id == worker_id:
                return worker
        raise ValueError(f"unknown worker: {worker_id}")

    def route_request(
        self,
        request_id: str,
        prompt_token_count: int,
        cached_prefix_tokens: int = 0,
    ) -> DisaggRoute:
        if not request_id:
            raise ValueError("request_id is required")
        if request_id in self.routes:
            raise ValueError(f"request already active: {request_id}")
        if prompt_token_count <= 0:
            raise ValueError("prompt_token_count must be positive")
        if cached_prefix_tokens < 0 or cached_prefix_tokens > prompt_token_count:
            raise ValueError("cached_prefix_tokens must be in [0, prompt_token_count]")

        prefill_tokens = prompt_token_count - cached_prefix_tokens
        prefill = self._select_least_loaded(self.prefill_workers)
        decode = self._select_least_loaded(self.decode_workers)

        prefill.load_tokens += prefill_tokens
        prefill.active_requests += 1
        decode.load_tokens += prompt_token_count
        decode.active_requests += 1
        self.transfer_kv(
            request_id,
            prefill.worker_id,
            decode.worker_id,
            token_count=prompt_token_count,
        )

        route = DisaggRoute(
            request_id=request_id,
            prefill_worker=prefill.worker_id,
            decode_worker=decode.worker_id,
            prompt_tokens=prompt_token_count,
            cached_prefix_tokens=cached_prefix_tokens,
            prefill_tokens=prefill_tokens,
            kv_transfer_tokens=prompt_token_count,
        )
        self.routes[request_id] = route
        return route

    def transfer_kv(
        self, request_id: str, prefill_worker: str, decode_worker: str, token_count: int
    ) -> KVTransfer:
        if not request_id:
            raise ValueError("request_id is required")
        if not prefill_worker or not decode_worker:
            raise ValueError("prefill_worker and decode_worker are required")
        if token_count <= 0:
            raise ValueError("token_count must be positive")
        transfer = KVTransfer(
            request_id, prefill_worker, decode_worker, int(token_count)
        )
        self.transfers.append(transfer)
        return transfer

    def metrics(self) -> dict:
        tokens_by_prefill_worker: dict[str, int] = {}
        tokens_by_decode_worker: dict[str, int] = {}
        for transfer in self.transfers:
            tokens_by_prefill_worker[transfer.prefill_worker] = (
                tokens_by_prefill_worker.get(transfer.prefill_worker, 0)
                + transfer.token_count
            )
            tokens_by_decode_worker[transfer.decode_worker] = (
                tokens_by_decode_worker.get(transfer.decode_worker, 0)
                + transfer.token_count
            )
        return {
            "kv_transfers": len(self.transfers),
            "tokens_transferred": sum(item.token_count for item in self.transfers),
            "prefill_tokens": sum(
                route.prefill_tokens for route in self.routes.values()
            ),
            "cached_prefix_tokens": sum(
                route.cached_prefix_tokens for route in self.routes.values()
            ),
            "active_requests": len(self.routes),
            "tokens_by_prefill_worker": tokens_by_prefill_worker,
            "tokens_by_decode_worker": tokens_by_decode_worker,
            "worker_loads": {
                worker.worker_id: worker.load_tokens
                for worker in [*self.prefill_workers, *self.decode_workers]
            },
        }

    def transfers_for_request(self, request_id: str) -> list[KVTransfer]:
        return [
            transfer for transfer in self.transfers if transfer.request_id == request_id
        ]

    def complete_request(self, request_id: str) -> None:
        route = self.routes.pop(request_id, None)
        if route is None:
            return
        prefill = self._worker_by_id(route.prefill_worker)
        decode = self._worker_by_id(route.decode_worker)
        prefill.load_tokens = max(0, prefill.load_tokens - route.prefill_tokens)
        decode.load_tokens = max(0, decode.load_tokens - route.kv_transfer_tokens)
        prefill.active_requests = max(0, prefill.active_requests - 1)
        decode.active_requests = max(0, decode.active_requests - 1)
