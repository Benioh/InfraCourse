from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from mini_infra.vllm.v1.core.kv_cache_manager import KVCacheManager


@dataclass
class Request:
    request_id: str
    prompt: str
    max_tokens: int
    prompt_token_ids: list[str] = field(default_factory=list)
    output_tokens: list[str] = field(default_factory=list)
    kv_blocks: list[int] = field(default_factory=list)


@dataclass
class SchedulerOutput:
    scheduled_prefill: list[str]
    scheduled_decode: list[str]
    finished: list[str]


class Scheduler:
    """Minimal vLLM-like scheduler: waiting -> running -> decode -> finished."""

    def __init__(
        self, kv_cache_manager: KVCacheManager, max_num_running_reqs: int = 4
    ) -> None:
        self.kv_cache_manager = kv_cache_manager
        self.max_num_running_reqs = max_num_running_reqs
        self.waiting: deque[Request] = deque()
        self.running: dict[str, Request] = {}
        self.finished: dict[str, Request] = {}

    def add_request(self, request: Request) -> None:
        request.prompt_token_ids = request.prompt.split()
        self.waiting.append(request)

    def schedule(self) -> SchedulerOutput:
        scheduled_prefill = []
        scheduled_decode = []
        finished = []
        while self.waiting and len(self.running) < self.max_num_running_reqs:
            request = self.waiting[0]
            blocks_needed = max(1, (len(request.prompt_token_ids) + 15) // 16)
            try:
                request.kv_blocks = self.kv_cache_manager.allocate_slots(
                    request.request_id, blocks_needed
                )
            except RuntimeError:
                break
            self.waiting.popleft()
            self.running[request.request_id] = request
            scheduled_prefill.append(request.request_id)
        for request_id, request in list(self.running.items()):
            if len(request.output_tokens) >= request.max_tokens:
                self.finished[request_id] = request
                self.kv_cache_manager.free(request_id)
                del self.running[request_id]
                finished.append(request_id)
            else:
                scheduled_decode.append(request_id)
        return SchedulerOutput(scheduled_prefill, scheduled_decode, finished)
