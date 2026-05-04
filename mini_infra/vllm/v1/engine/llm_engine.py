from __future__ import annotations

from dataclasses import dataclass

from mini_infra.vllm.v1.core.sched.scheduler import Request, Scheduler
from mini_infra.vllm.v1.kv_cache.kv_cache_manager import KVCacheManager


@dataclass
class RequestOutput:
    request_id: str
    text: str
    finished: bool
    kv_blocks: list[int]


class LLMEngine:
    """Tiny vLLM LLMEngine: add_request + step + RequestOutput."""

    def __init__(self, max_num_running_reqs: int = 4, num_kv_blocks: int = 16) -> None:
        self.kv_cache_manager = KVCacheManager(num_blocks=num_kv_blocks)
        self.scheduler = Scheduler(self.kv_cache_manager, max_num_running_reqs=max_num_running_reqs)
        self.outputs: dict[str, RequestOutput] = {}

    def add_request(self, request_id: str, prompt: str, max_tokens: int = 4) -> None:
        self.scheduler.add_request(
            Request(request_id=request_id, prompt=prompt, max_tokens=max_tokens)
        )

    def step(self) -> list[RequestOutput]:
        scheduled = self.scheduler.schedule()
        for request_id in scheduled.scheduled_decode:
            request = self.scheduler.running.get(request_id)
            if request is None:
                continue
            request.output_tokens.append(self._next_token(request))
            self.outputs[request_id] = RequestOutput(
                request_id,
                " ".join(request.output_tokens),
                False,
                list(request.kv_blocks),
            )
        for request_id in scheduled.finished:
            request = self.scheduler.finished[request_id]
            self.outputs[request_id] = RequestOutput(
                request_id,
                " ".join(request.output_tokens),
                True,
                list(request.kv_blocks),
            )
        return list(self.outputs.values())

    def has_unfinished_requests(self) -> bool:
        return bool(self.scheduler.waiting or self.scheduler.running)

    def _next_token(self, request: Request) -> str:
        if "apple" in request.prompt.lower():
            answer = ["Final", "answer:", "7"]
        else:
            answer = ["MiniInfra", "token"]
        return answer[min(len(request.output_tokens), len(answer) - 1)]
