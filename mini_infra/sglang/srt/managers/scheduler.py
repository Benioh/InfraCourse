from __future__ import annotations

from dataclasses import dataclass

from mini_infra.sglang.srt.mem_cache.radix_cache import (
    MatchPrefixParams,
    RadixCache,
    RadixKey,
)


@dataclass
class Req:
    request_id: str
    prompt_tokens: list[str]
    output_tokens: list[str]


class Scheduler:
    """Tiny SGLang scheduler: route requests through prefix cache before decode."""

    def __init__(self) -> None:
        self.prefix_cache = RadixCache()
        self.waiting: list[Req] = []
        self.running: list[Req] = []

    def add_request(self, req: Req) -> None:
        self.waiting.append(req)

    def run_batch(self) -> dict:
        cache_hits = 0
        prefill = []
        decode = []
        while self.waiting:
            req = self.waiting.pop(0)
            result = self.prefix_cache.match_prefix(
                MatchPrefixParams(RadixKey(req.prompt_tokens))
            )
            cache_hits += int(result.value is not None)
            prefill.append(
                {
                    "request_id": req.request_id,
                    "matched_prefix_tokens": result.matched_prefix_len,
                }
            )
            req.output_tokens.append("token")
            self.prefix_cache.cache_finished_req(req)
            decode.append(req.request_id)
            self.running.append(req)
        return {"prefill": prefill, "decode": decode, "cache_hits": cache_hits}
