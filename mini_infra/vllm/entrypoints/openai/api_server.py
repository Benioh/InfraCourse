from __future__ import annotations

from dataclasses import asdict

from mini_infra.vllm.v1.engine.llm_engine import LLMEngine


class OpenAIServingChat:
    """Small OpenAI-compatible facade over Mini LLMEngine."""

    def __init__(self, engine: LLMEngine | None = None, model: str = "mini-infra-tiny-lm") -> None:
        self.engine = engine or LLMEngine()
        self.model = model

    def create_chat_completion(
        self, request_id: str, messages: list[dict[str, str]], max_tokens: int = 4
    ) -> dict:
        prompt = "\n".join(message.get("content", "") for message in messages)
        self.engine.add_request(request_id, prompt, max_tokens=max_tokens)
        outputs = []
        while self.engine.has_unfinished_requests():
            outputs = self.engine.step()
        output = next(item for item in outputs if item.request_id == request_id)
        return {
            "id": request_id,
            "object": "chat.completion",
            "model": self.model,
            "choices": [
                {
                    "message": {"role": "assistant", "content": output.text},
                    "finish_reason": "stop",
                }
            ],
            "mini_vllm_debug": asdict(output),
        }
