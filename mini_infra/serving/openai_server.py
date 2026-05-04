from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


def complete(prompt: str, max_tokens: int = 64) -> str:
    if "apple" in prompt.lower() or "apples" in prompt.lower():
        return "Final answer: 7"
    if "birds" in prompt.lower():
        return "Final answer: 6"
    return f"MiniInfra local response with at most {max_tokens} tokens."


class MiniOpenAIHandler(BaseHTTPRequestHandler):
    model_name = "mini-infra-tiny-lm"

    def _send(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/v1/models":
            self._send(
                200,
                {
                    "object": "list",
                    "data": [{"id": self.model_name, "object": "model"}],
                },
            )
            return
        self._send(404, {"error": "not found"})

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
        if self.path != "/v1/chat/completions":
            self._send(404, {"error": "not found"})
            return
        messages = payload.get("messages", [])
        prompt = "\n".join(str(message.get("content", "")) for message in messages)
        text = complete(prompt, int(payload.get("max_tokens", 64)))
        self._send(
            200,
            {
                "id": "mini-infra-chatcmpl",
                "object": "chat.completion",
                "model": payload.get("model", self.model_name),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": len(prompt.split()),
                    "completion_tokens": len(text.split()),
                },
            },
        )

    def log_message(self, format: str, *args: Any) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MiniInfra OpenAI-compatible local server"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=31888)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), MiniOpenAIHandler)
    print(f"MiniInfra server listening on http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
