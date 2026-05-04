from __future__ import annotations
import argparse
import json
import urllib.request

parser = argparse.ArgumentParser(description="vLLM OpenAI-compatible 客户端")
parser.add_argument("--base-url", default="http://127.0.0.1:31000/v1")
parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
parser.add_argument("--prompt", default="用一句话解释 TTFT")
args = parser.parse_args()
payload = {
    "model": args.model,
    "messages": [{"role": "user", "content": args.prompt}],
    "max_tokens": 64,
}
req = urllib.request.Request(
    f"{args.base_url}/chat/completions",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
)
print(urllib.request.urlopen(req, timeout=60).read().decode()[:1000])
