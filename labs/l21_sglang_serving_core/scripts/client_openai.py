from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--prompt", default="Explain TTFT in one sentence.")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": args.prompt}],
        "max_tokens": args.max_new_tokens,
        "temperature": 0,
    }
    request = urllib.request.Request(
        f"{args.base_url}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            print(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        print(json.dumps({"error": str(exc), "status": "server_unreachable"}, indent=2))


if __name__ == "__main__":
    main()
