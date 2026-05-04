from __future__ import annotations

import argparse
import json
from typing import Any


def prefix_key(prompt: str, prefix_tokens: int) -> str:
    return " ".join(prompt.split()[:prefix_tokens])


def simulate(prompts: list[str], prefix_tokens: int) -> dict[str, Any]:
    cache = set()
    hits = 0
    events = []
    for prompt in prompts:
        key = prefix_key(prompt, prefix_tokens)
        hit = key in cache
        hits += int(hit)
        cache.add(key)
        events.append({"prompt": prompt, "key": key, "hit": hit})
    return {
        "requests": len(prompts),
        "hits": hits,
        "cache_hit_rate": round(hits / max(len(prompts), 1), 4),
        "events": events,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="MiniInfra prefix cache simulation")
    parser.add_argument("--prefix-tokens", type=int, default=4)
    args = parser.parse_args()
    prompts = [
        "system you are helpful math tutor question one",
        "system you are helpful math tutor question two",
        "system you are terse code assistant question three",
    ]
    print(
        json.dumps(simulate(prompts, args.prefix_tokens), ensure_ascii=False, indent=2)
    )


if __name__ == "__main__":
    main()
