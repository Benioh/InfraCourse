"""
Acceptance rate 滑动窗口跟踪（L25 教学）。

教学目的：
    投机解码的核心指标是 acceptance rate（draft 候选被 target 接受的比例）。
    我们用滑动窗口避免长期均值滞后，同时给出期望加速公式：
        speedup ≈ 1 / ((1 - rate) + draft_cost_ratio)
    让学员理解：
        - rate 必须高于一个临界值（draft_cost 越大临界越高）才有正收益
        - 窗口太短 rate 抖动剧烈；太长又跟不上 workload 变化

真实框架对照：
    - github_repo/vllm/vllm/spec_decode/metrics.py
        生产 metrics 还按 request 分组、跨 worker 聚合；本文件只做单机
        token-level 滑动窗口。
"""
from __future__ import annotations

from collections import deque


class AcceptanceTracker:
    """滑动窗口的 acceptance tracker。

    窗口大小 32 是教学默认；生产里 vLLM 用 1000+ 以稳定。窗口太小会
    让 dynamic spec 关闭/开启反复抖动。
    """
    def __init__(self, window: int = 32) -> None:
        self.window = window
        self.events: deque[tuple[int, int]] = deque(maxlen=window)

    def add(self, accepted: int, proposed: int) -> None:
        self.events.append((accepted, proposed))

    def rate(self) -> float:
        accepted = sum(item[0] for item in self.events)
        proposed = sum(item[1] for item in self.events)
        return round(accepted / max(proposed, 1), 6)

    def expected_speedup(self, draft_cost_ratio: float = 0.25) -> float:
        rate = self.rate()
        return round(1.0 / max((1 - rate) + draft_cost_ratio, 1e-6), 3)


def acceptance_summary(
    accepted: list[int] | None = None, proposed: int = 4
) -> dict[str, float | int]:
    tracker = AcceptanceTracker(window=8)
    for value in accepted or [3, 2, 4, 1, 3, 0, 2, 3]:
        tracker.add(value, proposed)
    return {
        "window": tracker.window,
        "acceptance_rate": tracker.rate(),
        "speedup": tracker.expected_speedup(),
        "proposed_per_step": proposed,
    }
