"""
L27 Patch · Prometheus Metrics Exporter

填空规则：
- TODO(student) 必须自己写
- 不许 import prometheus_client
- 允许 stdlib

完成度自检：
    make patch-test M=l26_sglang_pd_observability
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple


def _label_key(labels: Optional[Dict[str, str]]) -> Tuple[Tuple[str, str], ...]:
    if not labels:
        return ()
    return tuple(sorted(labels.items()))


def _format_labels(labels: Optional[Dict[str, str]]) -> str:
    if not labels:
        return ""
    parts = [f'{k}="{v}"' for k, v in sorted(labels.items())]
    return "{" + ",".join(parts) + "}"


class MetricsExporter:
    def __init__(self) -> None:
        # gauges[(name, label_key)] = value
        self.gauges: Dict[Tuple[str, Tuple], float] = {}
        # counters[(name, label_key)] = value
        self.counters: Dict[Tuple[str, Tuple], float] = {}
        # ratios[name] = (hits, misses)
        self.ratios: Dict[str, Tuple[int, int]] = {}

    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        # TODO(student): self.gauges[(name, _label_key(labels))] = float(value)
        raise NotImplementedError("L27: implement set_gauge")

    def inc_counter(
        self, name: str, by: float = 1, labels: Optional[Dict[str, str]] = None
    ) -> None:
        # TODO(student):
        #   key = (name, _label_key(labels))
        #   self.counters[key] = self.counters.get(key, 0.0) + by
        raise NotImplementedError("L27: implement inc_counter")

    def record_event(self, name: str, hit: bool) -> None:
        # TODO(student):
        #   h, m = self.ratios.get(name, (0, 0))
        #   if hit: h += 1
        #   else:   m += 1
        #   self.ratios[name] = (h, m)
        raise NotImplementedError("L27: implement record_event")

    def export(self) -> str:
        """返回 Prometheus exposition 文本。

        格式（按 metric 名分组 → label 字母序输出）：
            # TYPE <name> gauge|counter
            <name>[{labels}] <value>
        """
        lines = []

        # Gauges
        # TODO(student):
        #   按 metric_name 分组 self.gauges → 每组先出一行 # TYPE <name> gauge
        #   再每个 (name, label_key) 输出 "<name>{labels} <value>"
        #   labels 用 _format_labels() 还原
        #
        # Counters: 同上，TYPE counter
        #
        # Ratios: 每个 ratio 一行 # TYPE <name>_hit_rate gauge
        #         然后 "<name>_hit_rate <value>"，value = hits / (hits + misses)，
        #         分母为 0 时输出 0.0
        #
        # 最后 return "\n".join(lines) + "\n"
        raise NotImplementedError("L27: implement export")
