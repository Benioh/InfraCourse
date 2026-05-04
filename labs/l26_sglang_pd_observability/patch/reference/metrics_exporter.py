"""Reference solution for L09 Patch · MetricsExporter."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Optional, Tuple


def _label_key(labels: Optional[Dict[str, str]]) -> Tuple[Tuple[str, str], ...]:
    if not labels:
        return ()
    return tuple(sorted(labels.items()))


def _format_labels(labels_tuple: Tuple[Tuple[str, str], ...]) -> str:
    if not labels_tuple:
        return ""
    parts = [f'{k}="{v}"' for k, v in labels_tuple]
    return "{" + ",".join(parts) + "}"


class MetricsExporter:
    def __init__(self) -> None:
        self.gauges: Dict[Tuple[str, Tuple], float] = {}
        self.counters: Dict[Tuple[str, Tuple], float] = {}
        self.ratios: Dict[str, Tuple[int, int]] = {}

    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        self.gauges[(name, _label_key(labels))] = float(value)

    def inc_counter(
        self, name: str, by: float = 1, labels: Optional[Dict[str, str]] = None
    ) -> None:
        key = (name, _label_key(labels))
        self.counters[key] = self.counters.get(key, 0.0) + by

    def record_event(self, name: str, hit: bool) -> None:
        h, m = self.ratios.get(name, (0, 0))
        if hit:
            h += 1
        else:
            m += 1
        self.ratios[name] = (h, m)

    def export(self) -> str:
        lines = []

        gauges_by_name: Dict[str, list] = defaultdict(list)
        for (name, lk), v in self.gauges.items():
            gauges_by_name[name].append((lk, v))

        for name in sorted(gauges_by_name):
            lines.append(f"# TYPE {name} gauge")
            for lk, v in sorted(gauges_by_name[name]):
                lines.append(f"{name}{_format_labels(lk)} {v}")

        counters_by_name: Dict[str, list] = defaultdict(list)
        for (name, lk), v in self.counters.items():
            counters_by_name[name].append((lk, v))

        for name in sorted(counters_by_name):
            lines.append(f"# TYPE {name} counter")
            for lk, v in sorted(counters_by_name[name]):
                lines.append(f"{name}{_format_labels(lk)} {v}")

        for name in sorted(self.ratios):
            h, m = self.ratios[name]
            total = h + m
            rate = h / total if total > 0 else 0.0
            metric_name = f"{name}_hit_rate"
            lines.append(f"# TYPE {metric_name} gauge")
            lines.append(f"{metric_name} {rate}")

        return "\n".join(lines) + "\n"
