"""Profiler 数据解析与瓶颈分类 — 参考实现"""


def parse_op_summary(events: list[dict], top_k: int = 5) -> list[dict]:
    total_cuda = sum(e["cuda_time_us"] for e in events)
    sorted_events = sorted(events, key=lambda e: e["cuda_time_us"], reverse=True)

    result = []
    for e in sorted_events[:top_k]:
        result.append({
            "name": e["name"],
            "cuda_time_us": e["cuda_time_us"],
            "calls": e["calls"],
            "avg_cuda_time_us": e["cuda_time_us"] / e["calls"] if e["calls"] > 0 else 0,
            "pct": (e["cuda_time_us"] / total_cuda * 100) if total_cuda > 0 else 0,
        })
    return result


def classify_bottleneck(profile_stats: dict) -> dict:
    gpu_util = profile_stats["gpu_utilization"]
    top_op = profile_stats["top_op_type"]
    kernel_gap = profile_stats["avg_kernel_gap_us"]
    comm_pct = profile_stats["comm_pct"]
    data_wait = profile_stats["data_wait_pct"]

    if data_wait > 20:
        return {"bottleneck": "data_loading", "confidence": "high",
                "evidence": f"data_wait_pct={data_wait:.1f}% > 20%"}
    if comm_pct > 30 and gpu_util < 0.7:
        return {"bottleneck": "communication_bound", "confidence": "high",
                "evidence": f"comm_pct={comm_pct:.1f}% > 30% and gpu_util={gpu_util:.2f} < 0.7"}
    if kernel_gap > 50 and gpu_util < 0.6:
        return {"bottleneck": "launch_bound", "confidence": "high",
                "evidence": f"avg_kernel_gap={kernel_gap:.1f}μs > 50 and gpu_util={gpu_util:.2f} < 0.6"}
    if gpu_util > 0.8 and top_op == "matmul":
        return {"bottleneck": "compute_bound", "confidence": "high",
                "evidence": f"gpu_util={gpu_util:.2f} > 0.8 and top_op=matmul"}
    if gpu_util > 0.8 and top_op == "elementwise":
        return {"bottleneck": "memory_bound", "confidence": "high",
                "evidence": f"gpu_util={gpu_util:.2f} > 0.8 and top_op=elementwise"}
    return {"bottleneck": "compute_bound", "confidence": "medium",
            "evidence": "no strong signal, defaulting to compute_bound"}


def detect_gpu_idle_segments(timeline: list[dict], threshold_us: float = 100.0) -> list[dict]:
    if len(timeline) < 2:
        return []

    segments = []
    for i in range(len(timeline) - 1):
        gap_start = timeline[i]["end_us"]
        gap_end = timeline[i + 1]["start_us"]
        duration = gap_end - gap_start
        if duration > threshold_us:
            segments.append({
                "start_us": gap_start,
                "end_us": gap_end,
                "duration_us": duration,
                "before_kernel": timeline[i]["name"],
                "after_kernel": timeline[i + 1]["name"],
            })
    return segments


def memory_breakdown(allocations: list[dict]) -> dict:
    live = [a for a in allocations if a["is_live"]]
    total = sum(a["size_bytes"] for a in live)

    breakdown = {}
    for a in live:
        cat = a["category"]
        breakdown[cat] = breakdown.get(cat, 0) + a["size_bytes"]

    if total == 0:
        breakdown_pct = {cat: 0.0 for cat in breakdown}
        largest = "none"
    else:
        breakdown_pct = {cat: val / total * 100 for cat, val in breakdown.items()}
        largest = max(breakdown, key=breakdown.get)

    return {
        "total_live_bytes": total,
        "breakdown": breakdown,
        "breakdown_pct": breakdown_pct,
        "largest_category": largest,
    }
