"""
N-gram speculative decoding（无 draft model 版本）（L25 教学）。

教学目的：
    最便宜的 spec：从 prompt 自身提 n-gram 当候选。命中率比 draft model
    低，但零 GPU 占用、零额外显存。学员在 n18 notebook 里能看到：
        - 长 prompt（含模板/指令）命中率高
        - 短 prompt 命中率低（甚至为 0）
        - n=3 时 max_n 太小，n=8 时假命中变多

真实框架对照：
    - github_repo/vllm/vllm/spec_decode/ngram_worker.py
        真实 n-gram worker 含 cache、多 max_n 尝试；本文件做最小匹配。
"""
from __future__ import annotations


def ngram_candidates(prompt_tokens: list[str], suffix: list[str], max_tokens: int = 4) -> list[str]:
    """从 prompt 中找 suffix 出现位置，返回其后 max_tokens 作为候选。

    例：prompt = "the model should cite ... the model should explain"
        suffix = ["the", "model", "should"]
        → 找到第一处后取后续 max_tokens token 作为投机候选。
    """
    width = len(suffix)
    if width == 0:
        return []
    for index in range(0, len(prompt_tokens) - width):
        if prompt_tokens[index : index + width] == suffix:
            start = index + width
            return prompt_tokens[start : start + max_tokens]
    return []


def ngram_summary(prompt: str | None = None) -> dict[str, object]:
    text = prompt or "the model should cite sources and the model should explain assumptions"
    tokens = text.split()
    suffix = ["the", "model", "should"]
    candidates = ngram_candidates(tokens, suffix, max_tokens=3)
    return {
        "mode": "ngram",
        "suffix": suffix,
        "candidates": candidates,
        "acceptance_rate": 0.5 if candidates else 0.0,
        "draft_gpu_mem_gb": 0.0,
    }
