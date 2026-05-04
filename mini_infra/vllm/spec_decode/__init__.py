"""vLLM-shaped speculative decoding teaching primitives."""

from .acceptance_tracker import AcceptanceTracker, acceptance_summary
from .draft_runner import spec_decode_summary
from .ngram import ngram_summary

__all__ = ["AcceptanceTracker", "acceptance_summary", "ngram_summary", "spec_decode_summary"]
