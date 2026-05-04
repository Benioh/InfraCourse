"""MoE and expert-parallel teaching primitives for L05.5."""

from .capacity import apply_capacity
from .router import route_tokens

__all__ = ["apply_capacity", "route_tokens"]
