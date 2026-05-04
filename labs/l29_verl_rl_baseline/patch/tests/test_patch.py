"""L10 Patch tests · CPU OK."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pytest

PATCH_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PATCH_DIR))


def _impl():
    name = os.environ.get("IMPL", "starter")
    return importlib.import_module(f"{name}.kl_controller")


def test_initial_coef():
    impl = _impl()
    ctrl = impl.AdaptiveKLController(init_kl_coef=0.2, target_kl=0.05, horizon=10000)
    assert ctrl.get_coef() == pytest.approx(0.2)


def test_increase_when_kl_too_high():
    impl = _impl()
    ctrl = impl.AdaptiveKLController(init_kl_coef=0.2, target_kl=0.05, horizon=100)
    ctrl.update(current_kl=0.10, n_steps=1)  # 2x target → error = 1, clipped to 0.2
    assert ctrl.get_coef() > 0.2


def test_decrease_when_kl_too_low():
    impl = _impl()
    ctrl = impl.AdaptiveKLController(init_kl_coef=0.2, target_kl=0.05, horizon=100)
    ctrl.update(current_kl=0.01, n_steps=1)  # 0.2x target → error = -0.8, clipped to -0.2
    assert ctrl.get_coef() < 0.2


def test_no_update_when_at_target():
    impl = _impl()
    ctrl = impl.AdaptiveKLController(init_kl_coef=0.2, target_kl=0.05, horizon=100)
    ctrl.update(current_kl=0.05, n_steps=1)
    assert ctrl.get_coef() == pytest.approx(0.2)


def test_clip_limits_max_change():
    """Even with extreme current_kl, single update should change coef by at most ~20% per horizon-step."""
    impl = _impl()
    ctrl = impl.AdaptiveKLController(init_kl_coef=1.0, target_kl=0.01, horizon=10)
    initial = ctrl.get_coef()
    # current_kl 100x target → unclipped error would be huge; clip to +0.2
    # 1.0 * (1 + 0.2 * 1 / 10) = 1.0 * 1.02 = 1.02
    ctrl.update(current_kl=1.0, n_steps=1)
    # Change should be exactly +2% (0.2 * 1/10), not 100x
    assert abs(ctrl.get_coef() - 1.02) < 1e-6, (
        f"expected +2% change; got {ctrl.get_coef()}"
    )
