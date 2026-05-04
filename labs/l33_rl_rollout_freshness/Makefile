SHELL := /bin/bash
PYTHON ?= python
MISSION_ID := l33_rl_rollout_freshness

.PHONY: patch-test patch-hint patch-show-solution clean

patch-test:
	$(PYTHON) ../../scripts/run_patch_test.py --mission $(MISSION_ID)

patch-hint:
	@echo "==> TODO 列表："
	@grep -nH "TODO(student)" patch/starter/*.py | head -30 || true
	@echo ""
	@echo "==> 关键提示："
	@echo "  - gap = actor_step - engine_step"
	@echo "  - stale engine 需要 sync"
	@echo "  - 如果可用 fresh engine 少于 min_ready_engines，应阻塞新 rollout"

patch-show-solution:
	@cat patch/reference/rollout_manager.py

clean:
	rm -rf patch/__pycache__ patch/*/__pycache__ patch/.pytest_cache
