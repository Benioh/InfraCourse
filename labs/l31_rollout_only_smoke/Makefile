SHELL := /bin/bash
PYTHON ?= python
MISSION_ID := l31_rollout_only_smoke

.PHONY: patch-test patch-hint patch-show-solution clean

patch-test:
	$(PYTHON) ../../scripts/run_patch_test.py --mission $(MISSION_ID)

patch-hint:
	@echo "==> TODO 列表："
	@grep -nH "TODO(student)" patch/starter/*.py | head -20 || true
	@echo ""
	@echo "==> 关键提示："
	@echo "  - asyncio.Semaphore(max_concurrency)"
	@echo "  - async with sem: await self.generate_fn(p)"
	@echo "  - asyncio.gather 保序"

patch-show-solution:
	@cat patch/reference/rollout_pool.py

clean:
	rm -rf patch/__pycache__ patch/*/__pycache__ patch/.pytest_cache
