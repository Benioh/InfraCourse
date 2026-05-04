SHELL := /bin/bash
PYTHON ?= python
MISSION_ID := l20_vllm_scheduler_kv

.PHONY: patch-test patch-hint patch-show-solution clean

patch-test:
	$(PYTHON) ../../scripts/run_patch_test.py --mission $(MISSION_ID)

patch-hint:
	@echo "==> TODO 列表："
	@grep -nH "TODO(student)" patch/starter/*.py | head -30 || true
	@echo ""
	@echo "==> 关键提示："
	@echo "  - decode 优先消耗 token budget"
	@echo "  - prefill 需要 prompt token budget 和 KV blocks"
	@echo "  - finished running request 应释放已占 KV blocks"

patch-show-solution:
	@cat patch/reference/scheduler.py

clean:
	rm -rf patch/__pycache__ patch/*/__pycache__ patch/.pytest_cache
