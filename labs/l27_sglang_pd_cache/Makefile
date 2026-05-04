SHELL := /bin/bash
PYTHON ?= python
MISSION_ID := l27_sglang_pd_cache

.PHONY: patch-test patch-hint patch-show-solution clean

patch-test:
	$(PYTHON) ../../scripts/run_patch_test.py --mission $(MISSION_ID)

patch-hint:
	@echo "==> TODO 列表："
	@grep -nH "TODO(student)" patch/starter/*.py | head -30 || true
	@echo ""
	@echo "==> 关键提示："
	@echo "  - prefix cache 是 token 级精确前缀，不是字符串相似度"
	@echo "  - uncached_tokens 决定 prefill 工作量"
	@echo "  - prefill queue 压力过高时不要盲目远端 prefill"

patch-show-solution:
	@cat patch/reference/disagg_service.py

clean:
	rm -rf patch/__pycache__ patch/*/__pycache__ patch/.pytest_cache
