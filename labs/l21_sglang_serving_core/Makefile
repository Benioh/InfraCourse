SHELL := /bin/bash
PYTHON ?= python
MISSION_ID := l21_sglang_serving_core

.PHONY: patch-test patch-hint patch-show-solution clean

patch-test:
	$(PYTHON) ../../scripts/run_patch_test.py --mission $(MISSION_ID)

patch-hint:
	@echo "==> TODO 列表："
	@grep -nH "TODO(student)" patch/starter/*.py | head -30 || true
	@echo ""
	@echo "==> 关键提示："
	@echo "  - match_prefix：从 root 沿 token_ids 走，能走多深就走多深；同时更新 last_access"
	@echo "  - insert：缺失节点时新建；累计 _total；超过 max_tokens 触发 evict"
	@echo "  - evict：DFS 收集所有叶子，按 last_access 升序，先删 LRU"
	@echo "  - 共享 prefix 不能被一次 evict 全删——每次只删一个叶子"

patch-show-solution:
	@cat patch/reference/radix_cache.py

clean:
	rm -rf patch/__pycache__ patch/*/__pycache__ patch/.pytest_cache
