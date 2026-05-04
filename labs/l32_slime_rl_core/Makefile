SHELL := /bin/bash
PYTHON ?= python
MISSION_ID := l32_slime_rl_core

.PHONY: patch-test patch-hint patch-show-solution clean

patch-test:
	$(PYTHON) ../../scripts/run_patch_test.py --mission $(MISSION_ID)

patch-hint:
	@echo "==> TODO 列表："
	@grep -nH "TODO(student)" patch/starter/*.py | head -20 || true
	@echo ""
	@echo "==> 关键提示："
	@echo "  - 校验 shape 和 dtype 必须同时匹配"
	@echo "  - 不匹配 / 缺失 key 进 mismatched_keys 不报错"
	@echo "  - bytes_synced 用 numel × element_size 累加"

patch-show-solution:
	@cat patch/reference/weight_sync.py

clean:
	rm -rf patch/__pycache__ patch/*/__pycache__ patch/.pytest_cache
