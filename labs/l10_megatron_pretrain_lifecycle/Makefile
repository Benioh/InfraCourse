SHELL := /bin/bash
PYTHON ?= python
MISSION_ID := l10_megatron_pretrain_lifecycle

.PHONY: patch-test patch-hint patch-show-solution clean

patch-test:
	$(PYTHON) ../../scripts/run_patch_test.py --mission $(MISSION_ID)

patch-hint:
	@echo "==> TODO 列表："
	@grep -nH "TODO(student)" patch/starter/*.py | head -30 || true
	@echo ""
	@echo "==> 关键提示："
	@echo "  - 按 STAGE_ORDER 排序，而不是按输入顺序相信调用链"
	@echo "  - handoff = producer.writes 与 consumer.reads 的交集"
	@echo "  - missing_stages 用来暴露源码阅读没有覆盖到的生命周期节点"

patch-show-solution:
	@cat patch/reference/train_step.py

clean:
	rm -rf patch/__pycache__ patch/*/__pycache__ patch/.pytest_cache
