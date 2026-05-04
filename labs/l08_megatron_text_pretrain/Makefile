SHELL := /bin/bash
PYTHON ?= python
MISSION_ID := l08_megatron_text_pretrain

.PHONY: patch-test patch-hint patch-show-solution clean

patch-test:
	$(PYTHON) ../../scripts/run_patch_test.py --mission $(MISSION_ID)

patch-hint:
	@echo "==> TODO 列表："
	@grep -nH "TODO(student)" patch/starter/*.py | head -30 || true
	@echo ""
	@echo "==> 关键提示："
	@echo "  - boundaries = [0] + restart_steps + [total_steps]"
	@echo "  - _set_lr 写到所有 param_groups"
	@echo "  - 余弦公式：min + 0.5*(max-min)*(1+cos(π·t/segment_len))"

patch-show-solution:
	@cat patch/reference/lr_scheduler.py

clean:
	rm -rf patch/__pycache__ patch/*/__pycache__ patch/.pytest_cache
