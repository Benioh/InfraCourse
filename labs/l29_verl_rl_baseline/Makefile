SHELL := /bin/bash
PYTHON ?= python
MISSION_ID := l29_verl_rl_baseline

.PHONY: patch-test patch-hint patch-show-solution clean

patch-test:
	$(PYTHON) ../../scripts/run_patch_test.py --mission $(MISSION_ID)

patch-hint:
	@echo "==> TODO 列表："
	@grep -nH "TODO(student)" patch/starter/*.py | head -20 || true
	@echo ""
	@echo "==> 关键提示："
	@echo "  - error = current_kl / target_kl - 1.0"
	@echo "  - clip 到 [-0.2, 0.2]"
	@echo "  - self.value *= 1.0 + error * n_steps / horizon"

patch-show-solution:
	@cat patch/reference/kl_controller.py

clean:
	rm -rf patch/__pycache__ patch/*/__pycache__ patch/.pytest_cache
