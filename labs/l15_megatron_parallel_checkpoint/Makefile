SHELL := /bin/bash
PYTHON ?= python
MISSION_ID := l15_megatron_parallel_checkpoint

.PHONY: patch-test patch-hint patch-show-solution clean

patch-test:
	$(PYTHON) ../../scripts/run_patch_test.py --mission $(MISSION_ID)

patch-hint:
	@echo "==> TODO 列表："
	@grep -nH "TODO(student)" patch/starter/*.py | head -30 || true
	@echo ""
	@echo "==> 关键提示："
	@echo "  - world_size = dp * pp * tp * cp * ep"
	@echo "  - rank 坐标可以用连续取模拆解"
	@echo "  - 改 TP checkpoint 必须 gather -> concat -> split -> save"

patch-show-solution:
	@cat patch/reference/checkpointing.py

clean:
	rm -rf patch/__pycache__ patch/*/__pycache__ patch/.pytest_cache
