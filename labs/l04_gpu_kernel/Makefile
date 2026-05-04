SHELL := /bin/bash
PYTHON ?= python
MISSION_ID := l04_gpu_kernel

.PHONY: patch-test patch-hint patch-show-solution clean

patch-test:
	$(PYTHON) ../../scripts/run_patch_test.py --mission $(MISSION_ID)

patch-hint:
	@echo "==> TODO 列表："
	@grep -nH "TODO(student)" patch/starter/*.py | head -30 || true
	@echo ""
	@echo "==> 关键提示："
	@echo "  - kernel 里 row_idx = tl.program_id(0)，按行启动 grid"
	@echo "  - tl.load 用 mask + other=-inf 处理 padding"
	@echo "  - 必须减 max 再 exp（数值稳定）"
	@echo "  - BLOCK_SIZE = triton.next_power_of_2(n_cols)"

patch-show-solution:
	@cat patch/reference/triton_softmax.py

clean:
	rm -rf patch/__pycache__ patch/*/__pycache__ patch/.pytest_cache
