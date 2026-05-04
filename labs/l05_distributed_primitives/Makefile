SHELL := /bin/bash
PYTHON ?= python
MISSION_ID := l05_distributed_primitives

.PHONY: patch-test patch-hint patch-show-solution smoke clean

# ★ 本关核心动作：写补丁 + 跑测试 + 写 .last_run.json
patch-test:
	$(PYTHON) ../../scripts/run_patch_test.py --mission $(MISSION_ID)

patch-hint:
	@echo "==> 关键骨架提示（不是完整答案）："
	@grep -n "TODO\(student\)" patch/starter/tp_linear.py | head -20
	@echo ""
	@echo "==> 通信路径速查："
	@echo "  ColumnParallel forward : Identity (input)        → Linear (local) → optional all-gather"
	@echo "  ColumnParallel backward: all-reduce (grad-input)"
	@echo "  RowParallel    forward : split (input) → Linear (local) → all-reduce → bias on rank0"
	@echo "  RowParallel    backward: identity"

patch-show-solution:
	@echo "==> 完整参考解：patch/reference/tp_linear.py"
	@echo "    建议先关掉答案自己写一遍 starter/tp_linear.py，再对照。"
	@cat patch/reference/tp_linear.py

smoke:
	$(PYTHON) scripts/run_lab.py --mode smoke

clean:
	rm -rf patch/__pycache__ patch/*/__pycache__ .pytest_cache
