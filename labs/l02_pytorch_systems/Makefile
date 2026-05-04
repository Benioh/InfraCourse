SHELL := /bin/bash
PYTHON ?= python
MISSION_ID := l02_pytorch_systems

.PHONY: patch-test patch-hint patch-show-solution clean

patch-test:
	$(PYTHON) ../../scripts/run_patch_test.py --mission $(MISSION_ID)

patch-hint:
	@echo "==> TODO 列表："
	@grep -nH "TODO(student)" patch/starter/*.py | head -30 || true
	@echo ""
	@echo "==> 关键提示："
	@echo "  - count_param_bytes：sum(p.numel() * p.element_size() for p in model.parameters())"
	@echo "  - count_grad_bytes：同上但用 p.grad，注意 None 跳过"
	@echo "  - count_optimizer_state_bytes：optimizer.state.values() → 每个 dict → 只统计 Tensor"

patch-show-solution:
	@cat patch/reference/memory_probe.py

clean:
	rm -rf patch/__pycache__ patch/*/__pycache__ patch/.pytest_cache
