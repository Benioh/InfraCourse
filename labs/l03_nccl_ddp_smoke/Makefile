SHELL := /bin/bash
PYTHON ?= python
MISSION_ID := l03_nccl_ddp_smoke

.PHONY: patch-test patch-hint patch-show-solution clean

patch-test:
	$(PYTHON) ../../scripts/run_patch_test.py --mission $(MISSION_ID)

patch-hint:
	@echo "==> TODO 列表："
	@grep -nH "TODO(student)" patch/starter/*.py | head -30 || true
	@echo ""
	@echo "==> 关键提示："
	@echo "  - __init__：保留 self.module = model（不复制），记录 process_group 和 world_size"
	@echo "  - synchronize_grads：循环 params，跳过 None grad，dist.all_reduce(SUM) + /world_size"
	@echo "  - 数学合同：grad 必须取平均不是求和——这就是 DDP 的核心"

patch-show-solution:
	@cat patch/reference/manual_ddp.py

clean:
	rm -rf patch/__pycache__ patch/*/__pycache__ patch/.pytest_cache
