SHELL := /bin/bash
PYTHON ?= python
MISSION_ID := l11_megatron_scale_optimization

.PHONY: patch-test patch-hint patch-show-solution clean

patch-test:
	$(PYTHON) ../../scripts/run_patch_test.py --mission $(MISSION_ID)

patch-hint:
	@echo "==> TODO 列表："
	@grep -nH "TODO(student)" patch/starter/*.py | head -30 || true
	@echo ""
	@echo "==> 关键提示："
	@echo "  - _create_buckets：按 named_parameters 顺序累加，超过 bucket_size_bytes 开新桶"
	@echo "  - 单 param > bucket size 也独占一桶"
	@echo "  - synchronize_grads：_flatten_dense_tensors → all_reduce → _unflatten → copy_"

patch-show-solution:
	@cat patch/reference/bucketed_ddp.py

clean:
	rm -rf patch/__pycache__ patch/*/__pycache__ patch/.pytest_cache
