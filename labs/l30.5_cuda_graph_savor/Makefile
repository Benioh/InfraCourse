SHELL := /bin/bash
PYTHON ?= python
MISSION_ID := l30.5_cuda_graph_savor

.PHONY: patch-test patch-hint patch-show-solution clean

patch-test:
	$(PYTHON) ../../scripts/run_patch_test.py --mission $(MISSION_ID)

patch-hint:
	@echo "==> TODO 列表："
	@grep -nH "TODO(student)" patch/starter/*.py | head -20 || true
	@echo ""
	@echo "==> 关键提示："
	@echo "  - GraphCache 按 args/kwargs 的 shape/dtype/scalar 生成 key"
	@echo "  - 首次 key miss 保存 fn 并 capture_count += 1；命中时 replay_count += 1"
	@echo "  - MemorySavor.pause(tensor) 保存 handle/data，resume(handle) 消费 handle"
	@echo "  - CPU patch 验证结构语义，不验证真实 CUDA Graph timing 或 VMM 物理页"

patch-show-solution:
	@for f in patch/reference/*.py; do \
		if [ "$$(basename $$f)" != "__init__.py" ]; then \
			echo "----- $$f -----"; cat "$$f"; \
		fi; \
	done

clean:
	rm -rf patch/__pycache__ patch/*/__pycache__ patch/.pytest_cache
