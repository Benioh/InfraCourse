SHELL := /bin/bash
PYTHON ?= python
MISSION_ID := l34.5_gae_chunked_parallel

.PHONY: patch-test patch-hint patch-show-solution clean

patch-test:
	$(PYTHON) ../../scripts/run_patch_test.py --mission $(MISSION_ID)

patch-hint:
	@echo "==> TODO 列表："
	@grep -nH "TODO(student)" patch/starter/*.py | head -20 || true
	@echo ""
	@echo "==> 关键提示："
	@echo "  - naive 版本是反向递推：A_t = δ_t + γλ A_{t+1}"
	@echo "  - 分 chunk 后，每个 chunk 内部仍反向递推"
	@echo "  - 关键：从最后一个 chunk 往前传 'boundary correction' = (γλ)^k × A_next_chunk_start"
	@echo "  - 数学上必须与 naive 数值精确等价（差异 < 1e-6）"

patch-show-solution:
	@for f in patch/reference/*.py; do \
		if [ "$$(basename $$f)" != "__init__.py" ]; then \
			echo "----- $$f -----"; cat "$$f"; \
		fi; \
	done

clean:
	rm -rf patch/__pycache__ patch/*/__pycache__ patch/.pytest_cache
