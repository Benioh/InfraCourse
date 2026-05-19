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
	@echo "  - GraphCache 按 batch_size 缓存，捕获时复用 static input/output buffer"
	@echo "  - replay 必须 copy_ 到 static buffer，不能新 alloc"
	@echo "  - MemorySavor.pause() 释放物理字节但记元数据，resume() 用元数据重建"
	@echo "  - 真实场景：RL co-locate 训练时 savor.pause(rollout) 让出显存给 train"

patch-show-solution:
	@for f in patch/reference/*.py; do \
		if [ "$$(basename $$f)" != "__init__.py" ]; then \
			echo "----- $$f -----"; cat "$$f"; \
		fi; \
	done

clean:
	rm -rf patch/__pycache__ patch/*/__pycache__ patch/.pytest_cache
