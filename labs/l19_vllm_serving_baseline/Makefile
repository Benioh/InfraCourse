SHELL := /bin/bash
PYTHON ?= python
MISSION_ID := l19_vllm_serving_baseline

.PHONY: patch-test patch-hint patch-show-solution clean

patch-test:
	$(PYTHON) ../../scripts/run_patch_test.py --mission $(MISSION_ID)

patch-hint:
	@echo "==> TODO 列表："
	@grep -nH "TODO(student)" patch/starter/*.py | head -30 || true
	@echo ""
	@echo "==> 关键提示："
	@echo "  - probs = softmax; info = -log(probs); entropy = sum(p*info)"
	@echo "  - dist = |info - entropy|, 按 dist 升序排序"
	@echo "  - cumsum prob，> typical_p 截断；首位强制保留"
	@echo "  - scatter back to original vocab order，masked_fill"

patch-show-solution:
	@cat patch/reference/typical_p.py

clean:
	rm -rf patch/__pycache__ patch/*/__pycache__ patch/.pytest_cache
