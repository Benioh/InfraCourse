SHELL := /bin/bash
PYTHON ?= python
MISSION_ID := l18_data_engineering

.PHONY: patch-test patch-hint patch-show-solution clean

patch-test:
	$(PYTHON) ../../scripts/run_patch_test.py --mission $(MISSION_ID)

patch-hint:
	@echo "==> TODO 列表："
	@grep -nH "TODO(student)" patch/starter/*.py | head -30 || true
	@echo ""
	@echo "==> 关键提示："
	@echo "  - shingle 是 char-level n-gram；空文本返回全 0 签名"
	@echo "  - 每个 shingle 用 md5 算基础 hash；num_perm 个 (a,b) 算 num_perm 个签名"
	@echo "  - jaccard_estimate = 签名相同位置数 / num_perm"
	@echo "  - dedup 用 first-occurrence + O(N²) 比对（够小数据集用）"

patch-show-solution:
	@cat patch/reference/dedup_minhash.py

clean:
	rm -rf patch/__pycache__ patch/*/__pycache__ patch/.pytest_cache
