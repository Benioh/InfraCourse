SHELL := /bin/bash
PYTHON ?= python
MISSION_ID := l09_long_context_cp

.PHONY: patch-test patch-hint patch-show-solution clean

patch-test:
	$(PYTHON) ../../scripts/run_patch_test.py --mission $(MISSION_ID)

patch-hint:
	@echo "==> TODO 列表："
	@grep -nH "TODO(student)" patch/starter/*.py | head -30 || true
	@echo ""
	@echo "==> 关键提示："
	@echo "  - torch.chunk(k, num_chunks, dim=2) 把 K 沿 seq 切分"
	@echo "  - running_max 初始 -inf；first chunk 时 exp(-inf - new_max) → 用 torch.where 安全处理"
	@echo "  - 数学：out_new = out_old * exp(m_old - m_new) + exp(scores_new - m_new) @ V_new"

patch-show-solution:
	@cat patch/reference/ring_attention.py

clean:
	rm -rf patch/__pycache__ patch/*/__pycache__ patch/.pytest_cache
