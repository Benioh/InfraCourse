SHELL := /bin/bash
PYTHON ?= python
MISSION_ID := l29.5_train_infer_mismatch

.PHONY: patch-test patch-hint patch-show-solution clean

patch-test:
	$(PYTHON) ../../scripts/run_patch_test.py --mission $(MISSION_ID)

patch-hint:
	@echo "==> TODO 列表："
	@grep -nH "TODO(student)" patch/starter/*.py | head -20 || true
	@echo ""
	@echo "==> 关键提示："
	@echo "  - K3 KL 是 (p/q) - 1 - log(p/q)，给 log_p 和 log_q 时先算 log_ratio"
	@echo "  - TIS 是 clamp(ratio, lo, hi)；MIS 是把 ratio 越界的 token mask 掉"
	@echo "  - geometric IS = exp(mean(log_ratio))，序列长度归一化"
	@echo "  - veto 用对数比较：logp >= log(threshold)"
	@echo "  - batch_normalize 让权重均值为 1，解决 SNIS 学习率震荡"

patch-show-solution:
	@cat patch/reference/mismatch.py

clean:
	rm -rf patch/__pycache__ patch/*/__pycache__ patch/.pytest_cache
