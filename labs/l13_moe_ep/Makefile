SHELL := /bin/bash
PYTHON ?= python
MISSION_ID := l13_moe_ep

.PHONY: patch-test patch-hint patch-show-solution clean

patch-test:
	$(PYTHON) ../../scripts/run_patch_test.py --mission $(MISSION_ID)

patch-hint:
	@echo "==> TODO 列表："
	@grep -nH "TODO(student)" patch/starter/*.py | head -30 || true
	@echo ""
	@echo "==> 关键提示："
	@echo "  - probs = softmax(logits, dim=-1) → topk(k=2) → 归一化 top2_vals"
	@echo "  - scatter_ 把 top2 vals 散回 (num_tokens, num_experts)"
	@echo "  - capacity = max(1, c * num_tokens * 2 / num_experts)"
	@echo "  - aux_loss 用 dispatch_mask（不可微）× probs（可微）"

patch-show-solution:
	@cat patch/reference/moe_router.py

clean:
	rm -rf patch/__pycache__ patch/*/__pycache__ patch/.pytest_cache
