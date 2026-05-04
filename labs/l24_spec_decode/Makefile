SHELL := /bin/bash
PYTHON ?= python
MISSION_ID := l24_spec_decode

.PHONY: patch-test patch-hint patch-show-solution clean

patch-test:
	$(PYTHON) ../../scripts/run_patch_test.py --mission $(MISSION_ID)

patch-hint:
	@echo "==> TODO 列表："
	@grep -nH "TODO(student)" patch/starter/*.py | head -20 || true
	@echo ""
	@echo "==> 关键提示："
	@echo "  - target_argmax = target_logits.argmax(dim=-1).tolist() 长度 k+1"
	@echo "  - for i in range(k): 比对 target_argmax[i] vs draft_tokens[i]"
	@echo "  - 第一个 mismatch：return accepted + [target_argmax[i]], len(accepted)"
	@echo "  - 全 accept：append target_argmax[k]，return accepted, k"

patch-show-solution:
	@cat patch/reference/spec_decode.py

clean:
	rm -rf patch/__pycache__ patch/*/__pycache__ patch/.pytest_cache
