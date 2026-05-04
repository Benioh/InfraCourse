SHELL := /bin/bash
PYTHON ?= python
MISSION_ID := l06_torchtitan_training

.PHONY: patch-test patch-hint patch-show-solution clean

patch-test:
	$(PYTHON) ../../scripts/run_patch_test.py --mission $(MISSION_ID)

patch-hint:
	@echo "==> TODO 列表："
	@grep -nH "TODO(student)" patch/starter/*.py | head -30 || true
	@echo ""
	@echo "==> 关键提示："
	@echo "  - selective_checkpoint_wrap：遍历 named_children，policy True 就 setattr 成 _CheckpointWrapper"
	@echo "  - attention_only_policy：name.lower() 含 attn 或 attention"
	@echo "  - 用 use_reentrant=False（PyTorch 2.1+ 推荐）"

patch-show-solution:
	@cat patch/reference/selective_ckpt.py

clean:
	rm -rf patch/__pycache__ patch/*/__pycache__ patch/.pytest_cache
