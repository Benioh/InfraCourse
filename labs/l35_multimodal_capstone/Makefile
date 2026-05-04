SHELL := /bin/bash
PYTHON ?= python
MISSION_ID := l35_multimodal_capstone

.PHONY: patch-test patch-hint patch-show-solution clean

patch-test:
	$(PYTHON) ../../scripts/run_patch_test.py --mission $(MISSION_ID)

patch-hint:
	@echo "==> TODO 列表："
	@grep -nH "TODO(student)" patch/starter/*.py | head -20 || true
	@echo ""
	@echo "==> 关键提示："
	@echo "  - MultimodalProjector：image_proj + audio_proj + image_pos_emb (learnable)"
	@echo "  - wer：经典 Levenshtein DP，O(N*M) 时间"
	@echo "  - clip_score：归一化向量后内积"

patch-show-solution:
	@cat patch/reference/mm_omni.py

clean:
	rm -rf patch/__pycache__ patch/*/__pycache__ patch/.pytest_cache
