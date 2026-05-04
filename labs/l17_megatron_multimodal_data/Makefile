SHELL := /bin/bash
PYTHON ?= python
MISSION_ID := l17_megatron_multimodal_data

.PHONY: patch-test patch-hint patch-show-solution clean

patch-test:
	$(PYTHON) ../../scripts/run_patch_test.py --mission $(MISSION_ID)

patch-hint:
	@echo "==> TODO 列表："
	@grep -nH "TODO(student)" patch/starter/*.py | head -30 || true
	@echo ""
	@echo "==> 关键提示："
	@echo "  - per-item: 先拼 [image_token x N][audio_token x M][text_ids]"
	@echo "  - batch pad to S = max(per-item len), 用 PAD_TOKEN_ID 填"
	@echo "  - 收集 image: torch.stack(images), 记 image_batch_indices"
	@echo "  - 收集 audio: 截到 max_audio_frames, F.pad 到 max, 同步生成 audio_mask"

patch-show-solution:
	@cat patch/reference/multimodal_collate.py

clean:
	rm -rf patch/__pycache__ patch/*/__pycache__ patch/.pytest_cache
