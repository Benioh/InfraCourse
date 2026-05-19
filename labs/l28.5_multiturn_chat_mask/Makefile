SHELL := /bin/bash
PYTHON ?= python
MISSION_ID := l28.5_multiturn_chat_mask

.PHONY: patch-test patch-hint patch-show-solution clean

patch-test:
	$(PYTHON) ../../scripts/run_patch_test.py --mission $(MISSION_ID)

patch-hint:
	@echo "==> TODO 列表："
	@grep -nH "TODO(student)" patch/starter/*.py | head -20 || true
	@echo ""
	@echo "==> 关键提示："
	@echo "  - 用 fixed BASE conversation 作为公共前缀；每条 msg 单独 apply_chat_template(BASE + [msg])"
	@echo "  - delta = full_str[len(base_str):]"
	@echo "  - 对 assistant 消息：loss_mask = delta_ids；其他 role：loss_mask = [-100] * len(delta_ids)"
	@echo "  - tool / system / user 全部归为 'mask -100' 类"
	@echo "  - attention_mask 全 1（不像训练 padding，本算子不引入 padding）"

patch-show-solution:
	@cat patch/reference/multiturn_tokenizer.py

clean:
	rm -rf patch/__pycache__ patch/*/__pycache__ patch/.pytest_cache
