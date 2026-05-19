SHELL := /bin/bash
PYTHON ?= python
MISSION_ID := l02.5_memory_snapshot

.PHONY: patch-test patch-hint patch-show-solution clean

patch-test:
	$(PYTHON) ../../scripts/run_patch_test.py --mission $(MISSION_ID)

patch-hint:
	@echo "==> TODO 列表："
	@grep -nH "TODO(student)" patch/starter/*.py | head -20 || true
	@echo ""
	@echo "==> 关键提示："
	@echo "  - alloc 必须只在 enabled=True 时才记录；disabled 状态返回 -1"
	@echo "  - free 必须从 _live 中弹出，并把事件加入 _events"
	@echo "  - dump_snapshot 返回 events、live_allocations、total_leaked_bytes 三键"
	@echo "  - find_top_leaks_by_stack 按 stack tuple 分组求和，再按 size 降序排"
	@echo "  - get_caller_stack 用 inspect.stack()，深度由参数控制"

patch-show-solution:
	@cat patch/reference/memory_snapshot.py

clean:
	rm -rf patch/__pycache__ patch/*/__pycache__ patch/.pytest_cache
