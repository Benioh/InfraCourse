SHELL := /bin/bash
PYTHON ?= python
MISSION_ID := l26_sglang_pd_observability

.PHONY: patch-test patch-hint patch-show-solution clean

patch-test:
	$(PYTHON) ../../scripts/run_patch_test.py --mission $(MISSION_ID)

patch-hint:
	@echo "==> TODO 列表："
	@grep -nH "TODO(student)" patch/starter/*.py | head -30 || true
	@echo ""
	@echo "==> 关键提示："
	@echo "  - gauges/counters 用 (name, label_key) 作 dict key"
	@echo "  - record_event 维护 (hits, misses) tuple"
	@echo "  - export 时按 metric 名分组，labels 字母序"
	@echo "  - 0 events → ratio=0.0（不要 NaN）"

patch-show-solution:
	@cat patch/reference/metrics_exporter.py

clean:
	rm -rf patch/__pycache__ patch/*/__pycache__ patch/.pytest_cache
