SHELL := /bin/bash
PYTHON ?= python

.PHONY: patch-test patch-hint patch-show-solution clean

patch-test:
	$(PYTHON) ../../scripts/run_patch_test.py --mission $(notdir $(CURDIR))

patch-hint:
	@echo "==> TODO 列表（按出现顺序）："
	@grep -nH "TODO(student)" patch/starter/*.py | head -30 || true
	@echo ""
	@if [ -f patch/HINTS.md ]; then echo "==> 关键提示："; cat patch/HINTS.md; fi

patch-show-solution:
	@echo "==> 参考解：patch/reference/"
	@for f in patch/reference/*.py; do \
		if [ "$$(basename $$f)" != "__init__.py" ]; then \
			echo "----- $$f -----"; cat "$$f"; \
		fi; \
	done

clean:
	rm -rf patch/__pycache__ patch/*/__pycache__ patch/.pytest_cache patch/.hypothesis
