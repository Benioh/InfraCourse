SHELL := /bin/bash
PYTHON ?= python

.PHONY: patch-test patch-hint patch-show-solution clean

patch-test:
	$(PYTHON) ../../scripts/run_patch_test.py --mission $(notdir $(CURDIR))

patch-hint:
	@grep -nH "TODO(student)" patch/starter/*.py | head -30 || true

patch-show-solution:
	@for f in patch/reference/*.py; do \
		if [ "$$(basename $$f)" != "__init__.py" ]; then \
			echo "----- $$f -----"; cat "$$f"; \
		fi; \
	done

clean:
	rm -rf patch/__pycache__ patch/*/__pycache__ patch/.pytest_cache
