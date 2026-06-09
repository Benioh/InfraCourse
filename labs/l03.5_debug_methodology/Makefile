SHELL := /bin/bash
PYTHON ?= python
MISSION_ID := l03.5_debug_methodology
RUN_ID ?= $(shell date -u +%Y%m%dT%H%M%SZ)

.PHONY: mission smoke report grade clean-runs

mission:
	@sed -n '1,120p' README.md

smoke:
	$(PYTHON) scripts/run_smoke.py --mode smoke

report:
	@echo "报告会在 smoke 时写入 runs/$(MISSION_ID)/<run_id>/report.md"

grade:
	$(PYTHON) ../../autograder/grade.py --mission $(MISSION_ID) --run-id $(RUN_ID)

clean-runs:
	rm -rf ../../runs/$(MISSION_ID)
