SHELL := /bin/bash
PYTHON ?= python
MISSION_ID := l01_env_conda_cuda
RUN_ID ?= $(shell date -u +%Y%m%dT%H%M%SZ)

.PHONY: mission smoke run-4090 run-h200 report grade clean-runs download-data preprocess resume serve-vllm serve-sglang bench

mission:
	@sed -n '1,120p' briefing.md

smoke:
	$(PYTHON) scripts/run_smoke.py --run-id $(RUN_ID) --config configs/4090_debug.yaml --mode smoke

run-4090:
	$(PYTHON) scripts/run_smoke.py --run-id $(RUN_ID) --config configs/4090_debug.yaml --mode 4090

run-h200:
	$(PYTHON) scripts/run_smoke.py --run-id $(RUN_ID) --config configs/h200_node.yaml --mode h200

report:
	@echo "中文报告会在 smoke/run 时写入 runs/$(MISSION_ID)/<run_id>/report.md"

grade:
	$(PYTHON) ../../autograder/grade.py --mission $(MISSION_ID) --run-id $(RUN_ID)

clean-runs:
	rm -rf ../../runs/$(MISSION_ID)

download-data preprocess resume serve-vllm serve-sglang bench:
	@echo "$(MISSION_ID) 不使用该 target"
