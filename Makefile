SHELL := /bin/bash
.DEFAULT_GOAL := list-missions

PYTHON ?= python
BACKEND_APP ?= app.backend.main:app
FRONTEND_DIR := app/frontend

# Export key env vars to all sub-makes (so `IMPL=reference make patch-test-all`
# propagates IMPL into each lab's patch-test invocation).
export IMPL
export TP_IMPL
export PYTHON

define require_mission
	@if [[ -z "$(M)" ]]; then echo "Set M=<mission_id>"; exit 1; fi
endef

define mission_make
	$(MAKE) -C labs/$(M)
endef

.PHONY: env frontend-install check-env app backend notebook list-missions mini-infra mini-infra-smoke mini-infra-real-stack mini-infra-train mini-infra-bench mini-infra-delivery mission download-data preprocess smoke run-4090 run-h200 resume report clean-runs serve-vllm serve-sglang bench patch-test patch-hint patch-show-solution patch-test-all

env:
	@if [[ -z "$(ENV)" ]]; then echo "Set ENV=<env_name>"; exit 1; fi
	bash scripts/env/create_env.sh "$(ENV)"

frontend-install:
	npm --prefix $(FRONTEND_DIR) install

check-env:
	$(PYTHON) scripts/env/check_cuda.py
	$(PYTHON) scripts/env/check_nccl.py
	$(PYTHON) scripts/env/collect_env.py --pretty

app:
	npm --prefix $(FRONTEND_DIR) run dev

backend:
	uvicorn $(BACKEND_APP) --reload --port 8000

notebook:
	jupyter lab

list-missions:
	$(PYTHON) app/backend/manage.py list-missions

mini-infra:
	$(call require_mission)
	$(PYTHON) -m mini_infra.lab_runner --mission "$(M)" $(if $(RUN_ID),--run-id "$(RUN_ID)",)

mini-infra-smoke:
	$(PYTHON) -m mini_infra.scripts.run_smoke $(if $(RUN_ID),--run-id "$(RUN_ID)",)

mini-infra-real-stack:
	$(PYTHON) -m mini_infra.real_stack_smoke $(if $(RUN_ID),--run-id "$(RUN_ID)",)

mini-infra-train:
	$(PYTHON) -m mini_infra.training.train_tiny --backend $(or $(BACKEND),simulated) $(if $(RUN_ID),--run-id "$(RUN_ID)",)

mini-infra-bench:
	$(PYTHON) -m mini_infra.serving.benchmark $(if $(RUN_ID),--run-id "$(RUN_ID)",)

mini-infra-delivery:
	$(PYTHON) -m mini_infra.reports.build_delivery

mission:
	$(call require_mission)
	$(call mission_make) mission

download-data:
	$(call require_mission)
	$(call mission_make) download-data

preprocess:
	$(call require_mission)
	$(call mission_make) preprocess

smoke:
	$(call require_mission)
	$(call mission_make) smoke

run-4090:
	$(call require_mission)
	$(call mission_make) run-4090

run-h200:
	$(call require_mission)
	$(call mission_make) run-h200

resume:
	$(call require_mission)
	$(call mission_make) resume

serve-vllm:
	$(call require_mission)
	$(call mission_make) serve-vllm

serve-sglang:
	$(call require_mission)
	$(call mission_make) serve-sglang

bench:
	$(call require_mission)
	$(call mission_make) bench

report:
	$(call require_mission)
	$(call mission_make) report

# ★ Patch Track（每关的核心动作）：写补丁 + 跑测试 → PASS 即过关
patch-test:
	$(call require_mission)
	$(call mission_make) patch-test

patch-hint:
	$(call require_mission)
	$(call mission_make) patch-hint

patch-show-solution:
	$(call require_mission)
	$(call mission_make) patch-show-solution

# 跑全 patch lab 的 patch-test（CI 用）
patch-test-all:
	@for lab in labs/l*/; do \
		if [ -d "$$lab/patch" ]; then \
			echo "=== $$lab ==="; \
			$(MAKE) -C "$$lab" patch-test || exit 1; \
		fi; \
	done

clean-runs:
	$(call require_mission)
	$(call mission_make) clean-runs
