SHELL := /bin/bash
PYTHON ?= python
MISSION_ID := l23_quant_serving

.PHONY: patch-test patch-hint patch-show-solution clean

patch-test:
	$(PYTHON) ../../scripts/run_patch_test.py --mission $(MISSION_ID)

patch-hint:
	@echo "==> TODO 列表："
	@grep -nH "TODO(student)" patch/starter/*.py | head -30 || true
	@echo ""
	@echo "==> 关键提示："
	@echo "  - awq scale = (act_amax^α / w_amax_in^α), clamp(min=1.0)"
	@echo "  - quantize: w_scaled = w * scale; scale_o = max(|w_scaled|, dim=1) / 127"
	@echo "  - dequantize: int_w * scale_o; 再除回 awq_scale"

patch-show-solution:
	@cat patch/reference/awq_calibrate.py

clean:
	rm -rf patch/__pycache__ patch/*/__pycache__ patch/.pytest_cache
