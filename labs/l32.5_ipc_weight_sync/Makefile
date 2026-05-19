SHELL := /bin/bash
PYTHON ?= python
MISSION_ID := l32.5_ipc_weight_sync

.PHONY: patch-test patch-hint patch-show-solution clean

patch-test:
	$(PYTHON) ../../scripts/run_patch_test.py --mission $(MISSION_ID)

patch-hint:
	@echo "==> TODO 列表："
	@grep -nH "TODO(student)" patch/starter/*.py | head -20 || true
	@echo ""
	@echo "==> 关键提示："
	@echo "  - serialize_handle 不要把 tensor data 塞进 bytes，只塞 handle + meta"
	@echo "  - deserialize_handle 必须返回 share storage 的 tensor（不复制数据）"
	@echo "  - gather_handles_to_rank0：rank!=0 返回 None，rank==0 返回 list"
	@echo "  - LocalSerializedTensor.get(rank) 反序列化对应 rank 的 handle"
	@echo "  - flush_cache 在最后一个参数上才生效"

patch-show-solution:
	@cat patch/reference/ipc_weight_sync.py

clean:
	rm -rf patch/__pycache__ patch/*/__pycache__ patch/.pytest_cache
