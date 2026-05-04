from __future__ import annotations

import argparse
import json

from mini_infra.sglang.srt.managers.disagg_service import DisaggregationService
from mini_infra.sglang.srt.managers.scheduler import Req, Scheduler
from mini_infra.sglang.quant.kv_int8 import kv_int8_summary
from mini_infra.observability.io import command_snapshot, mini_run_dir, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Mini SGLang scheduler smoke")
    parser.add_argument("--pd", action="store_true")
    parser.add_argument("--kv-int8", action="store_true")
    parser.add_argument("--run-id")
    parser.add_argument("--port", type=int)
    args = parser.parse_args()
    scheduler = Scheduler()
    scheduler.add_request(Req("r0", "system math tutor question one".split(), []))
    scheduler.add_request(Req("r1", "system math tutor question two".split(), []))
    first = scheduler.run_batch()
    scheduler.add_request(Req("r2", "system math tutor question one".split(), []))
    second = scheduler.run_batch()
    payload = {"first": first, "second": second}
    if args.pd:
        service = DisaggregationService()
        service.transfer_kv("r2", "prefill-0", "decode-0", token_count=5)
        payload["pd_metrics"] = service.metrics()
    if args.kv_int8:
        payload["kv_int8"] = kv_int8_summary()
    if args.run_id:
        run_dir = mini_run_dir("sglang", args.run_id)
        command_snapshot(run_dir / "command.sh")
        write_json(run_dir / "artifacts" / "scheduler.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
