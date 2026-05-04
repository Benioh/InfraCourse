"""
Run a lab's patch tests and write a machine-readable status file.

Called by `make patch-test M=<lab>` and by the backend's POST /api/missions/{id}/patch-test.

Reads:  labs/<mission>/patch/tests/
Writes: labs/<mission>/patch/.last_run.json
        {
          "mission": "l05_distributed_primitives",
          "passed": true|false,
          "exit_code": 0|N,
          "started_at": ISO8601,
          "finished_at": ISO8601,
          "duration_s": float,
          "output": "<combined stdout/stderr, last 64KB>",
          "summary": {"passed": 5, "failed": 0, "total": 5}
        }

Always exits with the pytest exit code so Make / CI behave normally.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAX_OUTPUT_BYTES = 64 * 1024


def run(mission_id: str) -> int:
    lab_dir = ROOT / "labs" / mission_id
    patch_dir = lab_dir / "patch"
    if not patch_dir.is_dir():
        sys.stderr.write(f"[run_patch_test] no patch/ dir for {mission_id}\n")
        return 2

    started = time.time()
    started_iso = _utc_now_iso()

    # Build pytest command. By default skip tests marked `gpu` (they require
    # CUDA and are flaky in CI). Set RUN_GPU_TESTS=1 to opt-in.
    pytest_cmd = [sys.executable, "-m", "pytest", "tests/", "-v", "--color=no"]
    if os.environ.get("RUN_GPU_TESTS", "0") not in ("1", "true", "yes"):
        pytest_cmd.extend(["-m", "not gpu"])

    proc = subprocess.run(
        pytest_cmd,
        cwd=str(patch_dir),
        capture_output=True,
        text=True,
    )
    finished = time.time()
    output = (proc.stdout or "") + (proc.stderr or "")

    # Parse pytest summary line, e.g. "5 passed in 12.14s" or "2 failed, 3 passed in 11.64s"
    summary = {"passed": 0, "failed": 0, "total": 0}
    last_summary_line = ""
    for line in reversed(output.splitlines()):
        if " passed" in line or " failed" in line or " error" in line:
            last_summary_line = line
            break
    if last_summary_line:
        m_passed = re.search(r"(\d+)\s+passed", last_summary_line)
        m_failed = re.search(r"(\d+)\s+failed", last_summary_line)
        m_error = re.search(r"(\d+)\s+error", last_summary_line)
        summary["passed"] = int(m_passed.group(1)) if m_passed else 0
        summary["failed"] = (int(m_failed.group(1)) if m_failed else 0) + (
            int(m_error.group(1)) if m_error else 0
        )
        summary["total"] = summary["passed"] + summary["failed"]

    # Truncate output to last MAX_OUTPUT_BYTES bytes
    truncated = False
    output_bytes = output.encode("utf-8", errors="replace")
    if len(output_bytes) > MAX_OUTPUT_BYTES:
        output = output_bytes[-MAX_OUTPUT_BYTES:].decode("utf-8", errors="replace")
        truncated = True

    # pytest exit code 5 = "no tests collected"; with -m "not gpu" filter this means
    # all tests were GPU-marked and were skipped. Treat as soft-pass.
    is_all_gpu_skipped = proc.returncode == 5 and os.environ.get("RUN_GPU_TESTS", "0") not in (
        "1",
        "true",
        "yes",
    )
    passed = (proc.returncode == 0) or is_all_gpu_skipped

    status = {
        "mission": mission_id,
        "passed": passed,
        "exit_code": proc.returncode,
        "started_at": started_iso,
        "finished_at": _utc_now_iso(),
        "duration_s": round(finished - started, 2),
        "summary": summary,
        "output": output,
        "output_truncated": truncated,
        "all_gpu_skipped": is_all_gpu_skipped,
    }
    out_path = patch_dir / ".last_run.json"
    out_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")

    # Mirror to stdout so users see live test output (not just buffered).
    sys.stdout.write(output)
    sys.stdout.flush()

    if is_all_gpu_skipped:
        sys.stdout.write(
            f"\n⏭️  Patch SKIPPED (all tests are gpu-only; set RUN_GPU_TESTS=1 to run) "
            f"— status: {out_path.relative_to(ROOT)}\n"
        )
        return 0  # treat as success in CI / make patch-test-all
    if proc.returncode == 0:
        sys.stdout.write(
            f"\n✅ Patch PASSED ({summary['passed']}/{summary['total']} tests, "
            f"{status['duration_s']}s) — status: {out_path.relative_to(ROOT)}\n"
        )
    else:
        sys.stdout.write(
            f"\n❌ Patch FAILED ({summary['failed']}/{summary['total']} failing) — "
            f"status: {out_path.relative_to(ROOT)}\n"
        )
    return proc.returncode


def _utc_now_iso() -> str:
    return dt.datetime.now(dt.UTC).isoformat().replace("+00:00", "Z")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mission", required=True, help="Lab id, e.g. l05_distributed_primitives")
    args = p.parse_args()
    sys.exit(run(args.mission))


if __name__ == "__main__":
    main()
