from __future__ import annotations

import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).resolve().parent / "run_capstone_aggregator.py"),
        run_name="__main__",
    )
