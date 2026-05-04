from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    root_script = (
        Path(__file__).resolve().parents[3] / "scripts" / "env" / "check_cuda.py"
    )
    runpy.run_path(str(root_script), run_name="__main__")
