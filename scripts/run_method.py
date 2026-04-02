from __future__ import annotations

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
MPLCONFIGDIR = ROOT / ".cache" / "matplotlib"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
sys.path.insert(0, str(ROOT / "src"))

from lab.pipeline import run_stage_cli


if __name__ == "__main__":
    raise SystemExit(run_stage_cli("run_method"))
