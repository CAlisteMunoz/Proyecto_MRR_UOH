import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.absolute()
DATA_RAW = Path("/mnt/beegfs/labs/ESciences/data/MRR/NetCDF")
DATA_OUT = PROJECT_ROOT / "results" / "isoterma_plots"
LOGS_DIR = PROJECT_ROOT / "logs"

DATA_OUT.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

AÑOS_VALIDOS = [2022, 2023, 2024, 2025, 2026]
