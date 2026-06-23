from pathlib import Path

# Definición dinámica de la raíz del proyecto
PROJECT_ROOT = Path(__file__).resolve().parent

# Rutas principales adaptadas al clúster (beegfs)
DATA_RAW = Path("/mnt/beegfs/labs/pd1/MRR/Proyecto_MRR_UOH/data/raw/NetCDF")
DATA_OUT = PROJECT_ROOT / "results" / "isoterma_plots"
LOGS_DIR = PROJECT_ROOT / "logs"

# Ruta específica para las pruebas del método de gradiente (Ventanas para W)
DATA_EXP_W = PROJECT_ROOT / "results" / "experimentos_w"

# Años de análisis
AÑOS_VALIDOS = [2022, 2023, 2024, 2025, 2026]

# Creación automática de directorios si no existen
DATA_OUT.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
DATA_EXP_W.mkdir(parents=True, exist_ok=True)
