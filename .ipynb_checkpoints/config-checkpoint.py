# config.py
import os
from pathlib import Path

# Ruta base del proyecto
PROJECT_ROOT = Path(__file__).parent

# Rutas a tus datos NetCDF ya copiados
DATA_RAW = PROJECT_ROOT / "data" / "raw" / "NetCDF"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

# Crear directorios si no existen
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
