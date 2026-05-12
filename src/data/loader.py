import xarray as xr
from pathlib import Path
import sys

# Asegurar que el cargador conozca la ruta raíz para importar config
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
from config import DATA_RAW

def obtener_archivos_por_año(año):
    # DATA_RAW ya es una ruta absoluta definida en config.py
    ruta_año = DATA_RAW / str(año)
    if not ruta_año.exists():
        return []
    return sorted(list(ruta_año.glob("*.nc")))

def leer_netcdf(ruta):
    # decode_times=False es vital para los formatos de tiempo del MRR
    return xr.open_dataset(ruta, decode_times=False)
