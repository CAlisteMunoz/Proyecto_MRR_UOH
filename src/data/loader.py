import xarray as xr
from pathlib import Path

def obtener_archivos_por_año(año, ruta_base):
    ruta_año = ruta_base / str(año)
    if not ruta_año.exists():
        return []
    return sorted(list(ruta_año.glob("*.nc")))

def leer_netcdf(ruta):
    # Abrimos y forzamos la carga en memoria para evitar errores de puntero
    return xr.open_dataset(ruta, decode_times=False).load()
