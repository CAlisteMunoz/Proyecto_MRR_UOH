import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from config import DATA_RAW
from src.data.loader import obtener_archivos_por_año, leer_netcdf
from src.features.isoterma import calcular_gradiente, aplicar_filtro_kalman
from src.visualization.plots import generar_plot_mrr_dual

def test_final():
    print("Iniciando prueba de consistencia con blindaje numérico...")
    archivos = obtener_archivos_por_año(2023, DATA_RAW)
    if not archivos: return

    ruta = archivos[0]
    try:
        with leer_netcdf(ruta) as ds:
            # Forzar conversión a numpy float64 para evitar errores de ufunc
            h = ((ds.height.values[0,:] if ds.height.ndim > 1 else ds.height.values) + 500).astype(np.float64)
            
            # Limpieza de datos Ze y Vf
            ze_raw = ds['attenuated_radar_reflectivity'].values.astype(np.float64)
            vf_raw = ds['fall_velocity'].values.astype(np.float64)
            
            ze = ze_raw.T if ze_raw.shape[0] == len(ds.time) else ze_raw
            vf = vf_raw.T if vf_raw.shape[0] == len(ds.time) else vf_raw
            
            grad_ze = calcular_gradiente(ze).astype(np.float64)
            grad_vf = calcular_gradiente(vf).astype(np.float64)
            
            iso_z, var_z = aplicar_filtro_kalman(np.full(ze.shape[1], 2500.0), grad_ze, h)
            iso_v, var_v = aplicar_filtro_kalman(np.full(vf.shape[1], 2500.0), grad_vf, h)
            
            generar_plot_mrr_dual(ds, iso_z, var_z, iso_v, var_v, "test_result.png", "DRY_RUN_NUMERIC_OK")
            print("--- ÉXITO TOTAL: Imagen generada sin errores de tipos ---")
    except Exception as e:
        print(f"Fallo en la prueba: {e}")

if __name__ == "__main__":
    test_final()
