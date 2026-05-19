import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from config import DATA_RAW
from src.data.loader import obtener_archivos_por_año, leer_netcdf
from src.features.isoterma import procesar_dia_completo
from src.visualization.plots import generar_grafico_maestro

def ejecutar_test_limpio():
    print("Ejecutando Dry Run sin dependencias obsoletas...")
    archivos = obtener_archivos_por_año(2023, DATA_RAW)
    if not archivos: return

    try:
        with leer_netcdf(archivos[0]) as ds:
            times = pd.to_datetime(ds.time.values)
            xlim = [times[0], times[-1]]
            
            h_vals = ds.height.values
            heights_raw = h_vals[0, :] if h_vals.ndim > 1 else h_vals
            
            Ze_raw = ds['attenuated_radar_reflectivity'].values
            Vf_raw = ds['fall_velocity'].values
            
            isoterma_data = procesar_dia_completo(Ze_raw, Vf_raw, heights_raw, len(times))
            generar_grafico_maestro(xlim, times, heights_raw, Ze_raw, Vf_raw, isoterma_data, "test_result.png")
            print("--- ÉXITO TOTAL: Imagen generada con estética UOH perfecta ---")
    except Exception as e:
        print(f"Fallo crítico en el test: {e}")

if __name__ == "__main__":
    ejecutar_test_limpio()
