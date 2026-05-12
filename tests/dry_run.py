import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from config import DATA_RAW
from src.data.loader import obtener_archivos_por_año, leer_netcdf
from src.features.isoterma import calcular_gradiente, aplicar_filtro_kalman
from src.visualization.plots import generar_plot_isoterma

def test_final():
    print("Iniciando prueba de consistencia final (Estilo .txt)...")
    archivos = obtener_archivos_por_año(2023, DATA_RAW)
    if not archivos:
        print("No se encontraron archivos para el test.")
        return

    ruta = archivos[0]
    try:
        with leer_netcdf(ruta) as ds:
            # Réplica de dimensiones del .txt
            ze_raw = ds['attenuated_radar_reflectivity'].values
            times = ds.time.values
            
            # Altura msnm (Fuente 88 del .txt)
            if ds.height.ndim > 1:
                h = ds.height.values[0,:] + 500
            else:
                h = ds.height.values + 500
            
            # Orientación automática
            ze = ze_raw.T if ze_raw.shape[0] == len(times) else ze_raw
            
            # Gradiente y Kalman
            grad = calcular_gradiente(ze)
            iso_inicial = np.full(ze.shape[1], 2500.0)
            iso_final = aplicar_filtro_kalman(iso_inicial, grad, h)
            
            generar_plot_isoterma(ds, iso_final, "test_result.png", "DRY_RUN_OK")
            print("--- ÉXITO: Imagen 'test_result.png' generada correctamente ---")
    except Exception as e:
        print(f"Fallo en la prueba: {e}")

if __name__ == "__main__":
    test_final()
