import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from config import DATA_RAW
from src.data.loader import obtener_archivos_por_año, leer_netcdf
from src.features.isoterma import calcular_gradiente, aplicar_filtro_kalman
from src.visualization.plots import generar_plot_isoterma

def test_single_file():
    print("Iniciando prueba de consistencia...")
    archivos = obtener_archivos_por_año(2023)
    if not archivos:
        print("Error: No se encontraron archivos en la ruta configurada.")
        return

    test_file = archivos[0]
    output_test = PROJECT_ROOT / "test_result.png"
    
    try:
        with leer_netcdf(test_file) as ds:
            ze = ds['attenuated_radar_reflectivity'].values.T
            grad = calcular_gradiente(ze)
            heights = ds.height[0,:] + 500
            iso_inicial = np.full(ze.shape[1], 2500.0)
            iso_final = aplicar_filtro_kalman(iso_inicial, grad, heights)
            generar_plot_isoterma(ds, iso_final, output_test, "TEST_FILE")
            print(f"Prueba exitosa. Imagen generada en: {output_test}")
    except Exception as e:
        print(f"Fallo en la prueba: {e}")

if __name__ == "__main__":
    test_single_file()
