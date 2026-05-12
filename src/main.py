import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from config import AÑOS_VALIDOS, DATA_RAW, DATA_OUT
from src.data.loader import obtener_archivos_por_año, leer_netcdf
from src.features.isoterma import calcular_gradiente, aplicar_filtro_kalman, filtrar_ruido
from src.visualization.plots import generar_plot_isoterma

def run():
    print("Iniciando procesamiento universal...")
    for año in AÑOS_VALIDOS:
        archivos = obtener_archivos_por_año(año, DATA_RAW)
        output_dir = DATA_OUT / str(año)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for ruta in archivos:
            nombre = ruta.stem
            img_path = output_dir / f"Isoterma_{nombre}.png"
            if img_path.exists(): continue
                
            try:
                with leer_netcdf(ruta) as ds:
                    # Extracción robusta de variables
                    ze_raw = ds['attenuated_radar_reflectivity'].values
                    vf_raw = ds['fall_velocity'].values
                    
                    # Manejo dinámico de alturas (msnm + 500)
                    if ds.height.ndim > 1:
                        heights_msnm = ds.height[0,:].values + 500
                    else:
                        heights_msnm = ds.height.values + 500
                    
                    # Asegurar que Ze esté orientado correctamente (Vertical, Tiempo)
                    # Si ze_raw es (Time, Height), lo transponemos
                    if ze_raw.shape[0] == len(ds.time):
                        ze_proc = ze_raw.T
                    else:
                        ze_proc = ze_raw

                    # 1. Filtro de Ruido
                    ze_clean, _ = filtrar_ruido(ze_proc.T, vf_raw.T)
                    
                    # 2. Gradiente (usa Ze transpuesto para la lógica de niveles)
                    grad = calcular_gradiente(ze_clean.T)
                    
                    # 3. Kalman
                    iso_inicial = np.full(ze_clean.shape[0], 2500.0)
                    iso_final = aplicar_filtro_kalman(iso_inicial, grad, heights_msnm)
                    
                    # 4. Plot
                    generar_plot_isoterma(ds, iso_final, img_path, nombre)
                    print(f"Éxito: {nombre}")
            except Exception as e:
                print(f"Fallo en {nombre}: {e}")

if __name__ == "__main__":
    run()
