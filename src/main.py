import sys
from pathlib import Path
import numpy as np

# Configuración de entorno
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from config import AÑOS_VALIDOS, DATA_OUT
from src.data.loader import obtener_archivos_por_año, leer_netcdf
from src.features.isoterma import calcular_gradiente, aplicar_filtro_kalman, filtrar_ruido
from src.visualization.plots import generar_plot_isoterma

def run():
    print("Iniciando procesamiento secuencial de isotermas...")
    
    for año in AÑOS_VALIDOS:
        # 1. Obtener todos los archivos del año actual
        archivos = obtener_archivos_por_año(año)
        output_dir = DATA_OUT / str(año)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"--- Procesando Año {año}: {len(archivos)} archivos ---")
        
        for i, ruta in enumerate(archivos, 1):
            nombre = ruta.stem
            img_path = output_dir / f"Isoterma_{nombre}.png"
            
            # Saltar si el plot ya fue generado anteriormente
            if img_path.exists():
                continue
            
            try:
                with leer_netcdf(ruta) as ds:
                    # Preparación de datos
                    ze_raw = ds['attenuated_radar_reflectivity'].values
                    vf_raw = ds['fall_velocity'].values
                    
                    # Filtro de ruido (Referencia: >12 dBZ)
                    ze, _ = filtrar_ruido(ze_raw, vf_raw)
                    
                    # Cálculo de gradientes y filtro de Kalman
                    grad = calcular_gradiente(ze.T)
                    heights_msnm = ds.height[0,:] + 500
                    iso_inicial = np.full(ze.shape[0], 2500.0)
                    
                    iso_final = aplicar_filtro_kalman(iso_inicial, grad, heights_msnm)
                    
                    # Generación del gráfico individual
                    generar_plot_isoterma(ds, iso_final, img_path, nombre)
                    
                    if i % 10 == 0:
                        print(f"[{año}] Progreso: {i}/{len(archivos)} archivos finalizados.")
                        
            except Exception as e:
                print(f"Error en archivo {nombre} ({año}): {e}")

    print("Procesamiento masivo completado con éxito.")

if __name__ == "__main__":
    run()
