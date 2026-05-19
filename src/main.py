import sys
from pathlib import Path
import numpy as np
import pandas as pd
from config import AÑOS_VALIDOS, DATA_RAW, DATA_OUT
from src.data.loader import obtener_archivos_por_año, leer_netcdf
from src.features.isoterma import procesar_dia_completo
from src.visualization.plots import generar_grafico_maestro

def run():
    print("Iniciando pipeline global estructurado...")
    for año in AÑOS_VALIDOS:
        archivos = obtener_archivos_por_año(año, DATA_RAW)
        out_dir = DATA_OUT / str(año)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        for ruta in archivos:
            nombre = ruta.stem
            img_path = out_dir / f"Isoterma_{nombre}.png"
            if img_path.exists(): continue
            
            try:
                with leer_netcdf(ruta) as ds:
                    times = pd.to_datetime(ds.time.values)
                    xlim = [times[0], times[-1]]
                    
                    h_vals = ds.height.values
                    heights_raw = h_vals[0, :] if h_vals.ndim > 1 else h_vals
                    
                    Ze_raw = ds['attenuated_radar_reflectivity'].values
                    Vf_raw = ds['fall_velocity'].values
                    
                    isoterma_data = procesar_dia_completo(Ze_raw, Vf_raw, heights_raw, len(times))
                    generar_grafico_maestro(xlim, times, heights_raw, Ze_raw, Vf_raw, isoterma_data, img_path)
                    print(f"Completado con éxito: {nombre}")
            except Exception as e:
                print(f"Error en {nombre}: {e}")

if __name__ == "__main__":
    run()
