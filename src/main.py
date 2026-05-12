import sys
from pathlib import Path
import numpy as np
from config import AÑOS_VALIDOS, DATA_RAW, DATA_OUT
from src.data.loader import obtener_archivos_por_año, leer_netcdf
from src.features.isoterma import calcular_gradiente, aplicar_filtro_kalman, filtrar_ruido
from src.visualization.plots import generar_plot_mrr_dual

def run():
    for año in AÑOS_VALIDOS:
        archivos = obtener_archivos_por_año(año, DATA_RAW)
        out = DATA_OUT / str(año)
        out.mkdir(parents=True, exist_ok=True)
        
        for ruta in archivos:
            nombre = ruta.stem
            img_path = out / f"Isoterma_{nombre}.png"
            if img_path.exists(): continue
            
            try:
                with leer_netcdf(ruta) as ds:
                    h = (ds.height.values[0,:] if ds.height.ndim > 1 else ds.height.values) + 500
                    ze_raw, vf_raw = ds['attenuated_radar_reflectivity'].values, ds['fall_velocity'].values
                    
                    # Preparar datos y gradientes
                    # Nota: Aseguramos orientación (Niveles, Tiempo)
                    ze_c, vf_c = filtrar_ruido(ze_raw.T if ze_raw.shape[0] == len(ds.time) else ze_raw, 
                                               vf_raw.T if vf_raw.shape[0] == len(ds.time) else vf_raw)
                    
                    grad_ze = calcular_gradiente(ze_c)
                    grad_vf = calcular_gradiente(vf_c)
                    
                    # Calcular isotermas para ambos paneles
                    iso_z, var_z = aplicar_filtro_kalman(np.full(ze_c.shape[1], 2500.0), grad_ze, h)
                    iso_v, var_v = aplicar_filtro_kalman(np.full(vf_c.shape[1], 2500.0), grad_vf, h)
                    
                    generar_plot_mrr_dual(ds, iso_z, var_z, iso_v, var_v, img_path, nombre)
                    print(f"Completado: {nombre}")
            except Exception as e:
                print(f"Error en {nombre}: {e}")

if __name__ == "__main__":
    run()
