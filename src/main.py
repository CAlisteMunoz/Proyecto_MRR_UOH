import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from config import AÑOS_VALIDOS, DATA_RAW, DATA_OUT
from src.data.loader import obtener_archivos_por_año, leer_netcdf
from src.features.isoterma import calcular_gradiente, aplicar_filtro_kalman
from src.visualization.plots import generar_plot_isoterma

def run():
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
                    ze = ds['attenuated_radar_reflectivity'].values.T
                    grad = calcular_gradiente(ze)
                    heights_msnm = ds.height[0,:] + 500
                    iso_obs_inicial = np.full(ze.shape[1], 2500.0) 
                    
                    iso_final = aplicar_filtro_kalman(iso_obs_inicial, grad, heights_msnm)
                    generar_plot_isoterma(ds, iso_final, img_path, nombre)
                    print(f"Completado: {nombre}")
            except Exception as e:
                print(f"Error en {nombre}: {e}")

if __name__ == "__main__":
    run()
