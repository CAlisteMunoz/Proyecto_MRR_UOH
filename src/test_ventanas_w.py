import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from config import DATA_RAW, PROJECT_ROOT
from src.data.loader import obtener_archivos_por_año, leer_netcdf
from src.features.isoterma import calcular_gradiente_avanzado, aplicar_filtro_kalman, filtrar_ruido
from src.visualization.plots import generar_plot_5_ventanas_w

TEST_DIR = PROJECT_ROOT / "results" / "test_ventanas_w"
TEST_DIR.mkdir(parents=True, exist_ok=True)

# Las 5 ventanas canónicas de procesamiento de señales
CONFIGURACIONES = {
    "1_Lineal_Original":  {"ventana": "lineal",    "sigma": None, "umbral_w": None},
    "2_Uniforme":         {"ventana": "uniforme",  "sigma": None, "umbral_w": None},
    "3_Gaussiana":        {"ventana": "gaussiana", "sigma": 2.0,  "umbral_w": None},
    "4_Hamming":          {"ventana": "hamming",   "sigma": None, "umbral_w": None},
    "5_Hanning":          {"ventana": "hanning",   "sigma": None, "umbral_w": None}
}

def buscar_archivos_con_lluvia(archivos, cantidad_requerida=5):
    archivos_validos = []
    for ruta in archivos:
        try:
            with leer_netcdf(ruta) as ds:
                ze_raw = ds['attenuated_radar_reflectivity'].values
                if np.sum(ze_raw > 20.0) > 100:
                    archivos_validos.append(ruta)
                if len(archivos_validos) == cantidad_requerida:
                    break
        except Exception: pass
    return archivos_validos

def ejecutar_test_comparativo():
    print("=== INICIANDO GENERACIÓN DE PANELES MULTIPLES PARA W ===")
    
    archivos = buscar_archivos_con_lluvia(obtener_archivos_por_año(2023, DATA_RAW), 5)
    
    for ruta in archivos:
        nombre_evento = ruta.stem
        img_path = TEST_DIR / f"Panel_W_{nombre_evento}.png"
        print(f"\n>> Procesando Evento: {nombre_evento}")
        
        try:
            with leer_netcdf(ruta) as ds:
                h = (ds.height.values[0,:] if ds.height.ndim > 1 else ds.height.values) + 500
                ze_raw, vf_raw = ds['attenuated_radar_reflectivity'].values, ds['fall_velocity'].values
                
                ze_c, vf_c = filtrar_ruido(ze_raw.T if ze_raw.shape[0] == len(ds.time) else ze_raw, 
                                           vf_raw.T if vf_raw.shape[0] == len(ds.time) else vf_raw)
                
                resultados_evento = []
                
                for nombre_config, params in CONFIGURACIONES.items():
                    grad_vf = calcular_gradiente_avanzado(
                        vf_c, marco=7, tipo_ventana=params["ventana"], 
                        sigma=params["sigma"], umbral_min_w=params["umbral_w"]
                    )
                    
                    iso_v, var_v = aplicar_filtro_kalman(np.zeros(vf_c.shape[1]), grad_vf, h)
                    
                    resultados_evento.append({
                        "nombre": nombre_config,
                        "iso": iso_v,
                        "var": var_v
                    })
                
                generar_plot_5_ventanas_w(ds, vf_c, resultados_evento, img_path, nombre_evento)
                print(f"   [OK] Generado -> {img_path.name}")
                
        except Exception as e:
            print(f"   [ERROR] Falló procesamiento en {nombre_evento}: {e}")

if __name__ == "__main__":
    ejecutar_test_comparativo()
