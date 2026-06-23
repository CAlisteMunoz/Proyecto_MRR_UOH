import sys
from pathlib import Path
import numpy as np
from config import AÑOS_VALIDOS, DATA_RAW, PROJECT_ROOT
from src.data.loader import obtener_archivos_por_año, leer_netcdf
from src.features.isoterma import calcular_gradiente_avanzado, aplicar_filtro_kalman, filtrar_ruido
from src.visualization.plots import generar_plot_mrr_dual

# --- RUTA MAESTRA PARA ESTE TEST ---
TEST_DIR = PROJECT_ROOT / "results" / "test_ventanas_w"
TEST_DIR.mkdir(parents=True, exist_ok=True)

# --- LAS 5 CONFIGURACIONES A PROBAR ---
CONFIGURACIONES = {
    "1_Lineal_Clasica":      {"ventana": "lineal",    "sigma": None, "umbral_w": None},
    "2_Uniforme_Promedio":   {"ventana": "uniforme",  "sigma": None, "umbral_w": None},
    "3_Gaussiana_Estandar":  {"ventana": "gaussiana", "sigma": 2.0,  "umbral_w": None},
    "4_Gaussiana_Estricta":  {"ventana": "gaussiana", "sigma": 1.0,  "umbral_w": None},
    "5_Gaussiana_FiltroCinematico": {"ventana": "gaussiana", "sigma": 2.0, "umbral_w": 2.5}
}

def buscar_archivos_con_lluvia(archivos, cantidad_requerida=5):
    """
    Escanea los NetCDF y selecciona únicamente aquellos que contengan lluvia real.
    """
    archivos_validos = []
    print(f"-> Escaneando archivos en búsqueda de {cantidad_requerida} eventos con lluvia fuerte...")
    
    for ruta in archivos:
        try:
            with leer_netcdf(ruta) as ds:
                ze_raw = ds['attenuated_radar_reflectivity'].values
                if np.sum(ze_raw > 20.0) > 100:
                    archivos_validos.append(ruta)
                    print(f"   [+] Evento válido encontrado: {ruta.stem}")
                
                if len(archivos_validos) == cantidad_requerida:
                    break
        except Exception:
            continue
            
    return archivos_validos

def ejecutar_test_comparativo():
    print("=== INICIANDO TEST DE VENTANAS PARA VELOCIDAD (W) ===")
    
    archivos_totales = obtener_archivos_por_año(2023, DATA_RAW)
    if not archivos_totales:
        print("Error: No se encontraron archivos NetCDF.")
        return
        
    archivos_prueba = buscar_archivos_con_lluvia(archivos_totales, cantidad_requerida=5)
    
    if len(archivos_prueba) < 5:
        print(f"Advertencia: Solo se encontraron {len(archivos_prueba)} archivos con lluvia válida.")
        
    for ruta in archivos_prueba:
        nombre_evento = ruta.stem
        print(f"\n>> Procesando Evento: {nombre_evento}")
        
        evento_dir = TEST_DIR / nombre_evento
        evento_dir.mkdir(parents=True, exist_ok=True)
        
        for nombre_config, params in CONFIGURACIONES.items():
            img_path = evento_dir / f"{nombre_config}.png"
            if img_path.exists(): continue
            
            try:
                with leer_netcdf(ruta) as ds:
                    h = (ds.height.values[0,:] if ds.height.ndim > 1 else ds.height.values) + 500
                    ze_raw = ds['attenuated_radar_reflectivity'].values
                    vf_raw = ds['fall_velocity'].values
                    
                    ze_c, vf_c = filtrar_ruido(ze_raw.T if ze_raw.shape[0] == len(ds.time) else ze_raw, 
                                               vf_raw.T if vf_raw.shape[0] == len(ds.time) else vf_raw)
                    
                    grad_ze = calcular_gradiente_avanzado(ze_c, marco=5, tipo_ventana='lineal')
                    
                    grad_vf = calcular_gradiente_avanzado(
                        vf_c, 
                        marco=7, 
                        tipo_ventana=params["ventana"], 
                        sigma=params["sigma"], 
                        umbral_min_w=params["umbral_w"]
                    )
                    
                    iso_z, var_z = aplicar_filtro_kalman(np.full(ze_c.shape[1], 2500.0), grad_ze, h)
                    iso_v, var_v = aplicar_filtro_kalman(np.full(vf_c.shape[1], 2500.0), grad_vf, h)
                    
                    titulo_plot = f"Evento: {nombre_evento} | Config: {nombre_config}"
                    if params['umbral_w']: 
                        titulo_plot += f" (Umbral > {params['umbral_w']} m/s)"
                    
                    generar_plot_mrr_dual(ds, ze_c, vf_c, iso_z, var_z, iso_v, var_v, img_path, titulo_plot)
                    print(f"   [OK] Generado -> {img_path.name}")
                    
            except Exception as e:
                print(f"   [ERROR] Falló config {nombre_config} en {nombre_evento}: {e}")

if __name__ == "__main__":
    ejecutar_test_comparativo()
