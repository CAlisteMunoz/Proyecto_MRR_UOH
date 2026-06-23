import sys
from pathlib import Path
import numpy as np
from config import AÑOS_VALIDOS, DATA_RAW, DATA_EXP_W
from src.data.loader import obtener_archivos_por_año, leer_netcdf
from src.features.isoterma import calcular_gradiente_avanzado, aplicar_filtro_kalman, filtrar_ruido
from src.visualization.plots import generar_plot_mrr_dual

# Batería de pruebas a ejecutar
CONFIGURACIONES_PRUEBA = {
    "1_Lineal_Clasica":      {"ventana": "lineal",    "sigma": None, "umbral_w": None},
    "2_Uniforme_Promedio":   {"ventana": "uniforme",  "sigma": None, "umbral_w": None},
    "3_Gaussiana_Estandar":  {"ventana": "gaussiana", "sigma": 2.0,  "umbral_w": None},
    "4_Gaussiana_Estricta":  {"ventana": "gaussiana", "sigma": 1.0,  "umbral_w": None},
    "5_Gaussiana_FiltroCinematico": {"ventana": "gaussiana", "sigma": 2.0, "umbral_w": 2.5}
}

def ejecutar_experimentos():
    print(f"--- Iniciando Batería de Experimentos de Velocidad (W) ---")
    
    # Se extraen datos representativos (Ej. año 2023)
    archivos = obtener_archivos_por_año(2023, DATA_RAW) 
    if not archivos:
        print("No se encontraron archivos en DATA_RAW para el año especificado.")
        return
        
    # Limitamos a 3 archivos para una evaluación rápida de los contrastes
    archivos_prueba = archivos[:3] 
    
    for nombre_exp, params in CONFIGURACIONES_PRUEBA.items():
        print(f"\n>> Ejecutando configuración: {nombre_exp}")
        
        out_dir = DATA_EXP_W / nombre_exp
        out_dir.mkdir(parents=True, exist_ok=True)
        
        for ruta in archivos_prueba:
            nombre = ruta.stem
            img_path = out_dir / f"Exp_{nombre_exp}_{nombre}.png"
            
            if img_path.exists(): 
                print(f"   [SALTADO] Ya existe: {img_path.name}")
                continue
            
            try:
                with leer_netcdf(ruta) as ds:
                    h = (ds.height.values[0,:] if ds.height.ndim > 1 else ds.height.values) + 500
                    ze_raw, vf_raw = ds['attenuated_radar_reflectivity'].values, ds['fall_velocity'].values
                    
                    ze_c, vf_c = filtrar_ruido(ze_raw.T if ze_raw.shape[0] == len(ds.time) else ze_raw, 
                                               vf_raw.T if vf_raw.shape[0] == len(ds.time) else vf_raw)
                    
                    # Ze se mantiene estándar (lineal) para no distorsionar la prueba cruzada
                    grad_ze = calcular_gradiente_avanzado(ze_c, marco=5, tipo_ventana='lineal')
                    
                    # W es procesada con la configuración dinámica del ciclo actual
                    grad_vf = calcular_gradiente_avanzado(
                        vf_c, 
                        marco=7, 
                        tipo_ventana=params["ventana"], 
                        sigma=params["sigma"], 
                        umbral_min_w=params["umbral_w"]
                    )
                    
                    iso_z, var_z = aplicar_filtro_kalman(np.full(ze_c.shape[1], 2500.0), grad_ze, h)
                    iso_v, var_v = aplicar_filtro_kalman(np.full(vf_c.shape[1], 2500.0), grad_vf, h)
                    
                    # Construcción del título para la diapositiva
                    titulo_exp = f"{nombre_exp} | Tipo: {params['ventana'].upper()}"
                    if params['umbral_w']: titulo_exp += f" | Umbral W: {params['umbral_w']} m/s"
                    
                    generar_plot_mrr_dual(ds, ze_c, vf_c, iso_z, var_z, iso_v, var_v, img_path, titulo_exp)
                    print(f"   [OK] Guardado: {img_path.name}")
                    
            except Exception as e:
                print(f"   [ERROR] Fallo en {nombre}: {e}")

if __name__ == "__main__":
    ejecutar_experimentos()
