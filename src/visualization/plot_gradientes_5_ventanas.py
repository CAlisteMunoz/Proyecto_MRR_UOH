import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from config import DATA_RAW, PROJECT_ROOT
from src.data.loader import obtener_archivos_por_año, leer_netcdf
from src.features.isoterma import calcular_gradiente_avanzado, filtrar_ruido

CONFIGURACIONES = {
    "1_Original_Proyecto": {"ventana": "lineal",    "marco": 1,  "sigma": None},
    "2_Lineal_Normal":     {"ventana": "lineal",    "marco": 7,  "sigma": None},
    "3_Gaussiana_Normal":  {"ventana": "gaussiana", "marco": 7,  "sigma": 2.0},
    "4_Lineal_Amplia":     {"ventana": "lineal",    "marco": 15, "sigma": None},
    "5_Gaussiana_Amplia":  {"ventana": "gaussiana", "marco": 15, "sigma": 5.0}
}

def generar_plot_5_gradientes(ds, resultados_gradientes, ruta_salida, nombre_evento, h_arr):
    tiempos_raw = pd.to_datetime(ds.time.values)
    
    # Parámetros estrictos extraídos de la replicación
    extent = [mdates.date2num(tiempos_raw[0]), mdates.date2num(tiempos_raw[-1]), h_arr[0], h_arr[-1]]
    ylim = (0, 3600) if h_arr[-1] < 5000 else (0, 8000)

    fig, axes = plt.subplots(nrows=5, figsize=(14, 18), sharex=True)
    fig.suptitle(f"Replicación de Gradientes ($\\nabla W$) - Formato Original\nEvento: {nombre_evento}", 
                 fontsize=16, fontweight='bold', y=0.92)

    # Replicación del mapa original para Vf
    cmap = plt.get_cmap('RdBu').copy()
    cmap.set_bad('0.9', 1) 

    for ax, res in zip(axes, resultados_gradientes):
        im = ax.imshow(res['gradiente'], origin='lower', aspect='auto', extent=extent, 
                       cmap=cmap, vmin=-3, vmax=10)
        
        ax.set_title(f"Configuración: {res['nombre']}", fontsize=12, loc='left')
        ax.set_ylabel("Altitud [msnm]", fontsize=10)
        ax.set_ylim(ylim)
        ax.set_facecolor('0.9') 
        ax.grid(True, linestyle='--', alpha=0.3)

    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    axes[-1].xaxis.set_major_locator(locator)
    axes[-1].xaxis.set_major_formatter(formatter)
    axes[-1].set_xlabel(r'Hora UTC $\rightarrow$', fontsize=12)

    plt.subplots_adjust(left=0.08, right=0.88, top=0.89, bottom=0.06, hspace=0.25)

    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax, extend='both')
    cbar.set_label("[m/s]", fontsize=12)

    plt.savefig(ruta_salida, dpi=150, bbox_inches='tight')
    plt.close(fig)

def ejecutar_comparacion_gradientes():
    print("=== INICIANDO REPLICACIÓN EXACTA DE GRADIENTES ===")
    archivos = obtener_archivos_por_año(2023, DATA_RAW)
    
    archivos_validos = []
    for ruta in archivos:
        try:
            with leer_netcdf(ruta) as ds:
                if np.sum(ds['attenuated_radar_reflectivity'].values > 20.0) > 100:
                    archivos_validos.append(ruta)
                if len(archivos_validos) == 10:
                    break
        except Exception: pass

    out_dir = PROJECT_ROOT / "results" / "gradientes"
    out_dir.mkdir(parents=True, exist_ok=True)

    for ruta_muestra in archivos_validos:
        nombre = ruta_muestra.stem
        print(f"\n>> Analizando: {nombre}...")

        try:
            with leer_netcdf(ruta_muestra) as ds:
                # BLINDAJE CONTRA TUPLES: Forzamos np.asarray()
                ze_raw = np.asarray(ds['attenuated_radar_reflectivity'].values)
                vf_raw = np.asarray(ds['fall_velocity'].values)
                
                # Extracción de altura y desfase
                h_vals = np.asarray(ds.height.values[0,:] if ds.height.ndim > 1 else ds.height.values)
                altura_inicial_desfase = 500 + (h_vals[1] - h_vals[0]) / 2
                h_ajustado = h_vals + altura_inicial_desfase

                # Transposición controlada y blindada 
                ze_t = np.asarray(ze_raw.T if ze_raw.shape[0] == len(ds.time) else ze_raw)
                vf_t = np.asarray(vf_raw.T if vf_raw.shape[0] == len(ds.time) else vf_raw)

                ze_c, vf_c = filtrar_ruido(ze_t, vf_t)

                resultados = []
                for nombre_config, params in CONFIGURACIONES.items():
                    grad_vf = calcular_gradiente_avanzado(
                        vf_c, 
                        marco=params["marco"], 
                        tipo_ventana=params["ventana"], 
                        sigma=params["sigma"]
                    )
                    resultados.append({"nombre": nombre_config, "gradiente": grad_vf})

                out_file = out_dir / f"Comparacion_Gradientes_W_{nombre}.png"
                generar_plot_5_gradientes(ds, resultados, out_file, nombre, h_ajustado)
                print(f"   [OK] Lámina comparativa replicada.")
                
        except Exception as e:
            # Ahora capturará y te mostrará cualquier error, en lugar de crashear el script entero
            print(f"   [ERROR] Falló procesar el archivo {nombre}: {e}")

if __name__ == '__main__':
    ejecutar_comparacion_gradientes()
