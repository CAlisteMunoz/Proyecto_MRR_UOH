import sys
from pathlib import Path
import warnings

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import xarray as xr

# Silenciar advertencias irrelevantes de xarray
warnings.filterwarnings("ignore")

from config import DATA_RAW, PROJECT_ROOT
from src.data.loader import obtener_archivos_por_año, leer_netcdf

# === CONFIGURACIONES MODERADAS Y CLÁSICAS ===
CONFIGURACIONES = {
    "1_Lineal_Estrecha":   {"ventana": "lineal",    "marco": 3,  "sigma": None},
    "2_Lineal_Normal":     {"ventana": "lineal",    "marco": 5,  "sigma": None},
    "3_Gaussiana_Normal":  {"ventana": "gaussiana", "marco": 5,  "sigma": 1.5},
    "4_Lineal_Amplia":     {"ventana": "lineal",    "marco": 7,  "sigma": None},
    "5_Gaussiana_Amplia":  {"ventana": "gaussiana", "marco": 7,  "sigma": 2.0}
}

def filtrar_ruido(ze, vf, umbral_ze=12.0):
    # Uso invisible de Ze: Todo lo menor a 12 dBZ se vuelve NaN (gris en el plot)
    vf_f = np.where(ze >= umbral_ze, vf, np.nan)
    return vf_f

def calcular_gradiente_fisico(datos, marco=3, tipo_ventana='lineal', sigma=2.0):
    if tipo_ventana == 'uniforme': pesos = np.ones(marco)
    elif tipo_ventana == 'lineal': pesos = np.array([marco - i for i in range(marco)])
    elif tipo_ventana == 'gaussiana': pesos = np.exp(-0.5 * (np.arange(marco) / sigma)**2)
    elif tipo_ventana == 'hamming': pesos = np.hamming(marco)
    elif tipo_ventana == 'hanning': pesos = np.hanning(marco)
    else: raise ValueError("Ventana inválida")
        
    pesos = pesos / np.sum(pesos)
    gradiente_datos = np.full_like(datos, np.nan)
    
    for i in range(marco, datos.shape[0] - marco):
        sup = np.nansum([pesos[j] * datos[i + j, :] for j in range(marco)], axis=0)
        inf = np.nansum([pesos[j] * datos[i - j - 1, :] for j in range(marco)], axis=0)
        
        # FÍSICA CLIMATOLÓGICA REAL: Aceleración positiva (Lluvia rápida abajo - Nieve lenta arriba)
        gradiente_datos[i, :] = inf - sup

    return np.where(np.isnan(datos), np.nan, gradiente_datos)

def generar_plot_5_gradientes(ds, resultados_gradientes, ruta_salida, nombre_evento, h_arr):
    try:
        tiempos_raw = pd.to_datetime(ds.time.values)
    except:
        tiempos_raw = pd.to_datetime(ds.indexes['time'].astype(str))
        
    extent = [mdates.date2num(tiempos_raw[0]), mdates.date2num(tiempos_raw[-1]), h_arr[0], h_arr[-1]]
    ylim = [0, 3600] if h_arr[-1] < 5000 else [0, 8000]

    fig, axes = plt.subplots(nrows=5, figsize=(14, 18), sharex=True)
    fig.suptitle(f"Gradiente Físico de Velocidad de Caída ($\\nabla W$)\nEvento: {nombre_evento}", 
                 fontsize=16, fontweight='bold', y=0.92)

    # Replicación milimétrica del colormap y parámetros originales
    cmap = plt.get_cmap('RdBu').copy()
    cmap.set_bad('0.9', 1) 

    for ax, res in zip(axes, resultados_gradientes):
        # Al ser gradientes físicos (0 a ~6), RdBu pintará el 0 casi blanco y el pico en azul fuerte.
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
    cbar.set_label("Aceleración de Partículas [m/s]", fontsize=12)

    plt.savefig(ruta_salida, dpi=150, bbox_inches='tight')
    plt.close(fig)

def ejecutar_comparacion_gradientes():
    print("=== INICIANDO PROCESAMIENTO: FÍSICA REAL Y VENTANAS MODERADAS ===")
    archivos = obtener_archivos_por_año(2023, DATA_RAW)
    
    out_dir = PROJECT_ROOT / "results" / "gradientes"
    out_dir.mkdir(parents=True, exist_ok=True)

    eventos_procesados = 0
    for ruta_muestra in archivos:
        if eventos_procesados >= 10:
            break

        nombre = ruta_muestra.stem
        try:
            with leer_netcdf(ruta_muestra) as ds:
                # Lectura de datos blindada
                ze_raw = np.asarray(ds['attenuated_radar_reflectivity'].values)
                if np.sum(ze_raw > 20.0) < 100:
                    continue
                
                print(f"\n>> Analizando: {nombre}...")
                vf_raw = np.asarray(ds['fall_velocity'].values)
                
                h_vals = np.asarray(ds.height.values[0,:] if ds.height.ndim > 1 else ds.height.values)
                altura_inicial_desfase = 500 + (h_vals[1] - h_vals[0]) / 2
                h_ajustado = h_vals + altura_inicial_desfase

                ze_t = np.asarray(ze_raw.T if ze_raw.shape[0] == len(ds.time) else ze_raw)
                vf_t = np.asarray(vf_raw.T if vf_raw.shape[0] == len(ds.time) else vf_raw)

                # Filtrar ruido usando Ze como máscara
                vf_c = filtrar_ruido(ze_t, vf_t)

                resultados = []
                for nombre_config, params in CONFIGURACIONES.items():
                    grad_vf = calcular_gradiente_fisico(
                        vf_c, 
                        marco=params["marco"], 
                        tipo_ventana=params["ventana"], 
                        sigma=params["sigma"]
                    )
                    resultados.append({"nombre": nombre_config, "gradiente": grad_vf})

                out_file = out_dir / f"Comparacion_Gradientes_W_{nombre}.png"
                generar_plot_5_gradientes(ds, resultados, out_file, nombre, h_ajustado)
                print(f"   [OK] Lámina de gradientes físicos generada a la perfección.")
                
                eventos_procesados += 1
                
        except Exception as e:
            print(f"   [ERROR] Falló procesar el archivo {nombre}: {e}")

    print(f"\n✅ ¡Lote limpio de {eventos_procesados} eventos finalizado exitosamente!")

if __name__ == '__main__':
    ejecutar_comparacion_gradientes()
