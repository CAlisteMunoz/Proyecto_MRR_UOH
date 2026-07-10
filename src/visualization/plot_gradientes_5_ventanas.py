import sys
from pathlib import Path
import warnings

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import dates
import proplot as pplt

warnings.filterwarnings("ignore")

from config import DATA_RAW, PROJECT_ROOT
from src.data.loader import obtener_archivos_por_año, leer_netcdf

# === CONFIGURACIONES DE VENTANAS (MODERADAS) ===
CONFIGURACIONES = {
    "1_Lineal_Estrecha":   {"ventana": "lineal",    "marco": 3,  "sigma": None},
    "2_Lineal_Normal":     {"ventana": "lineal",    "marco": 5,  "sigma": None},
    "3_Gaussiana_Normal":  {"ventana": "gaussiana", "marco": 5,  "sigma": 1.5},
    "4_Lineal_Amplia":     {"ventana": "lineal",    "marco": 7,  "sigma": None},
    "5_Gaussiana_Amplia":  {"ventana": "gaussiana", "marco": 7,  "sigma": 2.0}
}

def filtrar_ruido_vf(ze, vf, umbral_ze=12.0):
    # Dejamos en NaN las zonas sin lluvia para que proplot las pinte de gris
    return np.where(ze >= umbral_ze, vf, np.nan)

def calcular_gradiente_puro(datos, marco=3, tipo_ventana='lineal', sigma=2.0):
    """
    Cálculo del gradiente exacto y literal, sin alteraciones ni sumas artificiales.
    Restaurado al método de diferencias finitas original.
    """
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
        
        # Matemáticamente puro (sin abs(min)+1)
        gradiente_datos[i, :] = sup - inf

    # Preservar la máscara de NaN original
    return np.where(np.isnan(datos), np.nan, gradiente_datos)

def plot_5_ventanas_proplot_vf_only(xlim, times, heights, resultados, ruta_salida, nombre_evento):
    # --- REPLICACIÓN LITERAL DE LA LÓGICA DE TIEMPO DE codigo_plot.txt ---
    total_seconds = (xlim[1] - xlim[0]).total_seconds()
    if total_seconds <= 14400:
        xlocator = ('hour', range(0, 24, 1))
        xminorlocator = ('minute', 30)
    elif (total_seconds > 14400) and (total_seconds <= 82800.0):
        xlocator = ('hour', range(0, 24, 3))
        xminorlocator = ('hour', range(0, 24, 1))
    else:
        xlocator = ('hour', range(0, 24, 6))
        xminorlocator = ('hour', range(0, 24, 2))

    ylim = [0, 3600] if heights[-1] < 5000 else [0, 8000]
    extent = [dates.date2num(times[0]), dates.date2num(times[-1]), heights[0], heights[-1]]

    # --- REPLICACIÓN LITERAL DEL PLOTEO (Solo Vf, 5 filas) ---
    fig, axes = pplt.subplots(nrows=5, refwidth=5, refaspect=3, sharex=True, sharey=True)

    for i, ax in enumerate(axes):
        res = resultados[i]
        # Límites exactos originales y colormap RdBu
        m = ax.imshow(res['gradiente'], origin='lower', aspect='auto',
                      vmin=-3, vmax=10, cmap='RdBu', extent=extent)
        
        ax.set_facecolor('0.9') # Equivalente directo de add_no_data
        ax.format(ultitle=f"Ventana: {res['nombre']}",
                  xrotation=False,
                  xformatter='concise',
                  xlocator=xlocator,
                  xminorlocator=xminorlocator,
                  ylim=ylim,
                  yticklabelloc='both',
                  ytickloc='both',
                  xticklabelsize=8)

    # Formateos globales
    axes.format(
        suptitle=f'Gradientes Datos MRR UOH - Solo Velocidad de Caída\nEvento: {nombre_evento}',
        ylabel='Altitud [msnm]',
        xlabel=r'Hora UTC $\rightarrow$'
    )

    # Barra de color literal del proyecto original (extend='both', length=0.4)
    fig.colorbar(m, loc='r', label='[m/s]', length=0.4, extend='both')
    axes.format(xlim=xlim)

    fig.savefig(ruta_salida, dpi=150, bbox_inches='tight')
    plt.close(fig)

def ejecutar_comparacion():
    print("=== INICIANDO PLOTEO 100% EXACTO (MATEMÁTICA PURA + PROPLOT) ===")
    archivos = obtener_archivos_por_año(2023, DATA_RAW)
    
    out_dir = PROJECT_ROOT / "results" / "gradientes"
    out_dir.mkdir(parents=True, exist_ok=True)

    eventos_procesados = 0
    for ruta_muestra in archivos:
        if eventos_procesados >= 10: break
            
        nombre = ruta_muestra.stem
        try:
            with leer_netcdf(ruta_muestra) as ds:
                ze_raw = np.asarray(ds['attenuated_radar_reflectivity'].values)
                if np.sum(ze_raw > 20.0) < 100: continue
                    
                print(f"\n>> Procesando Evento: {nombre}...")
                vf_raw = np.asarray(ds['fall_velocity'].values)
                
                try:
                    new_time = pd.to_datetime(ds.time.values)
                except:
                    new_time = pd.to_datetime(ds.indexes['time'].astype(str))
                
                xlim = [new_time[0], new_time[-1]]
                
                # Desfase de altura exacto de replicacion_repo.ipynb
                h_vals = np.asarray(ds.height.values[0,:] if ds.height.ndim > 1 else ds.height.values)
                altura_inicial_desfase = 500 + (h_vals[1] - h_vals[0]) / 2
                h_ajustado = h_vals + altura_inicial_desfase

                ze_t = ze_raw.T if ze_raw.shape[0] == len(new_time) else ze_raw
                vf_t = vf_raw.T if vf_raw.shape[0] == len(new_time) else vf_raw

                # Extraemos el ruido, solo conservamos Vf
                vf_c = filtrar_ruido_vf(ze_t, vf_t)

                resultados = []
                for nombre_config, params in CONFIGURACIONES.items():
                    grad_vf = calcular_gradiente_puro(
                        vf_c, marco=params["marco"], 
                        tipo_ventana=params["ventana"], sigma=params["sigma"]
                    )
                    resultados.append({"nombre": nombre_config, "gradiente": grad_vf})

                out_file = out_dir / f"Comparacion_Gradientes_W_{nombre}.png"
                plot_5_ventanas_proplot_vf_only(xlim, new_time, h_ajustado, resultados, out_file, nombre)
                print(f"   [OK] Lámina de 5 ventanas guardada (Sin Ze y matemática pura).")
                eventos_procesados += 1
                
        except Exception as e:
            print(f"   [ERROR] Falló en {nombre}: {e}")

    print("\n✅ ¡Lote completo generado con la matemática y estética exactas!")

if __name__ == '__main__':
    ejecutar_comparacion()
