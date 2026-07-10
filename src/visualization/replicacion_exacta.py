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
from matplotlib import colors, dates
import proplot as pplt

warnings.filterwarnings("ignore")

from config import DATA_RAW, PROJECT_ROOT
from src.data.loader import obtener_archivos_por_año, leer_netcdf

# Paleta estándar MRR (Aproximación de paleta0)
paleta0 = ['#ffffff', '#00ffff', '#0099ff', '#0000ff', '#00ff00', 
           '#009900', '#ffff00', '#ff9900', '#ff0000', '#ff00ff', '#9900cc']

def calcular_gradiente_original(datos, marco=1):
    """
    Replicación exacta del cálculo de gradiente de replicacion_repo.ipynb
    con la regla microfísica de gradientes estrictamente positivos.
    """
    grad = np.full_like(datos, np.nan)
    for i in range(marco, datos.shape[0] - marco):
        sup = datos[i + marco, :]
        inf = datos[i - marco, :]
        grad[i, :] = sup - inf
        
    # Regla del profesor: El gradiente no puede ser negativo
    min_val = np.nanmin(grad)
    if min_val < 0:
        grad = grad + np.abs(min_val) + 1.0
        
    return grad

def add_no_data_replica(ax):
    """Réplica del comportamiento de add_no_data pintando el fondo gris"""
    ax.set_facecolor('0.9')

# === TU CÓDIGO EXACTO DE codigo_plot.txt ===
def plot_mrr3_imshow(xlim, times, heights, Ze, Vf=None, hora_local=False, ax=None,
              ytickloc='both', colorhex=paleta0, ruta_salida=None):
    
    cmap4 = []
    stops = [0, 1./10, 2./10, 3./10, 4./10, 5./10, 6./10, 7./10, 8./10, 9./10, 1]
    for value, color in zip(stops, colorhex):
        cmap4.append((value, color))
    dbzmap = colors.LinearSegmentedColormap.from_list("custom", cmap4)
    dbzmap.set_bad('0.9', 1)
    bounds = np.arange(-5, 50, 1)
    norm = colors.BoundaryNorm(bounds, dbzmap.N)

    if hora_local:
        xlabel = r'Hora Local $\rightarrow$'
    else:
        xlabel = r'Hora UTC $\rightarrow$'

    if heights[-1] < 5000:
        ylim = [0, 3600]
    else:
        ylim = [0, 8000]

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

    extent = [dates.date2num(times[0]), dates.date2num(times[-1]), heights[0], heights[-1]]

    fig, ax = pplt.subplots(nrows=2, refwidth=5, refaspect=3)

    mZe = ax[0].imshow(Ze, origin='lower', aspect='auto',
                       cmap=dbzmap, norm=norm,
                       extent=extent)
    add_no_data_replica(ax[0])

    mVf = ax[1].imshow(Vf, origin='lower', aspect='auto',
                       vmin=-3, vmax=10,
                       cmap='RdBu',
                       extent=extent)
    add_no_data_replica(ax[1])

    ax[0].format(ultitle='Reflectividad Equivalente',
                 xrotation=False,
                 xformatter='concise',
                 xlocator=xlocator,
                 xminorlocator=xminorlocator,
                 ylim=ylim,
                 yticklabelloc='both',
                 ytickloc='both',
                 xticklabelsize=8,
                 suptitle='Gradientes datos MRR UOH',
                 ylabel='Altitud [msnm]',
                 xlabel=xlabel)

    ax[0].colorbar(mZe, loc='r', label='[dBZ]', length=0.4)

    ax[1].format(ultitle='Velocidad de caída',
                 xrotation=False,
                 xformatter='concise',
                 xlocator=xlocator,
                 xminorlocator=xminorlocator,
                 ylim=ylim,
                 yticklabelloc='both',
                 ytickloc='both',
                 xticklabelsize=8)

    ax[1].colorbar(mVf, loc='r', label='[m/s]', length=0.4,
                   extend='both')

    ax[0].format(xlim=xlim)
    ax[1].format(xlim=xlim)

    fig.savefig(ruta_salida, dpi=150, bbox_inches='tight')
    plt.close(fig)

def ejecutar_clonacion():
    print("=== INICIANDO CLONACIÓN EXACTA DEL PROYECTO ORIGINAL (PROPLOT) ===")
    archivos = obtener_archivos_por_año(2023, DATA_RAW)
    
    out_dir = PROJECT_ROOT / "results" / "replicacion_exacta"
    out_dir.mkdir(parents=True, exist_ok=True)

    eventos_procesados = 0
    for ruta_muestra in archivos:
        if eventos_procesados >= 10:
            break
            
        nombre = ruta_muestra.stem
        try:
            with leer_netcdf(ruta_muestra) as ds:
                ze_raw = np.asarray(ds['attenuated_radar_reflectivity'].values)
                if np.sum(ze_raw > 20.0) < 100:
                    continue
                    
                print(f"\n>> Replicando Evento: {nombre}...")
                vf_raw = np.asarray(ds['fall_velocity'].values)
                
                # Solución a cftime
                try:
                    new_time = pd.to_datetime(ds.time.values)
                except:
                    new_time = pd.to_datetime(ds.indexes['time'].astype(str))
                
                xlim = [new_time[0], new_time[-1]]
                
                # Ajuste de altura exacto de replicacion_repo.ipynb
                h_vals = np.asarray(ds.height.values[0,:] if ds.height.ndim > 1 else ds.height.values)
                altura_inicial_desfase = 500 + (h_vals[1] - h_vals[0]) / 2
                heights_ajustado = h_vals + altura_inicial_desfase

                # Filtro de ruido básico
                ze_filtered = np.where(ze_raw >= 12.0, ze_raw, np.nan)
                vf_filtered = np.where(ze_raw >= 12.0, vf_raw, np.nan)

                # CORRECCIÓN DE LA TRANSPUESTA (.T) Y GRADIENTE
                marco = 1
                ze_t = ze_filtered.T if ze_filtered.shape[0] == len(new_time) else ze_filtered
                vf_t = vf_filtered.T if vf_filtered.shape[0] == len(new_time) else vf_filtered
                
                gradiente_Ze = calcular_gradiente_original(ze_t, marco)
                gradiente_Vf = calcular_gradiente_original(vf_t, marco)

                out_file = out_dir / f"Replica_Original_{nombre}.png"
                
                # Llamada idéntica al plot
                plot_mrr3_imshow(xlim=xlim, times=new_time, heights=heights_ajustado, 
                                 Ze=gradiente_Ze, Vf=gradiente_Vf, hora_local=False, 
                                 ruta_salida=out_file)
                
                print(f"   [OK] Clonación generada perfectamente.")
                eventos_procesados += 1
                
        except Exception as e:
            print(f"   [ERROR] Falló en {nombre}: {e}")

if __name__ == '__main__':
    ejecutar_clonacion()
