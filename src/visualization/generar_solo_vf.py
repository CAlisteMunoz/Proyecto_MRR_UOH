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
from matplotlib import colors, dates
import proplot as pplt
import xarray as xr

warnings.filterwarnings("ignore", message="All-NaN slice encountered")
warnings.filterwarnings("ignore")

from config import DATA_RAW, PROJECT_ROOT
from src.data.loader import obtener_archivos_por_año, leer_netcdf

# ==========================================
# FUNCIONES MATEMÁTICAS Y DE FILTRADO
# ==========================================
def ruido(Ze, Vf):
    Ze_filtered = Ze.where(Ze >= 12, 12)
    Vf_filtered = np.where(Ze >= 12, Vf, 2)
    return Ze_filtered, Vf_filtered

def calcular_gradiente_fisico(datos, marco):
    if marco == 1:
        pesos = np.array([1.0])
    else:
        pesos = np.array([marco - i for i in range(marco)])
        pesos = pesos / np.sum(pesos)
        
    niveles_restantes = datos.shape[0] - 1 - 2 * (marco - 1)
    if niveles_restantes <= 0:
        return np.zeros_like(datos)
        
    gradiente_datos = np.zeros((niveles_restantes, datos.shape[1]))

    for i in range(marco, datos.shape[0] - marco):
        superior = np.sum([pesos[j] * datos[i + j, :] for j in range(marco)], axis=0)
        inferior = np.sum([pesos[j] * datos[i - j - 1, :] for j in range(marco)], axis=0)
        gradiente_datos[i - marco, :] = inferior - superior 

    filas_superior = marco
    filas_inferior = marco - 1

    gradiente_datos_completo = np.pad(gradiente_datos, ((filas_superior, filas_inferior), (0, 0)), mode='constant', constant_values=0)
    return np.maximum(gradiente_datos_completo, 0)

# ==========================================
# VISUALIZACIÓN AISLADA (SÓLO Vf)
# ==========================================
def plot_solo_vf(xlim, times, heights, Vf, ax=None, hora_local=False):
    if hora_local:
        xlabel = r'Hora Local $\rightarrow$'
    else:
        xlabel = r'Hora UTC $\rightarrow$'

    if heights[-1] < 5000:
        ylim = [0, 3600]
    else:
        ylim = [0, 8000]

    total_seconds = (xlim[1]-xlim[0]).total_seconds()

    if total_seconds <= 14400:
        xlocator=('hour',range(0,24,1))
        xminorlocator=('minute',30)
    elif (total_seconds>14400) and (total_seconds<=82800.0):
        xlocator=('hour',range(0,24,3))
        xminorlocator=('hour',range(0,24,1))
    else:
        xlocator=('hour',range(0,24,6))
        xminorlocator=('hour',range(0,24,2))

    extent = [dates.date2num(times[0]), dates.date2num(times[-1]), heights[0], heights[-1]]

    fig, ax = pplt.subplots(refwidth=6, refaspect=3)

    mVf = ax.imshow(Vf, origin='lower', aspect='auto', vmin=0, vmax=4, cmap='RdBu', extent=extent)

    ax.format(ultitle='Gradiente de Velocidad de Caída', xrotation=False, xformatter='concise',
              xlocator=xlocator, xminorlocator=xminorlocator, ylim=ylim,
              yticklabelloc='both', ytickloc='both', xticklabelsize=8,
              suptitle='Análisis de Isoterma 0°C', ylabel='Altitud [msnm]', xlabel=xlabel)
              
    ax.colorbar(mVf, loc='r', label='Gradiente [m/s]', length=0.7, extend='both')

    if xlim != '':
        ax.format(xlim=xlim)

    return fig, ax

# ==========================================
# MOTOR DE EJECUCIÓN (10 EVENTOS)
# ==========================================
def procesar_eventos_vf():
    print("=== INICIANDO: GENERACIÓN DE GRÁFICOS AISLADOS DE GRADIENTE (10 EVENTOS) ===")
    
    archivos = list(obtener_archivos_por_año(2023, DATA_RAW)) + list(obtener_archivos_por_año(2024, DATA_RAW))
    dir_salida = PROJECT_ROOT / "results" / "gradientes_vf_solo"
    dir_salida.mkdir(parents=True, exist_ok=True)

    eventos_procesados = 0
    max_eventos = 10

    for ruta in archivos:
        if eventos_procesados >= max_eventos:
            print(f"\n=== SE ALCANZÓ EL LÍMITE DE {max_eventos} EVENTOS ===")
            break

        try:
            with leer_netcdf(ruta) as ds:
                Ze = ds['attenuated_radar_reflectivity']
                
                # Filtro de densidad
                puntos_lluvia = np.count_nonzero(np.nan_to_num(Ze.values) > 12.0)
                if np.isnan(Ze.values).all() or puntos_lluvia < 1000: 
                    continue
                    
                nombre = ruta.stem
                print(f">> Procesando {nombre} (Densidad: {puntos_lluvia} px)")
                
                Vf = ds['fall_velocity']
                new_time = pd.to_datetime(ds.time.values) if hasattr(ds, 'time') else pd.to_datetime(ds.indexes['time'].astype(str))
                xlim = [new_time[0], new_time[-1]]
                
                h_vals = np.asarray(ds.height.values[0,:] if ds.height.ndim > 1 else ds.height.values)
                heights_ajustado = h_vals + (500 + (h_vals[1] - h_vals[0]) / 2)

                Ze_filtered, Vf_filtered = ruido(Ze, Vf)
                Vf_t = Vf_filtered.T if Vf_filtered.shape[0] == len(new_time) else Vf_filtered

                # Generar el gradiente ponderado (ventana 5)
                grad_vf_pond = calcular_gradiente_fisico(Vf_t, 5)
                fig_pond, ax_vf = plot_solo_vf(xlim, new_time, heights_ajustado, grad_vf_pond, hora_local=False)
                
                # Añadir la ecuación en la esquina superior derecha
                ecuacion_texto = (
                    r"$\nabla V_i = \sum w_j V_{i-j-1} - \sum w_j V_{i+j}$" + "\n\n" +
                    r"$w_j = \frac{m-j}{\sum (m-k)}$"
                )
                ax_vf.text(0.98, 0.95, ecuacion_texto, transform=ax_vf.transAxes,
                        fontsize=10, verticalalignment='top', horizontalalignment='right',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray'))

                out_pond = dir_salida / f"Solo_Vf_Ponderado_{nombre}.png"
                fig_pond.savefig(out_pond, dpi=200, bbox_inches='tight')
                plt.close(fig_pond)
                
                eventos_procesados += 1
                
        except Exception as e:
            print(f"   [ERROR] Falló {ruta.stem}: {e}")

if __name__ == '__main__':
    procesar_eventos_vf()
