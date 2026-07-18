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
# VISUALIZACIÓN APILADA (1, 3, 5, 7 px)
# ==========================================
def plot_comparativa_ventanas(xlim, times, heights, Vf_t, hora_local=False):
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

    # 4 filas para las 4 ventanas
    fig, axs = pplt.subplots(nrows=4, refwidth=6, refaspect=3.5, sharex=True, sharey=True)
    
    ventanas = [1, 3, 5, 7]
    for i, marco in enumerate(ventanas):
        grad_vf = calcular_gradiente_fisico(Vf_t, marco)
        
        m = axs[i].imshow(grad_vf, origin='lower', aspect='auto', vmin=0, vmax=4, cmap='RdBu', extent=extent)
        axs[i].format(ultitle=f'Velocidad de caída ({marco} px)')
        
        # Agregamos la barra de color a cada panel para mantener el estilo original
        axs[i].colorbar(m, loc='r', label='[m/s]', length=0.7, extend='both')

    # Formateo general de la figura (ejes y título principal)
    axs.format(
        suptitle='Gradientes datos MRR UOH - Comparativa Ventanas',
        ylabel='Altitud [msnm]', 
        xlabel=xlabel,
        xrotation=False, 
        xformatter='concise',
        xlocator=xlocator, 
        xminorlocator=xminorlocator, 
        ylim=ylim,
        yticklabelloc='both', 
        ytickloc='both', 
        xticklabelsize=8
    )

    if xlim != '':
        axs.format(xlim=xlim)

    return fig

# ==========================================
# MOTOR DE EJECUCIÓN (10 EVENTOS DE 10.000+ PX)
# ==========================================
def procesar_eventos_comparativa():
    print("=== INICIANDO: BÚSQUEDA DE 10 EVENTOS MASIVOS (>10.000 PX) ===")
    
    archivos = list(obtener_archivos_por_año(2023, DATA_RAW)) + list(obtener_archivos_por_año(2024, DATA_RAW))
    dir_salida = PROJECT_ROOT / "results" / "comparativa_ventanas"
    dir_salida.mkdir(parents=True, exist_ok=True)

    eventos_procesados = 0
    max_eventos = 10

    for ruta in archivos:
        if eventos_procesados >= max_eventos:
            print(f"\n=== SE ALCANZÓ EL LÍMITE DE {max_eventos} EVENTOS MASIVOS ===")
            break

        try:
            with leer_netcdf(ruta) as ds:
                Ze = ds['attenuated_radar_reflectivity']
                
                # Filtro ESTRICTO: Solo eventos con más de 10.000 puntos válidos
                puntos_lluvia = np.count_nonzero(np.nan_to_num(Ze.values) > 12.0)
                if np.isnan(Ze.values).all() or puntos_lluvia < 10000: 
                    continue
                    
                nombre = ruta.stem
                print(f">> Procesando {nombre} (Densidad: {puntos_lluvia} px válidos) - Evento {eventos_procesados + 1}/10")
                
                Vf = ds['fall_velocity']
                new_time = pd.to_datetime(ds.time.values) if hasattr(ds, 'time') else pd.to_datetime(ds.indexes['time'].astype(str))
                xlim = [new_time[0], new_time[-1]]
                
                h_vals = np.asarray(ds.height.values[0,:] if ds.height.ndim > 1 else ds.height.values)
                heights_ajustado = h_vals + (500 + (h_vals[1] - h_vals[0]) / 2)

                Ze_filtered, Vf_filtered = ruido(Ze, Vf)
                Vf_t = Vf_filtered.T if Vf_filtered.shape[0] == len(new_time) else Vf_filtered

                # Generar la figura comparativa
                fig = plot_comparativa_ventanas(xlim, new_time, heights_ajustado, Vf_t, hora_local=False)
                
                out_path = dir_salida / f"Comparativa_1-3-5-7px_{nombre}.png"
                fig.savefig(out_path, dpi=200, bbox_inches='tight')
                plt.close(fig)
                
                eventos_procesados += 1
                
        except Exception as e:
            pass # Silenciar errores de lectura para que siga buscando sin ensuciar la consola

if __name__ == '__main__':
    procesar_eventos_comparativa()
