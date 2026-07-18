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
from matplotlib import dates
import proplot as pplt
import xarray as xr

warnings.filterwarnings("ignore")

from config import DATA_RAW, PROJECT_ROOT
from src.data.loader import obtener_archivos_por_año, leer_netcdf

def add_no_data(ax, times, xlim):
    """Añade área de 'Sin Datos' al gráfico (Achurado)"""
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    if len(times) == 0 or (xlim is not None and (times[-1] < xlim[0] or times[0] > xlim[1])):
        rect = plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin,
                           fill=False, hatch='////', edgecolor='gray3',
                           linewidth=0, zorder=10, alpha=0.5)
        ax.add_patch(rect)
        ax.text((xmin+xmax)/2, (ymin+ymax)/2, 'SIN DATOS',
               color='red', ha='center', va='center',
               fontsize=12, weight='bold', zorder=11,
               bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

def ruido(Ze, Vf):
    """Filtro de ruido original del repositorio."""
    Ze_filtered = Ze.where(Ze >= 12, 12)
    Vf_filtered = np.where(Ze >= 12, Vf, 2)
    return Ze_filtered, Vf_filtered

def calcular_gradiente(datos, marco):
    """Cálculo de gradiente exacto del código original."""
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
        gradiente_datos[i - marco, :] = superior - inferior

    filas_superior = marco
    filas_inferior = marco - 1

    gradiente_datos_completo = np.pad(gradiente_datos, 
                                      ((filas_superior, filas_inferior), (0, 0)), 
                                      mode='constant', constant_values=0)
    return gradiente_datos_completo

def ejecutar_procesamiento():
    print("=== INICIANDO: ANÁLISIS DE GRADIENTES POR PÍXELES (ESTRUCTURA ORIGINAL) ===")
    
    archivos = list(obtener_archivos_por_año(2023, DATA_RAW)) + list(obtener_archivos_por_año(2024, DATA_RAW))
    out_dir = PROJECT_ROOT / "results" / "gradientes_pixeles"
    out_dir.mkdir(parents=True, exist_ok=True)

    VENTANAS_PIXELES = [1, 3, 5, 7, 9]
    eventos_procesados = 0

    for ruta in archivos:
        if eventos_procesados >= 10: break
        nombre = ruta.stem
        try:
            with leer_netcdf(ruta) as ds:
                Ze = ds['attenuated_radar_reflectivity']
                Vf = ds['fall_velocity']
                
                # Evitar procesar archivos casi sin datos válidos
                if np.nansum(Ze.values > 15.0) < 50: 
                    continue
                    
                print(f">> Procesando Evento: {nombre}")
                
                new_time = pd.to_datetime(ds.time.values) if hasattr(ds, 'time') else pd.to_datetime(ds.indexes['time'].astype(str))
                xlim = [new_time[0], new_time[-1]]
                
                h_vals = np.asarray(ds.height.values[0,:] if ds.height.ndim > 1 else ds.height.values)
                altura_inicial_desfase = 500 + (h_vals[1] - h_vals[0]) / 2
                heights_ajustado = h_vals + altura_inicial_desfase

                # 1. Aplicamos el ruido original (vital para mantener el fondo en cero)
                _, Vf_filtered = ruido(Ze, Vf)
                
                # 2. Transposición requerida por tu función original
                Vf_t = Vf_filtered.T if Vf_filtered.shape[0] == len(new_time) else Vf_filtered

                resultados = []
                for marco in VENTANAS_PIXELES:
                    grad_vf = calcular_gradiente(Vf_t, marco)
                    resultados.append({"nombre": f"Filtro: {marco} Píxeles", "gradiente": grad_vf})

                # --- CONFIGURACIÓN PROPLOT ORIGINAL ---
                times_num = dates.date2num(new_time)
                dx = (times_num[-1] - times_num[0]) / (len(times_num) - 1) if len(times_num) > 1 else 0
                dy = (heights_ajustado[-1] - heights_ajustado[0]) / (len(heights_ajustado) - 1) if len(heights_ajustado) > 1 else 0
                extent = [times_num[0] - dx/2, times_num[-1] + dx/2, heights_ajustado[0] - dy/2, heights_ajustado[-1] + dy/2]
                
                ylim = [0, 3600] if heights_ajustado[-1] < 5000 else [0, 8000]

                total_seconds = (xlim[1] - xlim[0]).total_seconds()
                if total_seconds <= 14400:
                    xlocator, xminorlocator = ('hour', range(0, 24, 1)), ('minute', 30)
                elif 14400 < total_seconds <= 82800.0:
                    xlocator, xminorlocator = ('hour', range(0, 24, 3)), ('hour', range(0, 24, 1))
                else:
                    xlocator, xminorlocator = ('hour', range(0, 24, 6)), ('hour', range(0, 24, 2))

                fig, axes = pplt.subplots(nrows=len(VENTANAS_PIXELES), refwidth=5, refaspect=3, sharex=True, sharey=True)

                for i, ax in enumerate(axes):
                    res = resultados[i]
                    add_no_data(ax, new_time, xlim)
                    m = ax.imshow(res['gradiente'], origin='lower', aspect='auto',
                                  vmin=-3, vmax=10, cmap='RdBu', extent=extent, interpolation='nearest')
                    
                    ax.format(ultitle=res['nombre'],
                              xrotation=False, xformatter='concise',
                              xlocator=xlocator, xminorlocator=xminorlocator,
                              ylim=ylim, yticklabelloc='both', ytickloc='both', xticklabelsize=8)

                axes.format(
                    suptitle=f'Radar Perfilador MRR en UOH Rancagua\nGradientes de Velocidad de Caída - Evento: {nombre}',
                    ylabel='Altitud [msnm]',
                    xlabel=r'Hora UTC $\rightarrow$'
                )

                fig.colorbar(m, loc='r', label='[m/s]', length=0.4, extend='both')
                axes.format(xlim=xlim)

                out_file = out_dir / f"Gradientes_Pixeles_{nombre}.png"
                fig.savefig(out_file, dpi=150, bbox_inches='tight')
                plt.close(fig)
                print(f"   [OK] Guardado: {out_file.name}")
                eventos_procesados += 1
                
        except Exception as e:
            print(f"   [ERROR] Falló el archivo {nombre}: {e}")

if __name__ == '__main__':
    ejecutar_procesamiento()
