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

warnings.filterwarnings("ignore")

from config import DATA_RAW, PROJECT_ROOT
from src.data.loader import obtener_archivos_por_año, leer_netcdf

# ==========================================
# PALETAS Y FUNCIONES AUXILIARES
# ==========================================
paleta0 = ['#ccd8ff','#3366ff','#9fdf9f','#00b300','#ffff00','#ffcc30','#e62e00','#ff6600','#fff0e5','#c03fc0','#602060']

def add_no_data(ax, times, xlim):
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
    Ze_filtered = Ze.where(Ze >= 12, 12)
    Vf_filtered = np.where(Ze >= 12, Vf, 2)
    return Ze_filtered, Vf_filtered

def calcular_gradiente_fisico(datos, marco):
    """
    Cálculo de gradiente ponderado.
    CORRECCIÓN FÍSICA: Se invierte la resta a (inferior - superior) 
    para que la aceleración de caída sea un valor positivo.
    """
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
        
        # FÍSICA: Aceleración positiva (Lluvia [mayor vel] - Nieve [menor vel])
        gradiente_datos[i - marco, :] = inferior - superior 

    filas_superior = marco
    filas_inferior = marco - 1

    gradiente_datos_completo = np.pad(gradiente_datos, ((filas_superior, filas_inferior), (0, 0)), mode='constant', constant_values=0)
    
    # Aseguramos que no haya artefactos negativos residuales extremos
    return np.maximum(gradiente_datos_completo, 0)

# ==========================================
# CÓDIGO EXACTO DE VISUALIZACIÓN PROPORCIONADO
# ==========================================
def plot_mrr3_imshow(xlim, times, heights, Ze, Vf=None, hora_local=False, ax=None,
              ytickloc='both', colorhex = paleta0):
    cmap4 = []
    stops = [0,1./10,2./10,3./10,4./10,5./10,6./10,7./10,8./10,9./10,1]
    for value, color in zip(stops,colorhex):
        cmap4.append((value,color))
    dbzmap = colors.LinearSegmentedColormap.from_list("custom",cmap4)
    dbzmap.set_bad('0.9',1)
    bounds = np.arange(-5,50,1)
    norm = colors.BoundaryNorm(bounds, dbzmap.N)

    if hora_local:
        xlabel = r'Hora Local $\rightarrow$'
    else:
        xlabel = r'Hora UTC $\rightarrow$'

    if heights[-1]< 5000:
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

    if Vf is None:
        if ax is None:
            fig, ax = pplt.subplots(refwidth=5, refaspect=3)

        mZe = ax.imshow(Ze, origin='lower', aspect='auto',
                        cmap=dbzmap, norm=norm,
                        extent=extent)

        ax.format(ultitle='Reflectividad Equivalente',
                  xrotation=False,
                  xformatter='concise',
                  xlocator=xlocator,
                  xminorlocator=xminorlocator,
                  ylim=ylim,
                  yticklabelloc=ytickloc,
                  ytickloc='both',
                  xticklabelsize=8,
                  suptitle='Gradientes datos MRR UOH',
                  ylabel='Altitud [msnm]',
                  xlabel=xlabel)

        ax.colorbar(mZe, loc='r', label='[dBZ]', length=0.7)

        if xlim != '':
            ax.format(xlim=xlim)

    else:
        fig, ax = pplt.subplots(nrows=2, refwidth=5, refaspect=3)

        mZe = ax[0].imshow(Ze, origin='lower', aspect='auto',
                           cmap=dbzmap, norm=norm,
                           extent=extent)
        add_no_data(ax[0], times, xlim)

        # CORRECCIÓN DE VISIBILIDAD: Ajustamos vmin y vmax a [-4, 4] 
        # para que los valores positivos (1 a 4 m/s) saturen en azul profundo
        mVf = ax[1].imshow(Vf, origin='lower', aspect='auto',
                           vmin=-4, vmax=4,
                           cmap='RdBu',
                           extent=extent)
        add_no_data(ax[1], times, xlim)

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

        ax[1].format(ultitle='Aceleración de caída (Gradiente)',
                     xrotation=False,
                     xformatter='concise',
                     xlocator=xlocator,
                     xminorlocator=xminorlocator,
                     ylim=ylim,
                     yticklabelloc='both',
                     ytickloc='both',
                     xticklabelsize=8)

        ax[1].colorbar(mVf, loc='r', label='[m/s²]', length=0.4,
                       extend='both')

        if xlim != '':
            ax[0].format(xlim=xlim)
            ax[1].format(xlim=xlim)

    return fig # Modificado de fig.show() a return fig para permitir el guardado automático

# ==========================================
# MOTOR DE EJECUCIÓN
# ==========================================
def procesar_eventos():
    print("=== INICIANDO: RENDERIZADO CON FÍSICA CORREGIDA Y ALTO CONTRASTE ===")
    
    archivos = list(obtener_archivos_por_año(2023, DATA_RAW)) + list(obtener_archivos_por_año(2024, DATA_RAW))
    dir_pixeles = PROJECT_ROOT / "results" / "gradientes_pixeles"
    dir_ponderados = PROJECT_ROOT / "results" / "gradientes_ponderados"
    
    eventos_procesados = 0

    for ruta in archivos:
        if eventos_procesados >= 5: break # Procesar los 5 mejores eventos
        
        try:
            with leer_netcdf(ruta) as ds:
                Ze = ds['attenuated_radar_reflectivity']
                if np.nansum(Ze.values > 15.0) < 50: 
                    continue
                    
                nombre = ruta.stem
                print(f">> Procesando Evento: {nombre}")
                
                Vf = ds['fall_velocity']
                new_time = pd.to_datetime(ds.time.values) if hasattr(ds, 'time') else pd.to_datetime(ds.indexes['time'].astype(str))
                xlim = [new_time[0], new_time[-1]]
                
                h_vals = np.asarray(ds.height.values[0,:] if ds.height.ndim > 1 else ds.height.values)
                heights_ajustado = h_vals + (500 + (h_vals[1] - h_vals[0]) / 2)

                Ze_filtered, Vf_filtered = ruido(Ze, Vf)
                Vf_t = Vf_filtered.T if Vf_filtered.shape[0] == len(new_time) else Vf_filtered
                Ze_t = Ze_filtered.T if Ze_filtered.shape[0] == len(new_time) else Ze_filtered

                # 1. BUCLE DE PÍXELES (Múltiples Ventanas)
                ventanas = [1, 3, 5, 7]
                for marco in ventanas:
                    grad_vf = calcular_gradiente_fisico(Vf_t, marco)
                    
                    # Usamos tu función plot_mrr3_imshow exacta
                    fig = plot_mrr3_imshow(xlim, new_time, heights_ajustado, Ze_t, Vf=grad_vf, hora_local=False)
                    
                    out_file = dir_pixeles / f"Gradiente_{marco}px_{nombre}.png"
                    fig.savefig(out_file, dpi=150, bbox_inches='tight')
                    plt.close(fig)

                # 2. SECCIÓN PONDERADA (Ventana 5 con Ecuación)
                grad_vf_pond = calcular_gradiente_fisico(Vf_t, 5)
                fig_pond = plot_mrr3_imshow(xlim, new_time, heights_ajustado, Ze_t, Vf=grad_vf_pond, hora_local=False)
                
                # Obtener el eje inferior (Vf) para inyectar la ecuación
                ax_vf = fig_pond.subplot(2) 
                ecuacion_texto = (
                    r"$\nabla V_i = \sum w_j V_{i-j-1} - \sum w_j V_{i+j}$" + "\n\n" +
                    r"$w_j = \frac{m-j}{\sum (m-k)}$"
                )
                ax_vf.text(0.98, 0.95, ecuacion_texto, transform=ax_vf.transAxes,
                        fontsize=10, verticalalignment='top', horizontalalignment='right',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray'))

                out_pond = dir_ponderados / f"Gradiente_Ponderado_Ecuacion_{nombre}.png"
                fig_pond.savefig(out_pond, dpi=200, bbox_inches='tight')
                plt.close(fig_pond)
                
                print(f"   [OK] Láminas generadas para {nombre}")
                eventos_procesados += 1
                
        except Exception as e:
            print(f"   [ERROR] Falló {ruta.stem}: {e}")

if __name__ == '__main__':
    procesar_eventos()
