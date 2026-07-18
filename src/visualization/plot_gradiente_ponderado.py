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

# Importamos las funciones del script que acabamos de crear para no repetir código
from src.visualization.plot_gradientes_ventanas_pixeles import ruido, calcular_gradiente, add_no_data

def ejecutar_ponderado():
    print("=== INICIANDO: ANÁLISIS PONDERADO CON ECUACIÓN (ESTRUCTURA ORIGINAL) ===")
    
    archivos = list(obtener_archivos_por_año(2023, DATA_RAW)) + list(obtener_archivos_por_año(2024, DATA_RAW))
    out_dir = PROJECT_ROOT / "results" / "gradientes_ponderados"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Buscamos un archivo que tenga datos suficientes
    ruta = None
    for r in archivos:
        try:
            with leer_netcdf(r) as ds:
                if np.nansum(ds['attenuated_radar_reflectivity'].values > 15.0) > 50:
                    ruta = r
                    break
        except:
            continue
            
    if ruta is None:
        print("[ERROR] No se encontraron archivos con datos válidos.")
        return

    nombre = ruta.stem
    print(f">> Procesando Evento Representativo: {nombre}")
    
    try:
        with leer_netcdf(ruta) as ds:
            Ze = ds['attenuated_radar_reflectivity']
            Vf = ds['fall_velocity']
            
            new_time = pd.to_datetime(ds.time.values) if hasattr(ds, 'time') else pd.to_datetime(ds.indexes['time'].astype(str))
            xlim = [new_time[0], new_time[-1]]
            
            h_vals = np.asarray(ds.height.values[0,:] if ds.height.ndim > 1 else ds.height.values)
            heights_ajustado = h_vals + (500 + (h_vals[1] - h_vals[0]) / 2)

            _, Vf_filtered = ruido(Ze, Vf)
            Vf_t = Vf_filtered.T if Vf_filtered.shape[0] == len(new_time) else Vf_filtered

            # Ventana de 5 píxeles para mostrar el ponderador
            marco = 5
            grad_vf = calcular_gradiente(Vf_t, marco)

            times_num = dates.date2num(new_time)
            dx = (times_num[-1] - times_num[0]) / (len(times_num) - 1) if len(times_num) > 1 else 0
            dy = (heights_ajustado[-1] - heights_ajustado[0]) / (len(heights_ajustado) - 1) if len(heights_ajustado) > 1 else 0
            extent = [times_num[0] - dx/2, times_num[-1] + dx/2, heights_ajustado[0] - dy/2, heights_ajustado[-1] + dy/2]
            ylim = [0, 3600] if heights_ajustado[-1] < 5000 else [0, 8000]

            fig, ax = pplt.subplots(nrows=1, refwidth=6, refaspect=2.5)

            add_no_data(ax, new_time, xlim)
            m = ax.imshow(grad_vf, origin='lower', aspect='auto',
                          vmin=-3, vmax=10, cmap='RdBu', extent=extent, interpolation='nearest')
            
            ax.format(ultitle=f'Gradiente Normalizado (Ventana: {marco} píxeles)',
                      xrotation=False, xformatter='concise',
                      ylim=ylim, yticklabelloc='both', ytickloc='both',
                      suptitle=f'Radar Perfilador MRR en UOH Rancagua\nAnálisis Ponderado - Evento: {nombre}',
                      ylabel='Altitud [msnm]', xlabel=r'Hora UTC $\rightarrow$')

            fig.colorbar(m, loc='r', label='[m/s]', length=0.7)

            # Ecuación matemática inyectada en la esquina superior derecha
            ecuacion_texto = (
                r"$\nabla V_i = \sum_{j=0}^{m-1} w_j V_{i+j} - \sum_{j=0}^{m-1} w_j V_{i-j-1}$" + "\n\n" +
                r"$w_j = \frac{m-j}{\sum_{k=0}^{m-1} (m-k)}$"
            )
            
            ax.text(0.98, 0.95, ecuacion_texto, transform=ax.transAxes,
                    fontsize=10, verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray'))

            ax.format(xlim=xlim)

            out_file = out_dir / f"Gradiente_Ponderado_Ecuacion_{nombre}.png"
            fig.savefig(out_file, dpi=200, bbox_inches='tight')
            plt.close(fig)

            print(f"   [OK] Guardado: {out_file.name}")
            
    except Exception as e:
        print(f"   [ERROR] Falló el procesamiento: {e}")

if __name__ == '__main__':
    ejecutar_ponderado()
