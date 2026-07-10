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

warnings.filterwarnings("ignore")

from config import DATA_RAW, PROJECT_ROOT
from src.data.loader import obtener_archivos_por_año, leer_netcdf

def plot_vf_exclusivo(xlim, times, heights, Vf, ruta_salida, nombre_evento):
    # Formateo de tiempo exacto de tu proyecto
    total_seconds = (xlim[1] - xlim[0]).total_seconds()
    if total_seconds <= 14400:
        xlocator, xminorlocator = ('hour', range(0, 24, 1)), ('minute', 30)
    elif total_seconds <= 82800.0:
        xlocator, xminorlocator = ('hour', range(0, 24, 3)), ('hour', range(0, 24, 1))
    else:
        xlocator, xminorlocator = ('hour', range(0, 24, 6)), ('hour', range(0, 24, 2))

    ylim = [0, 3600] if heights[-1] < 5000 else [0, 8000]
    extent = [dates.date2num(times[0]), dates.date2num(times[-1]), heights[0], heights[-1]]

    # CREAMOS UN SOLO PANEL (nrows=1) en lugar de dos
    fig, ax = pplt.subplots(nrows=1, refwidth=6, refaspect=3)

    # PALETA ROJIZA (Yellow-Orange-Red)
    cmap = plt.get_cmap('YlOrRd').copy()
    cmap.set_bad('0.9', 1) # Mantenemos el fondo gris intacto para datos nulos

    # Ploteo exclusivo de Vf (Escala 0 a 10 m/s es ideal para velocidad bruta)
    mVf = ax.imshow(Vf, origin='lower', aspect='auto',
                    vmin=0, vmax=10, cmap=cmap, extent=extent)
    
    ax.set_facecolor('0.9')
    
    ax.format(ultitle='Velocidad de Caída (Vf)',
              xrotation=False, xformatter='concise',
              xlocator=xlocator, xminorlocator=xminorlocator,
              ylim=ylim, yticklabelloc='both', ytickloc='both', xticklabelsize=8,
              suptitle=f'Datos MRR UOH - Análisis Exclusivo Vf\nEvento: {nombre_evento}',
              ylabel='Altitud [msnm]',
              xlabel=r'Hora UTC $\rightarrow$')

    # Barra de color integrada
    fig.colorbar(mVf, loc='r', label='[m/s]', length=0.8, extend='max')
    ax.format(xlim=xlim)

    fig.savefig(ruta_salida, dpi=150, bbox_inches='tight')
    plt.close(fig)

def ejecutar_ploteo_vf():
    print("=== INICIANDO VISUALIZACIÓN EXCLUSIVA DE VF (PALETA ROJIZA) ===")
    
    archivos = list(obtener_archivos_por_año(2023, DATA_RAW))
    
    out_dir = PROJECT_ROOT / "results" / "raw_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    eventos = 0
    for ruta in archivos:
        if eventos >= 5: break # Generará 5 imágenes de muestra rápida
            
        nombre = ruta.stem
        try:
            with leer_netcdf(ruta) as ds:
                ze_raw = np.asarray(ds['attenuated_radar_reflectivity'].values)
                
                # Buscamos días con precipitaciones
                if np.nansum(ze_raw > 15.0) < 50: 
                    continue
                    
                print(f">> Generando gráfica para: {nombre}...")
                vf_raw = np.asarray(ds['fall_velocity'].values)
                
                try:
                    new_time = pd.to_datetime(ds.time.values)
                except:
                    new_time = pd.to_datetime(ds.indexes['time'].astype(str))
                
                xlim = [new_time[0], new_time[-1]]
                
                h_vals = np.asarray(ds.height.values[0,:] if ds.height.ndim > 1 else ds.height.values)
                desfase = 500 + (h_vals[1] - h_vals[0]) / 2
                heights = h_vals + desfase

                ze_t = ze_raw.T if ze_raw.shape[0] == len(new_time) else ze_raw
                vf_t = vf_raw.T if vf_raw.shape[0] == len(new_time) else vf_raw

                # Enmascaramiento: Usamos Ze solo por debajo de la mesa para filtrar estática
                vf_filtrada = np.where(ze_t >= 10.0, vf_t, np.nan)

                out_file = out_dir / f"Velocidad_Caida_Rojiza_{nombre}.png"
                plot_vf_exclusivo(xlim, new_time, heights, vf_filtrada, out_file, nombre)
                print(f"   [OK] Gráfica rojiza generada.")
                eventos += 1
                
        except Exception as e:
            pass 

    print("\n✅ ¡Imágenes generadas con éxito en results/raw_plots/!")

if __name__ == '__main__':
    ejecutar_ploteo_vf()
