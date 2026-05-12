import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors, dates
import numpy as np
import pandas as pd

PALETA_ZE = ['#ccd8ff','#3366ff','#9fdf9f','#00b300','#ffff00','#ffcc30','#e62e00','#ff6600','#fff0e5','#c03fc0','#602060']

def generar_plot_mrr_dual(ds, iso_ze, var_ze, iso_vf, var_vf, ruta_salida, titulo):
    # Convertir tiempos a formato numérico que Matplotlib entiende
    tiempos_raw = pd.to_datetime(ds.time.values)
    tiempos_num = dates.date2num(tiempos_raw)
    
    heights = (ds.height.values[0,:] if ds.height.ndim > 1 else ds.height.values) + 500
    
    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(12, 10), sharex=True)
    
    # El extent DEBE ser numérico (xmin, xmax, ymin, ymax)
    extent = [tiempos_num[0], tiempos_num[-1], heights[0], heights[-1]]
    
    # --- PANEL 1: Reflectividad ---
    cmap_ze = colors.LinearSegmentedColormap.from_list("ze_uoh", PALETA_ZE)
    ze = ds['attenuated_radar_reflectivity'].values
    im1 = ax1.imshow(ze.T, origin='lower', aspect='auto', extent=extent, cmap=cmap_ze, vmin=-5, vmax=50)
    
    ax1.plot(tiempos_num, iso_ze, color='black', linewidth=1.5, label='Isoterma 0°C detectada')
    ax1.fill_between(tiempos_num, iso_ze - np.sqrt(var_ze), iso_ze + np.sqrt(var_ze), color='darkred', alpha=0.3)
    
    ax1.set_title(f"Radar Perfilador MRR en UOH Rancagua", fontsize=16, fontweight='bold')
    ax1.set_ylabel("Altitud [msnm]")
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2)
    plt.colorbar(im1, ax=ax1, label="[dBZ]")

    # --- PANEL 2: Velocidad de Caída ---
    vf = ds['fall_velocity'].values
    im2 = ax2.imshow(vf.T, origin='lower', aspect='auto', extent=extent, cmap='RdBu_r', vmin=-2, vmax=10)
    
    ax2.plot(tiempos_num, iso_vf, color='black', linewidth=1.5)
    ax2.fill_between(tiempos_num, iso_vf - np.sqrt(var_vf), iso_vf + np.sqrt(var_vf), color='darkred', alpha=0.3)
    
    ax2.set_ylabel("Altitud [msnm]")
    ax2.set_xlabel(f"Hora UTC $\\rightarrow$ {tiempos_raw[0].strftime('%Y-%b-%d')}")
    plt.colorbar(im2, ax=ax2, label="[m/s]")
    
    for ax in [ax1, ax2]:
        ax.set_ylim(0, 8000)
        ax.xaxis.set_major_formatter(dates.DateFormatter('%H:%M'))

    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=150)
    plt.close(fig)
