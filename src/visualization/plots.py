import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors, dates
import numpy as np
import pandas as pd

# Paleta oficial del repositorio UOH
PALETA_ZE = ['#ccd8ff','#3366ff','#9fdf9f','#00b300','#ffff00','#ffcc30','#e62e00','#ff6600','#fff0e5','#c03fc0','#602060']

def generar_plot_mrr_dual(ds, iso_ze, var_ze, iso_vf, var_vf, ruta_salida, titulo):
    tiempos = pd.to_datetime(ds.time.values)
    # Ajuste de alturas msnm
    heights = (ds.height.values[0,:] if ds.height.ndim > 1 else ds.height.values) + 500
    
    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(12, 10), sharex=True)
    extent = [tiempos[0], tiempos[-1], heights[0], heights[-1]]
    
    # --- PANEL 1: Reflectividad Equivalente ---
    cmap_ze = colors.LinearSegmentedColormap.from_list("ze_uoh", PALETA_ZE)
    ze = ds['attenuated_radar_reflectivity'].values
    im1 = ax1.imshow(ze.T, origin='lower', aspect='auto', extent=extent, cmap=cmap_ze, vmin=-5, vmax=50)
    
    # Isoterma y sombreado de confianza
    ax1.plot(tiempos, iso_ze, color='black', linewidth=1.5, label='Isoterma 0°C detectada')
    ax1.fill_between(tiempos, iso_ze - np.sqrt(var_ze), iso_ze + np.sqrt(var_ze), 
                     color='darkred', alpha=0.3, label='Intervalo confianza Ze')
    
    ax1.set_title(f"Radar Perfilador MRR en UOH Rancagua", fontsize=16, fontweight='bold')
    ax1.text(0.02, 0.9, "Reflectividad Equivalente", transform=ax1.transAxes, fontsize=14)
    ax1.set_ylabel("Altitud [msnm]", fontsize=12)
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, frameon=True, shadow=True)
    plt.colorbar(im1, ax=ax1, label="[dBZ]")

    # --- PANEL 2: Velocidad de Caída ---
    vf = ds['fall_velocity'].values
    im2 = ax2.imshow(vf.T, origin='lower', aspect='auto', extent=extent, cmap='RdBu_r', vmin=-2, vmax=10)
    
    ax2.plot(tiempos, iso_vf, color='black', linewidth=1.5, label='Isoterma 0°C detectada')
    ax2.fill_between(tiempos, iso_vf - np.sqrt(var_vf), iso_vf + np.sqrt(var_vf), 
                     color='darkred', alpha=0.3, label='Intervalo confianza Vf')
    
    ax2.text(0.02, 0.9, "Velocidad de caída", transform=ax2.transAxes, fontsize=14)
    ax2.set_ylabel("Altitud [msnm]", fontsize=12)
    fecha_str = tiempos[0].strftime('%Y-%b-%d')
    ax2.set_xlabel(f"Hora UTC $\\rightarrow$ {fecha_str}", fontsize=12)
    plt.colorbar(im2, ax=ax2, label="[m/s]")
    
    # Ajustes finales de ejes (Idéntico a imagen 2)
    for ax in [ax1, ax2]:
        ax.set_ylim(0, 8000)
        ax.grid(alpha=0.2)
        ax.xaxis.set_major_formatter(dates.DateFormatter('%H:%M'))

    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=150, bbox_inches='tight')
    plt.close(fig)
