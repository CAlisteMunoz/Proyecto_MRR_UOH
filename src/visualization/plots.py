import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors, dates
import numpy as np

PALETA0 = ['#ccd8ff','#3366ff','#9fdf9f','#00b300','#ffff00','#ffcc30','#e62e00','#ff6600','#fff0e5','#c03fc0','#602060']

def generar_plot_isoterma(ds, iso_kalman, ruta_salida, titulo):
    cmap = colors.LinearSegmentedColormap.from_list("custom", PALETA0)
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ze = ds['attenuated_radar_reflectivity'].values
    heights = ds.height[0,:].values + 500
    times = dates.date2num(ds.time.values)
    
    extent = [times[0], times[-1], heights[0], heights[-1]]
    im = ax.imshow(ze.T, origin='lower', aspect='auto', 
                  extent=extent, cmap=cmap, vmin=-5, vmax=50)
    
    ax.plot(times, iso_kalman, color='black', linewidth=1.5)
    ax.set_title(f"MRR UOH - {titulo}")
    ax.set_ylabel("Altitud [msnm]")
    ax.set_ylim(0, 6000)
    ax.xaxis_date()
    
    plt.colorbar(im, label="Reflectividad [dBZ]")
    plt.savefig(ruta_salida, dpi=150, bbox_inches='tight')
    plt.close(fig)
    plt.close('all')
