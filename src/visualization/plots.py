import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors, dates
import numpy as np

# Paleta oficial del repositorio de referencia
PALETA0 = ['#ccd8ff','#3366ff','#9fdf9f','#00b300','#ffff00','#ffcc30','#e62e00','#ff6600','#fff0e5','#c03fc0','#602060']

def generar_plot_isoterma(ds, iso_kalman, ruta_salida, titulo):
    # Configuración del colormap
    cmap = colors.LinearSegmentedColormap.from_list("custom", PALETA0)
    
    # Creación de la figura
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Extracción de datos
    ze = ds['attenuated_radar_reflectivity'].values
    # Ajuste de altitud msnm (+500m según sistematización UOH)
    heights = ds.height[0,:].values + 500
    times = dates.date2num(ds.time.values)
    
    # Definición de la extensión del gráfico
    extent = [times[0], times[-1], heights[0], heights[-1]]
    
    # Plot de reflectividad (Capa base)
    im = ax.imshow(ze.T, origin='lower', aspect='auto', 
                  extent=extent, cmap=cmap, vmin=-5, vmax=50, zorder=1)
    
    # Superposición de la Isoterma calculada
    ax.plot(times, iso_kalman, color='black', linewidth=1.5, zorder=2, label='Isoterma 0°C (Kalman)')
    
    # Formateo de ejes y etiquetas
    ax.set_title(f"Radar Perfilador MRR - {titulo}")
    ax.set_ylabel("Altitud [msnm]")
    ax.set_xlabel("Hora UTC")
    ax.set_ylim(0, 6000)
    ax.xaxis_date()
    
    # Añadir barra de colores
    plt.colorbar(im, label="Reflectividad [dBZ]")
    ax.legend(loc='upper right')
    
    # Guardado y limpieza de memoria
    plt.savefig(ruta_salida, dpi=150, bbox_inches='tight')
    plt.close(fig)
    plt.close('all')
