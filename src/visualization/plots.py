import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors, dates
import numpy as np
import pandas as pd

def generar_plot_5_ventanas_w(ds, vf, resultados, ruta_salida, titulo_evento):
    tiempos_raw = pd.to_datetime(ds.time.values)
    tiempos_num = dates.date2num(tiempos_raw)
    heights = (ds.height.values[0,:] if ds.height.ndim > 1 else ds.height.values) + 500
    extent = [tiempos_num[0], tiempos_num[-1], heights[0], heights[-1]]
    
    # Crear 5 subplots apilados
    fig, axes = plt.subplots(nrows=5, figsize=(14, 18), sharex=True)
    fig.suptitle(f"Análisis de Velocidad de Caída (W) y Seguimiento Isoterma 0°C\nEvento: {titulo_evento}", fontsize=18, fontweight='bold', y=0.92)
    
    for ax, res in zip(axes, resultados):
        # Fondo de radar (Velocidad)
        im = ax.imshow(vf, origin='lower', aspect='auto', extent=extent, cmap='RdBu_r', vmin=-2, vmax=10)
        
        # Filtro de Kalman: Línea negra gruesa e Incertidumbre amarilla brillante
        ax.plot(tiempos_num, res['iso'], color='black', linewidth=2.5, label='Predicción Filtro Kalman')
        std_dev = np.sqrt(res['var'])
        ax.fill_between(tiempos_num, res['iso'] - std_dev, res['iso'] + std_dev, color='yellow', alpha=0.6, label='Incertidumbre ($\\sigma$)')
        
        # Estética de cada subplot
        ax.set_title(f"Configuración: {res['nombre']}", fontsize=14, fontweight='bold', loc='left')
        ax.set_ylabel("Altitud [msnm]")
        ax.set_ylim(500, 6000) # Centrado en la zona de interés
        ax.grid(True, linestyle='--', alpha=0.4)
        if res == resultados[0]:
            ax.legend(loc='upper right', framealpha=0.9)
        
    # Eje X global
    axes[-1].set_xlabel(f"Hora UTC $\\rightarrow$ {tiempos_raw[0].strftime('%Y-%b-%d')}", fontsize=14)
    axes[-1].xaxis.set_major_formatter(dates.DateFormatter('%H:%M'))
    
    # Barra de color global a la derecha
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Velocidad de Caída [m/s]")
    
    plt.tight_layout(rect=[0, 0, 0.9, 0.9])
    plt.savefig(ruta_salida, dpi=150, bbox_inches='tight')
    plt.close(fig)
