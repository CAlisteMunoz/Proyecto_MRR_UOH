import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors, dates
import numpy as np

PALETA_UOH = ['#ccd8ff','#3366ff','#9fdf9f','#00b300','#ffff00','#ffcc30','#e62e00','#ff6600','#fff0e5','#c03fc0','#602060']

def generar_grafico_maestro(xlim, times, heights_raw, Ze, Vf, isoterma_data, ruta_salida):
    """Genera la salida gráfica dual idéntica a la del laboratorio utilizando Matplotlib puro."""
    heights = np.asarray(heights_raw) + 500
    times_num = dates.date2num(times)
    
    cmap_list = list(zip(np.linspace(0, 1, len(PALETA_UOH)), PALETA_UOH))
    dbzmap = colors.LinearSegmentedColormap.from_list("custom_mrr", cmap_list)
    dbzmap.set_bad('0.9', 1)
    norm = colors.BoundaryNorm(np.arange(-5, 50, 1), dbzmap.N)
    
    dx = (times_num[-1] - times_num[0]) / (len(times_num) - 1) if len(times_num) > 1 else 0
    dy = (heights[-1] - heights[0]) / (len(heights) - 1) if len(heights) > 1 else 0
    extent = [times_num[0] - dx/2, times_num[-1] + dx/2, heights[0] - dy/2, heights[-1] + dy/2]
    
    fig, ax = plt.subplots(nrows=2, figsize=(11, 8), sharex=True, gridspec_kw={'hspace': 0.15})
    
    # --- PANEL 1: Reflectividad (Ze) ---
    mZe = ax[0].imshow(np.asarray(Ze).T, extent=extent, aspect='auto', origin='lower', norm=norm, cmap=dbzmap, interpolation='nearest')
    ax[0].plot(times_num, isoterma_data['iso_z'], color='black', linewidth=1.5, label='Isoterma 0°C detectada')
    ax[0].fill_between(times_num, isoterma_data['inf_z'], isoterma_data['sup_z'], color='darkred', alpha=0.5, label='Intervalo confianza Ze')
    ax[0].set_title('Reflectividad Equivalente', fontsize=12, fontweight='bold', loc='left')
    fig.colorbar(mZe, ax=ax[0], pad=0.02).set_label('[dBZ]')
    
    # --- PANEL 2: Velocidad de Caída (Vf) ---
    mVf = ax[1].imshow(np.asarray(Vf).T, extent=extent, aspect='auto', origin='lower', vmin=-3, vmax=10, cmap='RdBu', interpolation='nearest')
    ax[1].plot(times_num, isoterma_data['iso_v'], color='black', linewidth=1.5, label='Isoterma 0°C detectada')
    ax[1].fill_between(times_num, isoterma_data['inf_v'], isoterma_data['sup_v'], color='darkred', alpha=0.5, label='Intervalo confianza Vf')
    ax[1].set_title('Velocidad de caída', fontsize=12, fontweight='bold', loc='left')
    fig.colorbar(mVf, ax=ax[1], pad=0.02, extend='both').set_label('[m/s]')
    
    for a in ax:
        a.set_ylim(0, 8000)
        a.set_ylabel('Altitud [msnm]', fontsize=11)
        a.legend(loc='upper right', fontsize=9, framealpha=0.9)
        a.grid(True, linestyle=':', alpha=0.6)
        
    ax[1].set_xlim(dates.date2num(xlim[0]), dates.date2num(xlim[1]))
    ax[1].xaxis.set_major_formatter(dates.DateFormatter('%H:%M'))
    ax[1].set_xlabel(r'Hora UTC $\rightarrow$', fontsize=11)
    
    fig.suptitle('Radar Perfilador MRR en UOH Rancagua', fontsize=15, fontweight='bold', y=0.96)
    plt.savefig(ruta_salida, dpi=150, bbox_inches='tight')
    plt.close(fig)
