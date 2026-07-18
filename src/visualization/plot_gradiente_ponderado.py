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
from src.visualization.plot_gradientes_ventanas_pixeles import calcular_gradiente_original, ruido

def ejecutar_ponderado():
    print("=== INICIANDO: ANÁLISIS PONDERADO CON ECUACIÓN ===")
    
    archivos = list(obtener_archivos_por_año(2024, DATA_RAW))
    
    # Nueva carpeta para este análisis específico
    out_dir = PROJECT_ROOT / "results" / "gradientes_ponderados"
    out_dir.mkdir(parents=True, exist_ok=True)

    ruta = archivos[0] # Tomamos solo un evento representativo para esta lámina especial
    nombre = ruta.stem
    
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

            # Calculamos con ventana de 5 píxeles
            marco = 5
            grad_vf = calcular_gradiente_original(Vf_t, marco)

            extent = [dates.date2num(new_time[0]), dates.date2num(new_time[-1]), heights_ajustado[0], heights_ajustado[-1]]
            ylim = [0, 3600] if heights_ajustado[-1] < 5000 else [0, 8000]

            fig, ax = pplt.subplots(nrows=1, refwidth=6, refaspect=2.5)

            m = ax.imshow(grad_vf, origin='lower', aspect='auto',
                          vmin=-3, vmax=10, cmap='RdBu', extent=extent)
            
            ax.format(ultitle=f'Gradiente Ponderado (Ventana: {marco} píxeles)',
                      xrotation=False, xformatter='concise',
                      ylim=ylim, yticklabelloc='both', ytickloc='both',
                      suptitle=f'Análisis de Gradiente Normalizado - Evento: {nombre}',
                      ylabel='Altitud [msnm]', xlabel=r'Hora UTC $\rightarrow$')

            fig.colorbar(m, loc='r', label='Gradiente Vf [m/s]', length=0.7)

            # --- INYECCIÓN DE LA ECUACIÓN MATEMÁTICA ---
            # Se renderiza la ecuación de tu código en formato LaTeX
            ecuacion_texto = (
                r"$\nabla V_i = \sum_{j=0}^{m-1} w_j V_{i+j} - \sum_{j=0}^{m-1} w_j V_{i-j-1}$" + "\n\n" +
                r"$w_j = \frac{m-j}{\sum_{k=0}^{m-1} (m-k)}$"
            )
            
            # Colocamos el recuadro en la esquina superior derecha (coordenadas relativas de los ejes)
            ax.text(0.98, 0.95, ecuacion_texto, transform=ax.transAxes,
                    fontsize=10, verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray'))

            out_file = out_dir / f"Gradiente_Ponderado_Ecuacion_{nombre}.png"
            fig.savefig(out_file, dpi=200, bbox_inches='tight')
            plt.close(fig)

            print(f"   [OK] Lámina ponderada con ecuación generada en: {out_file.name}")
            
    except Exception as e:
        print(f"   [ERROR] Falló el procesamiento: {e}")

if __name__ == '__main__':
    ejecutar_ponderado()
