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

def aplicar_rolling_std(Vf_t, window_size, win_type=None, gaussian_std=None):
    df = pd.DataFrame(Vf_t.T)
    if win_type == 'gaussian':
        rolling = df.rolling(window=window_size, win_type=win_type, center=True).std(std=gaussian_std)
    else:
        rolling = df.rolling(window=window_size, win_type=win_type, center=True).std()
    return rolling.fillna(0).values.T

def generar_analisis_std():
    print("=== INICIANDO: ANÁLISIS DE DESVIACIÓN ESTÁNDAR POR TIPO DE VENTANA ===")
    dir_salida = PROJECT_ROOT / "results" / "analisis_estadistico"
    dir_salida.mkdir(parents=True, exist_ok=True)
    
    archivos = list(obtener_archivos_por_año(2023, DATA_RAW))
    evento_procesado = False
    
    for ruta in archivos:
        try:
            with leer_netcdf(ruta) as ds:
                Ze = ds['attenuated_radar_reflectivity']
                puntos_lluvia = np.count_nonzero(np.nan_to_num(Ze.values) > 12.0)
                
                if puntos_lluvia > 8000: 
                    print(f">> Analizando evento válido: {ruta.stem} ({puntos_lluvia} px)")
                    Vf = ds['fall_velocity']
                    
                    new_time = pd.to_datetime(ds.time.values) if hasattr(ds, 'time') else pd.to_datetime(ds.indexes['time'].astype(str))
                    xlim = [new_time[0], new_time[-1]]
                    h_vals = np.asarray(ds.height.values[0,:] if ds.height.ndim > 1 else ds.height.values)
                    heights_ajustado = h_vals + (500 + (h_vals[1] - h_vals[0]) / 2)
                    
                    Vf_filtered = np.where(Ze >= 12, Vf, 2)
                    Vf_t = Vf_filtered.T if Vf_filtered.shape[0] == len(new_time) else Vf_filtered
                    
                    win_size = 15
                    std_plana = aplicar_rolling_std(Vf_t, win_size, win_type=None)
                    std_gaussiana = aplicar_rolling_std(Vf_t, win_size, win_type='gaussian', gaussian_std=3)
                    std_hanning = aplicar_rolling_std(Vf_t, win_size, win_type='hanning')

                    extent = [dates.date2num(new_time[0]), dates.date2num(new_time[-1]), heights_ajustado[0], heights_ajustado[-1]]
                    fig, axs = pplt.subplots(nrows=3, refwidth=6, refaspect=3.5, sharex=True, sharey=True)
                    
                    matrices = [std_plana, std_gaussiana, std_hanning]
                    titulos = ['Ventana Plana (Rectangular)', 'Ventana Gaussiana', 'Ventana Hanning']
                    
                    for i in range(3):
                        m = axs[i].imshow(matrices[i], origin='lower', aspect='auto', vmin=0, vmax=2, cmap='viridis', extent=extent)
                        axs[i].format(ultitle=titulos[i])
                        axs[i].colorbar(m, loc='r', label='STD [m/s]', length=0.7)

                    axs.format(
                        suptitle=f'Desviación Estándar Temporal de Vf (Tamaño Ventana: {win_size})',
                        ylabel='Altitud [msnm]', xlabel='Hora UTC', xrotation=False, xformatter='concise', ylim=[0, 8000]
                    )
                    
                    fig.savefig(dir_salida / f"Analisis_STD_Ventanas_{ruta.stem}.png", dpi=200, bbox_inches='tight')
                    plt.close(fig)
                    print(f"   [ÉXITO] Análisis guardado en results/analisis_estadistico/")
                    evento_procesado = True
                    break 
                    
        except Exception as e:
            # Aquí aplicamos la buena práctica: avisamos del error pero continuamos con el siguiente archivo
            error_name = type(e).__name__
            if error_name == "OutOfBoundsDatetime" or error_name == "ValueError":
                print(f" [!] Saltando archivo {ruta.stem}: Variable de tiempo vacía o ilegible.")
            else:
                pass
            
    if not evento_procesado:
        print("No se encontró ningún evento que cumpla los requisitos en 2023.")

if __name__ == '__main__':
    generar_analisis_std()
