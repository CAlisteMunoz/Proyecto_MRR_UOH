import sys
from pathlib import Path
import warnings
import traceback

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

CONFIGURACIONES = [
    {"nombre": "1_Original_Proyecto", "tipo": "original",  "marco": 1},
    {"nombre": "2_Lineal_Estrecha",   "tipo": "lineal",    "marco": 3},
    {"nombre": "3_Gaussiana_Normal",  "tipo": "gaussiana", "marco": 5, "sigma": 1.5},
    {"nombre": "4_Lineal_Amplia",     "tipo": "lineal",    "marco": 7},
    {"nombre": "5_Gaussiana_Amplia",  "tipo": "gaussiana", "marco": 7, "sigma": 2.0}
]

def calcular_aceleracion_fisica(datos, conf):
    """
    Cálculo de Aceleración: Velocidad Inferior (Lluvia) - Velocidad Superior (Nieve).
    Genera un POSITIVO MÁXIMO exacto en la isoterma 0°C.
    """
    gradiente = np.zeros_like(datos) # Fondo 0 para que sea "Salmón" en la paleta RdBu
    m = conf['marco']
    t = conf['tipo']
    
    if t == 'original':
        for i in range(m, datos.shape[0] - m):
            inf = datos[i - m, :]
            sup = datos[i + m, :]
            gradiente[i, :] = inf - sup  # Aceleración real
    else:
        if t == 'lineal':
            w = np.array([m - j for j in range(m)], dtype=float)
        else:
            w = np.exp(-0.5 * (np.arange(m) / conf.get('sigma', 2.0))**2)
            
        w = w / np.sum(w)
        
        for i in range(m, datos.shape[0] - m):
            # Extraemos los bloques de datos y multiplicamos por los pesos de la ventana
            bloque_inf = datos[i-m : i, :][::-1, :] # Altitudes menores
            bloque_sup = datos[i+1 : i+1+m, :]      # Altitudes mayores
            
            inf = np.sum(bloque_inf * w[:, None], axis=0)
            sup = np.sum(bloque_sup * w[:, None], axis=0)
            
            gradiente[i, :] = inf - sup

    return gradiente

def ejecutar_replicacion():
    print("=== INICIANDO: DATOS CRUDOS Y ACELERACIÓN FÍSICA ===")
    
    archivos = list(obtener_archivos_por_año(2023, DATA_RAW))
    
    out_dir = PROJECT_ROOT / "results" / "gradientes"
    out_dir.mkdir(parents=True, exist_ok=True)

    eventos = 0
    for ruta in archivos:
        if eventos >= 10: break
            
        nombre = ruta.stem
        try:
            with leer_netcdf(ruta) as ds:
                ze_raw = np.asarray(ds['attenuated_radar_reflectivity'].values)
                if np.nansum(ze_raw > 15.0) < 50: 
                    continue
                    
                print(f"\n>> Procesando Evento: {nombre}...")
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

                # REPLICACIÓN DEL RUIDO ORIGINAL:
                # Todo lo que no es nube (Ze < 12) se vuelve 2.0 (Nieve constante).
                # Esto garantiza fondo 0 en el gradiente pero conserva el ruido crudo dentro de la nube.
                vf_f = np.where(ze_t >= 12.0, vf_t, 2.0)

                resultados = []
                for conf in CONFIGURACIONES:
                    grad_vf = calcular_aceleracion_fisica(vf_f, conf)
                    resultados.append({"nombre": conf["nombre"], "gradiente": grad_vf})

                # --- CONFIGURACIÓN DE PLOTEO PROPLOT ---
                total_seconds = (xlim[1] - xlim[0]).total_seconds()
                if total_seconds <= 14400:
                    xlocator, xminorlocator = ('hour', range(0, 24, 1)), ('minute', 30)
                elif total_seconds <= 82800.0:
                    xlocator, xminorlocator = ('hour', range(0, 24, 3)), ('hour', range(0, 24, 1))
                else:
                    xlocator, xminorlocator = ('hour', range(0, 24, 6)), ('hour', range(0, 24, 2))

                ylim = [0, 3600] if heights[-1] < 5000 else [0, 8000]
                extent = [dates.date2num(new_time[0]), dates.date2num(new_time[-1]), heights[0], heights[-1]]

                fig, axes = pplt.subplots(nrows=5, refwidth=5, refaspect=3, sharex=True, sharey=True)

                for i, ax in enumerate(axes):
                    res = resultados[i]
                    # Parámetros originales RdBu. Como es Aceleración, lo Positivo será Azul.
                    m = ax.imshow(res['gradiente'], origin='lower', aspect='auto',
                                  vmin=-3, vmax=10, cmap='RdBu', extent=extent)
                    
                    ax.format(ultitle=f"Filtro: {res['nombre']}",
                              xrotation=False, xformatter='concise',
                              xlocator=xlocator, xminorlocator=xminorlocator,
                              ylim=ylim, yticklabelloc='both', ytickloc='both', xticklabelsize=8)

                axes.format(
                    suptitle=f'Aceleración de Caída MRR UOH (Derivada Vf)\nEvento: {nombre}',
                    ylabel='Altitud [msnm]',
                    xlabel=r'Hora UTC $\rightarrow$'
                )

                # CORRECCIÓN DE UNIDADES DE MEDIDA
                fig.colorbar(m, loc='r', label='Aceleración [m/s / nivel]', length=0.4, extend='both')
                axes.format(xlim=xlim)

                out_file = out_dir / f"Comparacion_Gradientes_W_{nombre}.png"
                fig.savefig(out_file, dpi=150, bbox_inches='tight')
                plt.close(fig)

                print(f"   [OK] Lámina de Aceleración con ruido original guardada.")
                eventos += 1
                
        except Exception as e:
            print(f"   [ERROR CRÍTICO] Falló el archivo {nombre}:")
            traceback.print_exc()

    print("\n✅ ¡Lote generado! Física y datos crudos restaurados con éxito.")

if __name__ == '__main__':
    ejecutar_replicacion()
