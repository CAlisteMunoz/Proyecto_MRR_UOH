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

# Configuraciones estrictas para las 5 ventanas
CONFIGURACIONES = [
    {"nombre": "1_Original_Proyecto", "tipo": "borde",     "marco": 1},
    {"nombre": "2_Lineal_Estrecha",   "tipo": "lineal",    "marco": 3},
    {"nombre": "3_Gaussiana_Normal",  "tipo": "gaussiana", "marco": 5, "sigma": 1.5},
    {"nombre": "4_Lineal_Amplia",     "tipo": "lineal",    "marco": 7},
    {"nombre": "5_Gaussiana_Amplia",  "tipo": "gaussiana", "marco": 7, "sigma": 2.0}
]

def calcular_gradiente_perfecto(datos, conf):
    """
    Replicación matemática exacta. Maneja los NaNs rigurosamente para 
    evitar falsos bordes y aplica la corrección microfísica literal.
    """
    gradiente = np.full_like(datos, np.nan)
    m = conf['marco']
    t = conf['tipo']
    
    if t == 'borde':
        for i in range(m, datos.shape[0] - m):
            gradiente[i, :] = datos[i + m, :] - datos[i - m, :]
    else:
        if t == 'lineal':
            w = np.array([m - j for j in range(m)], dtype=float)
        else:
            w = np.exp(-0.5 * (np.arange(m) / conf.get('sigma', 2.0))**2)
            
        w = w / np.sum(w)
        
        for i in range(m, datos.shape[0] - m):
            bloque_sup = datos[i+1 : i+1+m, :]
            bloque_inf = datos[i-m : i, :][::-1, :]
            
            # Usar np.sum normal (no nansum) asegura que si el bloque toca
            # el cielo vacío (NaN), no genere un falso gradiente gigante.
            sup = np.sum(bloque_sup * w[:, None], axis=0)
            inf = np.sum(bloque_inf * w[:, None], axis=0)
            
            gradiente[i, :] = sup - inf

    # --- LA CORRECCIÓN MICROFÍSICA EXACTA DE REPLICACION_REPO.PY ---
    # Validamos que el arreglo no sea puro NaN antes de buscar el mínimo
    if not np.all(np.isnan(gradiente)):
        min_val = np.nanmin(gradiente)
        if min_val < 0:
            gradiente = gradiente + np.abs(min_val) + 1.0

    return np.where(np.isnan(datos), np.nan, gradiente)

def ejecutar_replicacion():
    print("=== INICIANDO REPLICACIÓN 100% EXACTA Y SIN SILENCIOS ===")
    
    archivos = list(obtener_archivos_por_año(2023, DATA_RAW))
    print(f"-> Escaneando {len(archivos)} archivos NetCDF...")
    
    out_dir = PROJECT_ROOT / "results" / "gradientes"
    out_dir.mkdir(parents=True, exist_ok=True)

    eventos = 0
    for ruta in archivos:
        if eventos >= 10: break
            
        nombre = ruta.stem
        try:
            with leer_netcdf(ruta) as ds:
                ze_raw = np.asarray(ds['attenuated_radar_reflectivity'].values)
                
                # Filtro muy relajado para asegurar que procese datos
                if np.nansum(ze_raw > 10.0) < 10: 
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

                # Enmascaramiento: Lo menor a 12 dBZ se vuelve NaN (gris en el plot)
                vf_f = np.where(ze_t >= 12.0, vf_t, np.nan)

                resultados = []
                for conf in CONFIGURACIONES:
                    grad_vf = calcular_gradiente_perfecto(vf_f, conf)
                    resultados.append({"nombre": conf["nombre"], "gradiente": grad_vf})

                # --- CONFIGURACIÓN DE PROPLOT EXACTA ---
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
                    # Parámetros originales: RdBu, -3 a 10
                    m = ax.imshow(res['gradiente'], origin='lower', aspect='auto',
                                  vmin=-3, vmax=10, cmap='RdBu', extent=extent)
                    
                    ax.set_facecolor('0.9') # Fondo gris
                    ax.format(ultitle=f"Filtro: {res['nombre']}",
                              xrotation=False, xformatter='concise',
                              xlocator=xlocator, xminorlocator=xminorlocator,
                              ylim=ylim, yticklabelloc='both', ytickloc='both', xticklabelsize=8)

                axes.format(
                    suptitle=f'Comparativa Gradientes Vf MRR UOH\nEvento: {nombre}',
                    ylabel='Altitud [msnm]',
                    xlabel=r'Hora UTC $\rightarrow$'
                )

                fig.colorbar(m, loc='r', label='[m/s]', length=0.4, extend='both')
                axes.format(xlim=xlim)

                out_file = out_dir / f"Comparacion_Gradientes_W_{nombre}.png"
                fig.savefig(out_file, dpi=150, bbox_inches='tight')
                plt.close(fig)

                print(f"   [OK] Lámina de 5 ventanas guardada a la perfección.")
                eventos += 1
                
        except Exception as e:
            # AHORA EL SCRIPT GRITA SI HAY UN ERROR
            print(f"   [ERROR CRÍTICO] Falló el archivo {nombre}:")
            traceback.print_exc()

    print("\n✅ ¡Lote generado! Revisa la carpeta de resultados.")

if __name__ == '__main__':
    ejecutar_replicacion()
