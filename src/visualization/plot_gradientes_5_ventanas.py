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

# === VENTANAS ESTÁNDAR ===
CONFIGURACIONES = {
    "1_Lineal_Estrecha":   {"ventana": "lineal",    "marco": 3,  "sigma": None},
    "2_Lineal_Normal":     {"ventana": "lineal",    "marco": 5,  "sigma": None},
    "3_Gaussiana_Normal":  {"ventana": "gaussiana", "marco": 5,  "sigma": 1.5},
    "4_Lineal_Amplia":     {"ventana": "lineal",    "marco": 7,  "sigma": None},
    "5_Gaussiana_Amplia":  {"ventana": "gaussiana", "marco": 7,  "sigma": 2.0}
}

def calcular_gradiente_literal(datos, marco=3, tipo_ventana='lineal', sigma=2.0):
    """
    Cálculo del gradiente idéntico al de replicacion_repo.py.
    Aplica Superior - Inferior, y luego la corrección microfísica 
    para desplazar el mapa de colores RdBu y obtener la banda rojiza.
    """
    if tipo_ventana == 'uniforme': pesos = np.ones(marco)
    elif tipo_ventana == 'lineal': pesos = np.array([marco - i for i in range(marco)])
    elif tipo_ventana == 'gaussiana': pesos = np.exp(-0.5 * (np.arange(marco) / sigma)**2)
    elif tipo_ventana == 'hamming': pesos = np.hamming(marco)
    elif tipo_ventana == 'hanning': pesos = np.hanning(marco)
    else: pesos = np.ones(marco)
        
    pesos = pesos / np.sum(pesos)
    
    # Inicialización en ceros como en tu proyecto original
    gradiente_datos = np.zeros_like(datos)
    
    for i in range(marco, datos.shape[0] - marco):
        sup = np.nansum([pesos[j] * datos[i + j + 1, :] for j in range(marco)], axis=0)
        inf = np.nansum([pesos[j] * datos[i - j - 1, :] for j in range(marco)], axis=0)
        
        # Matemática Original (Superior - Inferior)
        gradiente_datos[i, :] = sup - inf

    # LA CORRECCIÓN EXACTA (Responsable del color rojo en RdBu)
    min_val = np.nanmin(gradiente_datos)
    if min_val < 0:
        gradiente_datos = gradiente_datos + np.abs(min_val) + 1.0

    # Enmascaramos los NaNs para que proplot los pinte de gris
    return np.where(np.isnan(datos), np.nan, gradiente_datos)

def ejecutar_replicacion_literal():
    print("=== INICIANDO REPLICACIÓN 100% EXACTA AL PROYECTO ORIGINAL ===")
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
                
                valid_pixels = np.nansum(ze_raw > 15.0)
                if valid_pixels < 50: 
                    continue
                    
                print(f">> Replicando Evento: {nombre}...")
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

                # Generamos el filtro usando Ze
                vf_f = np.where(ze_t >= 12.0, vf_t, np.nan)

                resultados = []
                for nom_config, params in CONFIGURACIONES.items():
                    grad_vf = calcular_gradiente_literal(
                        vf_f, marco=params["marco"], 
                        tipo_ventana=params["ventana"], sigma=params["sigma"]
                    )
                    resultados.append({"nombre": nom_config, "gradiente": grad_vf})

                # --- PLOTEO PROPLOT LITERAL (Solo Vf, 5 Paneles) ---
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
                    # Parámetros intactos de codigo_plot.txt
                    m = ax.imshow(res['gradiente'], origin='lower', aspect='auto',
                                  vmin=-3, vmax=10, cmap='RdBu', extent=extent)
                    
                    ax.set_facecolor('0.9') 
                    ax.format(ultitle=f"Filtro: {res['nombre']}",
                              xrotation=False, xformatter='concise',
                              xlocator=xlocator, xminorlocator=xminorlocator,
                              ylim=ylim, yticklabelloc='both', ytickloc='both', xticklabelsize=8)

                axes.format(
                    suptitle=f'Gradientes Datos MRR UOH - Solo Velocidad de Caída\nEvento: {nombre}',
                    ylabel='Altitud [msnm]',
                    xlabel=r'Hora UTC $\rightarrow$'
                )

                fig.colorbar(m, loc='r', label='[m/s]', length=0.4, extend='both')
                axes.format(xlim=xlim)

                out_file = out_dir / f"Replica_Gradientes_W_{nombre}.png"
                fig.savefig(out_file, dpi=150, bbox_inches='tight')
                plt.close(fig)

                print(f"   [OK] Lámina replicada y guardada.")
                eventos += 1
                
        except Exception as e:
            print(f"   [ERROR] Falló en {nombre}: {e}")

    print("\n✅ ¡Ejecución completa: Replicación exacta terminada!")

if __name__ == '__main__':
    ejecutar_replicacion_literal()
