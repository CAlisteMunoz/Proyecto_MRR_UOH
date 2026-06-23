import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import dates
from config import DATA_RAW, PROJECT_ROOT
from src.data.loader import obtener_archivos_por_año, leer_netcdf
from src.features.isoterma import calcular_gradiente_avanzado, filtrar_ruido

# === SELECCIÓN DEFINITIVA DE 5 VENTANAS (NORMALES VS AMPLIAS) ===
# Normales: marco = 7 | Amplias: marco = 15
CONFIGURACIONES = {
    "1_Lineal_Normal":    {"ventana": "lineal",    "marco": 7,  "sigma": None},
    "2_Uniforme_Normal":  {"ventana": "uniforme",  "marco": 7,  "sigma": None},
    "3_Gaussiana_Normal": {"ventana": "gaussiana", "marco": 7,  "sigma": 2.0},
    "4_Lineal_Amplia":    {"ventana": "lineal",    "marco": 15, "sigma": None},
    "5_Gaussiana_Amplia": {"ventana": "gaussiana", "marco": 15, "sigma": 5.0}
}

def generar_plot_5_gradientes(ds, resultados_gradientes, ruta_salida, nombre_evento):
    tiempos_raw = pd.to_datetime(ds.time.values)
    tiempos_num = dates.date2num(tiempos_raw)
    h = (ds.height.values[0,:] if ds.height.ndim > 1 else ds.height.values) + 500
    extent = [tiempos_num[0], tiempos_num[-1], h[0], h[-1]]

    fig, axes = plt.subplots(nrows=5, figsize=(14, 18), sharex=True)
    fig.suptitle(f"Comparación de Gradientes ($\\nabla W$): Normales vs Amplias\nEvento: {nombre_evento}", 
                 fontsize=18, fontweight='bold', y=0.92)

    for ax, res in zip(axes, resultados_gradientes):
        im = ax.imshow(res['gradiente'], origin='lower', aspect='auto', extent=extent, 
                       cmap='RdBu_r', vmin=-2.5, vmax=2.5)
        
        ax.set_title(f"Filtro: {res['nombre']}", fontsize=14, fontweight='bold', loc='left')
        ax.set_ylabel("Altitud [msnm]")
        ax.set_ylim(500, 4500) 
        ax.grid(True, linestyle='--', alpha=0.4)

    axes[-1].set_xlabel(f"Hora UTC $\\rightarrow$ {tiempos_raw[0].strftime('%Y-%b-%d')}", fontsize=14)
    axes[-1].xaxis.set_major_formatter(dates.DateFormatter('%H:%M'))

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Tasa de Cambio ($\\nabla$ m/s)", fontsize=12)

    plt.tight_layout(rect=[0, 0, 0.9, 0.9])
    plt.savefig(ruta_salida, dpi=150, bbox_inches='tight')
    plt.close(fig)

def ejecutar_comparacion_gradientes():
    print("=== GENERANDO PANELES COMPARATIVOS DE GRADIENTES ===")
    archivos = obtener_archivos_por_año(2023, DATA_RAW)
    
    ruta_muestra = None
    for ruta in archivos:
        try:
            with leer_netcdf(ruta) as ds:
                if np.sum(ds['attenuated_radar_reflectivity'].values > 20.0) > 100:
                    ruta_muestra = ruta
                    break
        except Exception: pass

    if not ruta_muestra:
        print("❌ Error: No se encontró un evento de lluvia válido.")
        return

    nombre = ruta_muestra.stem
    out_dir = PROJECT_ROOT / "results" / "gradientes"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"Comparacion_Gradientes_W_{nombre}.png"

    with leer_netcdf(ruta_muestra) as ds:
        ze_raw, vf_raw = ds['attenuated_radar_reflectivity'].values, ds['fall_velocity'].values
        _, vf_c = filtrar_ruido(ze_raw.T if ze_raw.shape[0] == len(ds.time) else ze_raw,
                                vf_raw.T if vf_raw.shape[0] == len(ds.time) else vf_raw)

        resultados = []
        for nombre_config, params in CONFIGURACIONES.items():
            grad_vf = calcular_gradiente_avanzado(
                vf_c, 
                marco=params["marco"], 
                tipo_ventana=params["ventana"], 
                sigma=params["sigma"]
            )
            resultados.append({"nombre": nombre_config, "gradiente": grad_vf})

        generar_plot_5_gradientes(ds, resultados, out_file, nombre)
        print(f"✅ ¡Panel comparativo generado en: {out_file}!")

if __name__ == '__main__':
    ejecutar_comparacion_gradientes()
