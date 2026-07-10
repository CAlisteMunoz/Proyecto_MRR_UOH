import numpy as np

def filtrar_ruido(ze, vf, umbral_ze=12.0):
    # Todo lo menor al umbral se vuelve NaN para que el ploteo lo pinte gris (0.9)
    ze_f = np.where(ze >= umbral_ze, ze, np.nan)
    vf_f = np.where(ze >= umbral_ze, vf, np.nan)
    return ze_f, vf_f

def calcular_gradiente_avanzado(datos, marco=1, tipo_ventana='lineal', sigma=2.0):
    if tipo_ventana == 'uniforme': pesos = np.ones(marco)
    elif tipo_ventana == 'lineal': pesos = np.array([marco - i for i in range(marco)])
    elif tipo_ventana == 'gaussiana': pesos = np.exp(-0.5 * (np.arange(marco) / sigma)**2)
    elif tipo_ventana == 'hamming': pesos = np.hamming(marco)
    elif tipo_ventana == 'hanning': pesos = np.hanning(marco)
    else: raise ValueError("Ventana inválida")
        
    pesos = pesos / np.sum(pesos)
    gradiente_datos = np.full_like(datos, np.nan)
    
    for i in range(marco, datos.shape[0] - marco):
        sup = np.nansum([pesos[j] * datos[i + j, :] for j in range(marco)], axis=0)
        inf = np.nansum([pesos[j] * datos[i - j - 1, :] for j in range(marco)], axis=0)
        gradiente_datos[i, :] = sup - inf

    # Restauramos exactamente la máscara original de NaN para que el gráfico sea perfecto
    return np.where(np.isnan(datos), np.nan, gradiente_datos)
