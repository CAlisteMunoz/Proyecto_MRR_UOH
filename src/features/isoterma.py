import numpy as np
from filterpy.kalman import KalmanFilter

def filtrar_ruido(ze, vf, umbral_ze=12.0, valor_fondo_ze=12.0, valor_fondo_vf=2.0):
    # Todo lo menor al umbral se vuelve NaN para el enmascaramiento visual
    ze_f = np.where(ze >= umbral_ze, ze, np.nan)
    vf_f = np.where(ze >= umbral_ze, vf, np.nan)
    return ze_f, vf_f

def calcular_gradiente_avanzado(datos, marco=1, tipo_ventana='lineal', sigma=2.0, umbral_min_w=None):
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
        grad = sup - inf
        
        if umbral_min_w is not None:
            velocidad_actual = datos[i, :]
            grad = np.where(velocidad_actual < umbral_min_w, np.nan, grad)
            
        gradiente_datos[i, :] = grad

    # REPLICACIÓN EXACTA: Restauramos la máscara de NaN del fondo para el ploteo
    gradiente_datos = np.where(np.isnan(datos), np.nan, gradiente_datos)
    
    return gradiente_datos

def aplicar_filtro_kalman(gradiente, velocidades, heights, delta_t=1):
    q_var = 0.5
    r_var = 15000.0
    
    f = KalmanFilter(dim_x=2, dim_z=1)
    f.x = np.array([2400.0, 0.]) 
    f.F = np.array([[1., delta_t], [0., 1.]])
    f.H = np.array([[1., 0.]])
    f.P *= 1000. 
    f.R = np.array([[r_var]])  
    f.Q = np.array([[q_var, 0], [0, q_var/10.0]]) 

    alturas_f = []
    varianzas_f = []
    h_arr = getattr(heights, 'values', heights)

    for t in range(gradiente.shape[1]):
        f.predict()
        col_grad = gradiente[:, t]
        
        if not np.all(np.isnan(col_grad)):
            min_grad = np.nanmin(col_grad)
            if min_grad < -0.3:
                idx_medicion = np.nanargmin(col_grad)
                medicion = h_arr[idx_medicion]
                
                if f.P[0,0] > 50.0 or abs(medicion - f.x[0]) < 1500:
                    f.update(medicion)
                else:
                    f.P[0,0] += 2.0
                    f.x[1] *= 0.5 
            else:
                f.P[0,0] += 1.0 
                f.x[1] *= 0.85 
        else:
            f.P[0,0] += 1.0 
            f.x[1] *= 0.85 
            
        f.x[0] = np.clip(f.x[0], 1000.0, 4800.0)
            
        alturas_f.append(f.x[0])
        varianzas_f.append(f.P[0, 0])
        
    return np.array(alturas_f), np.array(varianzas_f)
