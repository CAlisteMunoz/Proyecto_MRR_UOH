import numpy as np
from filterpy.kalman import KalmanFilter

def filtrar_ruido(ze, vf, umbral_ze=12.0, valor_fondo_ze=12.0, valor_fondo_vf=2.0):
    """Filtra ecos parásitos bajo un umbral de dBZ."""
    ze_f = np.where(ze >= umbral_ze, ze, valor_fondo_ze)
    vf_f = np.where(ze >= umbral_ze, vf, valor_fondo_vf)
    return ze_f, vf_f

def calcular_gradiente_avanzado(datos, marco=5, tipo_ventana='gaussiana', sigma=2.0, umbral_min_w=None):
    """
    Calcula el gradiente espacial utilizando núcleos de convolución intercambiables.
    Optimizado para operaciones vectorizadas en matrices de radar.
    """
    # 1. Definición del núcleo (kernel) de pesos
    if tipo_ventana == 'uniforme':
        pesos = np.ones(marco)
    elif tipo_ventana == 'lineal':
        pesos = np.array([marco - i for i in range(marco)])
    elif tipo_ventana == 'gaussiana':
        pesos = np.exp(-0.5 * (np.arange(marco) / sigma)**2)
    else:
        raise ValueError("tipo_ventana debe ser 'uniforme', 'lineal' o 'gaussiana'")
        
    pesos = pesos / np.sum(pesos)
    
    # 2. Preparación del tensor
    niveles_restantes = datos.shape[0] - 1 - 2 * (marco - 1)
    gradiente_datos = np.zeros((niveles_restantes, datos.shape[1]))
    
    # 3. Convolución (Diferencias finitas ponderadas)
    for i in range(marco, datos.shape[0] - marco):
        sup = np.sum([pesos[j] * datos[i + j, :] for j in range(marco)], axis=0)
        inf = np.sum([pesos[j] * datos[i - j - 1, :] for j in range(marco)], axis=0)
        
        grad = sup - inf
        
        # 4. Umbralización Cinemática (Filtro para la Velocidad de Caída W)
        if umbral_min_w is not None:
            velocidad_actual = datos[i, :]
            # Si la velocidad bruta en este gate no alcanza el umbral, anulamos el gradiente
            grad = np.where(velocidad_actual < umbral_min_w, 0, grad)
            
        gradiente_datos[i - marco, :] = grad

    return np.pad(gradiente_datos, ((marco, marco-1), (0, 0)), mode='constant', constant_values=0)

def aplicar_filtro_kalman(alturas_obs, gradiente, heights, delta_t=1, q_var=0.5, r_var=10.0):
    """Aplica el Filtro de Kalman con matrices Q y R expuestas como parámetros."""
    f = KalmanFilter(dim_x=2, dim_z=1)
    f.x = np.array([alturas_obs[0], 0.])
    f.F = np.array([[1., delta_t], [0., 1.]])
    f.H = np.array([[1., 0.]])
    f.P *= 500. 
    
    f.R = np.array([[r_var]])  
    f.Q = np.array([[q_var, 0], [0, q_var/10.0]]) 

    alturas_f = []
    varianzas_f = []
    h_arr = getattr(heights, 'values', heights)

    for t in range(len(alturas_obs)):
        f.predict()
        idx = int((np.abs(h_arr - f.x[0])).argmin())
        idx_s, idx_i = max(idx-1, 0), min(idx+1, len(h_arr)-1)
        medicion = h_arr[idx_s + np.argmin(gradiente[idx_s:idx_i+1, t])]
        
        f.update(medicion)
        alturas_f.append(f.x[0])
        varianzas_f.append(f.P[0, 0])
        
    return np.array(alturas_f), np.array(varianzas_f)
