import numpy as np
from filterpy.kalman import KalmanFilter

def filtrar_ruido(ze, vf):
    ze_f = np.where(ze >= 12, ze, 12)
    vf_f = np.where(ze >= 12, vf, 2)
    return ze_f, vf_f

def calcular_gradiente(datos, marco=5):
    pesos = np.array([marco - i for i in range(marco)])  
    pesos = pesos / np.sum(pesos)                        
    niveles_restantes = datos.shape[0] - 1 - 2 * (marco - 1)
    gradiente_datos = np.zeros((niveles_restantes, datos.shape[1]))
    for i in range(marco, datos.shape[0] - marco):
        sup = np.sum([pesos[j] * datos[i + j, :] for j in range(marco)], axis=0)
        inf = np.sum([pesos[j] * datos[i - j - 1, :] for j in range(marco)], axis=0)
        gradiente_datos[i - marco, :] = sup - inf
    return np.pad(gradiente_datos, ((marco, marco-1), (0, 0)), mode='constant', constant_values=0)

def aplicar_filtro_kalman(alturas_obs, gradiente, heights, delta_t=1):
    f = KalmanFilter(dim_x=2, dim_z=1)
    f.x = np.array([alturas_obs[0], 0.])
    f.F = np.array([[1., delta_t], [0., 1.]])
    f.H = np.array([[1., 0.]])
    f.P *= 500. # Incertidumbre inicial
    f.R = 10.0  # Ruido de medición
    f.Q = np.array([[0.1, 0], [0, 0.01]]) # Ruido de proceso

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
