import numpy as np
from filterpy.kalman import KalmanFilter

def filtrar_ruido(ze, vf):
    ze_filtered = np.where(ze >= 12, ze, 12)
    vf_filtered = np.where(ze >= 12, vf, 2)
    return ze_filtered, vf_filtered

def calcular_gradiente(datos, marco=5):
    """Réplica exacta de la lógica de gradientes del .txt"""
    pesos = np.array([marco - i for i in range(marco)])  
    pesos = pesos / np.sum(pesos)                        
    niveles_restantes = datos.shape[0] - 1 - 2 * (marco - 1)
    gradiente_datos = np.zeros((niveles_restantes, datos.shape[1]))

    for i in range(marco, datos.shape[0] - marco):
        superior = np.sum([pesos[j] * datos[i + j, :] for j in range(marco)], axis=0)
        inferior = np.sum([pesos[j] * datos[i - j - 1, :] for j in range(marco)], axis=0)
        gradiente_datos[i - marco, :] = superior - inferior

    return np.pad(gradiente_datos, ((marco, marco-1), (0, 0)), mode='constant', constant_values=0)

def aplicar_filtro_kalman(alturas_obs, gradiente, heights_ajustado, delta_t=1):
    """Implementación robusta del filtro del .txt sin errores de atributo"""
    f = KalmanFilter(dim_x=2, dim_z=1)
    f.x = np.array([alturas_obs[0], 0.])
    f.F = np.array([[1., delta_t], [0., 1.]])
    f.H = np.array([[1., 0.]])
    f.P = np.array([[500.0, 0.0], [0.0, 100.0]])
    f.Q = np.array([[0.1, 0], [0, 0.01]])
    f.R = 10.0

    alturas_filtradas = []
    altura_actual = np.median(alturas_obs)
    gradiente_norm = gradiente + (np.abs(gradiente.min()) + 1)
    
    # Convertir a numpy si es xarray para evitar el error .values
    h_arr = getattr(heights_ajustado, 'values', heights_ajustado)

    for t in range(len(alturas_obs)):
        distancias = np.abs(h_arr - altura_actual)
        dist_norm = (distancias / np.max(distancias)) + 1
        grad_pond = gradiente_norm[:, t] * (dist_norm * 2)
        
        idx_actual = int((np.abs(h_arr - altura_actual)).argmin())
        idx_sup = max(idx_actual - 1, 0)
        idx_inf = min(idx_actual + 1, len(h_arr) - 1)
        
        # Lógica de mínimo local (Fuente 62 del .txt)
        min_idx = idx_sup + np.argmin(grad_pond[idx_sup:idx_inf+1])
        medicion_altura = h_arr[min_idx] # CORRECCIÓN: Quitamos .values aquí

        f.predict()
        f.update(medicion_altura)
        altura_actual = f.x[0]
        alturas_filtradas.append(altura_actual)
    
    return np.array(alturas_filtradas)
