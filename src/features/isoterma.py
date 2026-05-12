import numpy as np
from filterpy.kalman import KalmanFilter

def filtrar_ruido(ze, vf):
    ze_filtered = np.where(ze >= 12, ze, 12)
    vf_filtered = np.where(ze >= 12, vf, 2)
    return ze_filtered, vf_filtered

def ajustar_largo(array, n):
    array = np.array(array)
    if len(array) < n:
        return np.pad(array, (n - len(array), 0), mode='edge')
    return array[:n]

def calcular_gradiente(datos, marco=5):
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

    for t in range(len(alturas_obs)):
        distancias = np.abs(heights_ajustado - altura_actual)
        dist_norm = (distancias / np.max(distancias)) + 1
        grad_pond = gradiente_norm[:, t] * (dist_norm * 2)
        
        idx_actual = int((np.abs(heights_ajustado - altura_actual)).argmin())
        idx_sup = max(idx_actual - 1, 0)
        idx_inf = min(idx_actual + 1, len(heights_ajustado) - 1)
        
        min_idx = idx_sup + np.argmin(grad_pond[idx_sup:idx_inf+1])
        medicion_altura = heights_ajustado[min_idx].values

        f.predict()
        f.update(medicion_altura)
        altura_actual = f.x[0]
        alturas_filtradas.append(altura_actual)
    
    return np.array(alturas_filtradas)

def reporte_sistematizacion(iso_ze, iso_vf, dia):
    diferencias = iso_ze - iso_vf
    correlacion = np.corrcoef(iso_ze, iso_vf)[0,1]
    print(f"--- Reporte Sistematizacion: {dia} ---")
    print(f"Correlacion Ze/Vf: {correlacion:.3f}")
    print(f"Diferencia Promedio: {np.mean(diferencias):.2f} m")
    print(f"Desviacion Estandar: {np.std(diferencias):.2f} m")
    print(f"Casos extremos (>500m): {np.sum(np.abs(diferencias) > 500)}")
