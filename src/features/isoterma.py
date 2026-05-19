import numpy as np
from filterpy.kalman import KalmanFilter

def filtrar_ruido(Ze, Vf):
    """Filtra la reflectividad y velocidad de caída aplicando los umbrales base."""
    Ze_np = np.asarray(Ze, dtype=np.float64)
    Vf_np = np.asarray(Vf, dtype=np.float64)
    
    Ze_filtrado = np.where(Ze_np >= 12, Ze_np, 12)
    Vf_filtrado = np.where(Ze_np >= 12, Vf_np, 2)
    return Ze_filtrado, Vf_filtrado

def calcular_gradiente(datos, marco=1):
    """Calcula el gradiente vertical ponderado sobre una matriz (Alturas, Tiempos)."""
    datos_np = np.asarray(datos, dtype=np.float64)
    pesos = np.array([marco - i for i in range(marco)])
    if np.sum(pesos) != 0: 
        pesos = pesos / np.sum(pesos)
        
    niveles_restantes = datos_np.shape[0] - 1 - 2 * (marco - 1)
    gradiente = np.zeros((niveles_restantes, datos_np.shape[1]))

    for i in range(marco, datos_np.shape[0] - marco):
        superior = np.sum([pesos[j] * datos_np[i + j, :] for j in range(marco)], axis=0)
        inferior = np.sum([pesos[j] * datos_np[i - j - 1, :] for j in range(marco)], axis=0)
        gradiente[i - marco, :] = superior - inferior

    return np.pad(gradiente, ((marco, marco-1), (0, 0)), mode='constant', constant_values=0)

def detectar_isoterma_cero(gradiente, altura_inicial, heights_ajustado):
    """Búsqueda inicial aproximada de la isoterma basada en el gradiente mínimo."""
    resultados = []
    altura_actual = altura_inicial
    grad = np.asarray(gradiente, dtype=np.float64)
    h_adj = np.asarray(heights_ajustado, dtype=np.float64)

    for t in range(grad.shape[1]):
        idx_altura = int((np.abs(h_adj - altura_actual)).argmin())
        grad_minimo = grad[idx_altura, t]
        alt_minima = h_adj[idx_altura]
        
        idx_superior = max(idx_altura - 1, 0)
        idx_inferior = min(idx_altura + 1, len(h_adj) - 1)

        for i in range(idx_superior, idx_inferior + 1):
            if grad[i, t] < grad_minimo:
                grad_minimo = grad[i, t]
                alt_minima = h_adj[i]

        altura_actual = (1 - 0.5) * altura_actual + 0.5 * alt_minima
        resultados.append({"iter": t, "altura_minima": altura_actual})

    return resultados

def filtro_kalman_ponderado(alturas, gradiente, heights_ajustado, delta_t=20):
    """Filtro de Kalman con ponderación según distancia vertical al estado previo."""
    f = KalmanFilter(dim_x=2, dim_z=1)
    f.x = np.array([alturas[0], 0.])
    f.F = np.array([[1., delta_t], [0., 1.]])
    f.H = np.array([[1., 0.]])
    f.P = np.array([[500.0, 0.0], [0.0, 100.0]])
    f.Q = np.array([[10, 0], [0, 0.05]])
    f.R = 5000.0

    alturas_filtradas = [f.x[0]]
    var_altura = [f.P[0, 0]]
    altura_actual = np.median(alturas)
    
    grad = np.asarray(gradiente, dtype=np.float64)
    h_adj = np.asarray(heights_ajustado, dtype=np.float64)
    grad = grad + (np.abs(grad.min()) + 1)

    for t in range(len(alturas)):
        distancias = np.abs(h_adj - altura_actual)
        dist_norm = (distancias / np.max(distancias)) + 1
        grad_pond = grad[:, t] * (dist_norm * 2)
        
        idx_altura = int((np.abs(h_adj - altura_actual)).argmin())
        grad_min = grad_pond[idx_altura]
        alt_min = h_adj[idx_altura]
        
        idx_s = max(idx_altura - 1, 0)
        idx_i = min(idx_altura + 1, len(h_adj) - 1)
        
        for i in range(idx_s, idx_i + 1):
            if grad_pond[i] < grad_min:
                grad_min = grad_pond[i]
                alt_min = h_adj[i]
        
        f.predict()
        f.update(alt_min)
        altura_actual = f.x[0]
        alturas_filtradas.append(f.x[0])
        var_altura.append(f.P[0, 0])
        
    return np.array(alturas_filtradas)[1:], np.array(var_altura)[1:]

def ajustar_largo(array, N):
    array = np.array(array)
    if len(array) < N: return np.pad(array, (N - len(array), 0), mode='edge')
    elif len(array) > N: return array[:N]
    return array

def procesar_dia_completo(Ze_raw, Vf_raw, heights_raw, N_times):
    """Pipeline completo estructurado para procesar un archivo NetCDF."""
    h_base = np.asarray(heights_raw) + 500
    Ze_f, Vf_f = filtrar_ruido(Ze_raw, Vf_raw)
    
    alt_desfase = 500 + (h_base[1] - h_base[0]) / 2
    h_ajustado = np.asarray(heights_raw) + alt_desfase
    
    grad_z = calcular_gradiente(Ze_f.T, marco=1)
    grad_v = calcular_gradiente(Vf_f.T, marco=1)
    
    res_z = detectar_isoterma_cero(grad_z, 2000, h_ajustado)
    res_v = detectar_isoterma_cero(grad_v, 2000, h_ajustado)
    
    factor = (h_base[-1] - 500) / len(h_base)
    alt_z = (np.array([int((np.abs(h_ajustado - r["altura_minima"])).argmin()) for r in res_z]) * factor) + alt_desfase
    alt_v = (np.array([int((np.abs(h_ajustado - r["altura_minima"])).argmin()) for r in res_v]) * factor) + alt_desfase
    
    iso_z, var_z = filtro_kalman_ponderado(alt_z, grad_z, h_ajustado)
    iso_v, var_v = filtro_kalman_ponderado(alt_v, grad_v, h_ajustado)
    
    return {
        'iso_z': ajustar_largo(iso_z, N_times),
        'sup_z': ajustar_largo(iso_z + np.sqrt(var_z), N_times),
        'inf_z': ajustar_largo(iso_z - np.sqrt(var_z), N_times),
        'iso_v': ajustar_largo(iso_v, N_times),
        'sup_v': ajustar_largo(iso_v + np.sqrt(var_v), N_times),
        'inf_v': ajustar_largo(iso_v - np.sqrt(var_v), N_times),
    }
