import numpy as np
from filterpy.kalman import KalmanFilter

def filtrar_ruido(ze, vf, umbral_ze=12.0, valor_fondo_ze=12.0, valor_fondo_vf=2.0):
    ze_f = np.where(ze >= umbral_ze, ze, valor_fondo_ze)
    vf_f = np.where(ze >= umbral_ze, vf, valor_fondo_vf)
    return ze_f, vf_f

def calcular_gradiente_avanzado(datos, marco=5, tipo_ventana='gaussiana', sigma=2.0, umbral_min_w=None):
    if tipo_ventana == 'uniforme': pesos = np.ones(marco)
    elif tipo_ventana == 'lineal': pesos = np.array([marco - i for i in range(marco)])
    elif tipo_ventana == 'gaussiana': pesos = np.exp(-0.5 * (np.arange(marco) / sigma)**2)
    elif tipo_ventana == 'hamming': pesos = np.hamming(marco)
    elif tipo_ventana == 'hanning': pesos = np.hanning(marco)
    else: raise ValueError("Ventana inválida")
        
    pesos = pesos / np.sum(pesos)
    niveles_restantes = datos.shape[0] - 1 - 2 * (marco - 1)
    gradiente_datos = np.zeros((niveles_restantes, datos.shape[1]))
    
    for i in range(marco, datos.shape[0] - marco):
        sup = np.sum([pesos[j] * datos[i + j, :] for j in range(marco)], axis=0)
        inf = np.sum([pesos[j] * datos[i - j - 1, :] for j in range(marco)], axis=0)
        grad = sup - inf
        
        if umbral_min_w is not None:
            velocidad_actual = datos[i, :]
            grad = np.where(velocidad_actual < umbral_min_w, 0, grad)
            
        gradiente_datos[i - marco, :] = grad

    return np.pad(gradiente_datos, ((marco, marco-1), (0, 0)), mode='constant', constant_values=0)

def aplicar_filtro_kalman(gradiente, velocidades, heights, delta_t=1):
    q_var = 0.1      
    r_var_base = 25000.0  
    
    f = KalmanFilter(dim_x=2, dim_z=1)
    f.x = np.array([2400.0, 0.]) 
    f.F = np.array([[1., delta_t], [0., 1.]])
    f.H = np.array([[1., 0.]])
    f.P *= 100. 
    f.R = np.array([[r_var_base]])  
    f.Q = np.array([[q_var, 0], [0, q_var/10.0]]) 

    alturas_f = []
    varianzas_f = []
    h_arr = getattr(heights, 'values', heights)
    
    # MÁSCARA CLIMATOLÓGICA: Restringimos la búsqueda entre 1000m y 4000m
    idx_1000 = (np.abs(h_arr - 1000)).argmin()
    idx_4000 = (np.abs(h_arr - 4000)).argmin()

    for t in range(gradiente.shape[1]):
        f.predict()
        col_grad = gradiente[:, t]
        col_vel = velocidades[:, t]
        
        # PILAR 1: Filtro de Masa. Exigimos al menos ~150 metros (15 pixeles) de lluvia real.
        pixeles_lluvia = np.sum(col_vel > 2.5)
        
        if pixeles_lluvia > 15: 
            # PILAR 2: Búsqueda restringida a la zona físicamente posible
            grad_valido = col_grad[idx_1000:idx_4000]
            min_grad = np.min(grad_valido)
            
            if min_grad < -0.3:
                idx_local = np.argmin(grad_valido)
                medicion = h_arr[idx_1000 + idx_local]
                
                # PILAR 3: Confianza Dinámica Anti-Saltos
                innovacion = abs(medicion - f.x[0])
                # Si el salto es grande, aumentamos drásticamente el ruido de medición R
                # obligando al filtro a moverse suavemente hacia la curva sin teletransportarse
                f.R = np.array([[r_var_base + (innovacion ** 2) * 20]])
                
                f.update(medicion)
                
                # Control de desbordamiento de incertidumbre
                if f.P[0,0] > 500: f.P[0,0] = 500
        else:
            # Cielo vacío o nevada débil: Inercia térmica pura
            f.P[0,0] += 2.0 
            f.x[1] *= 0.85      
            
        # FAILSAFE ABSOLUTO
        f.x[0] = np.clip(f.x[0], 1000.0, 4200.0)
            
        alturas_f.append(f.x[0])
        varianzas_f.append(f.P[0, 0])
        
    return np.array(alturas_f), np.array(varianzas_f)
