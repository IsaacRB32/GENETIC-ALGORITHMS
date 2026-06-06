import random
import numpy as np

t = np.array([
    [ 0.0 , 61.82, 18.54, 37.52, 54.08,  1.88, 59.98, 32.82, 69.42, 36.76, 60.26],
    [61.82,  0.0 , 50.84, 33.62,  7.5 , 59.88,  2.76, 28.84,  7.78, 28.14,  5.8 ],
    [18.54, 50.84,  0.0 , 26.74, 43.38, 18.6 , 49.28, 22.0 , 58.7 , 23.36, 49.3 ],
    [37.52, 33.62, 26.74,  0.0 , 26.16, 35.56, 32.06,  4.8 , 41.5 ,  3.26, 32.08],
    [54.08,  7.5 , 43.38, 26.16,  0.0 , 52.06,  7.32, 21.38, 15.34, 20.68,  5.92],
    [ 1.88, 59.88, 18.6 , 35.56, 52.06,  0.0 , 57.96, 30.86, 67.38, 34.8 , 58.3 ],
    [59.98,  2.76, 49.28, 32.06,  7.32, 57.96,  0.0 , 27.28, 10.62, 26.58,  6.76],
    [32.82, 28.84, 22.0 ,  4.8 , 21.38, 30.86, 27.28,  0.0 , 36.72,  4.02, 27.3 ],
    [69.42,  7.78, 58.7 , 41.5 , 15.34, 67.38, 10.62, 36.72,  0.0 , 36.02, 12.14],
    [36.76, 28.14, 23.36,  3.26, 20.68, 34.8 , 26.58,  4.02, 36.02,  0.0 , 26.6 ],
    [60.26,  5.8 , 49.3 , 32.08,  5.92, 58.3 ,  6.76, 27.3 , 12.14, 26.6 ,  0.0 ]
])

Tw = [
    [-float('inf'), float('inf')],
    [50, 90],
    [15, 25],
    [30, 55],
    [15, 75],
    [5, 35],
    [150, 200],
    [25, 50],
    [65, 100],
    [120, 150],
    [30, 85]
]


def generar_poblacion_inicial(tamano_poblacion, n_ciudades):
    ##Aquí se guardarán la población inicial
    poblacion = []
    
    ## Creamos una lista con las ciudades de 1 a n
    ciudades = list(range(1, n_ciudades + 1))
    
    for _ in range(tamano_poblacion):
        individuo = ciudades.copy()
        # Desordenamos la lista de forma aleatoria
        random.shuffle(individuo)

        individuo.insert(0, 0)

        poblacion.append(individuo)
        
    return poblacion

def evaluar_aptitud_tsptw(ruta, t, Tw, lambda_penal=10):
    T = [0] * (len(ruta) + 1) # T(x1) = 0 al inicio
    penalizacion_total = 0
    
    # Iteramos sobre las ciudades para calcular la secuencia de tiempos
    for i in range(len(ruta)):
        origen = ruta[i]
        
        ## Si es el último nodo, el destino es el primero (ruta[0])
        if i == len(ruta) - 1:
            destino = ruta[0]
        else:
            destino = ruta[i + 1]

        ## Tiempo estimado de llegada = Tiempo actual + tiempo de viaje entre origen y destino
        llegada_estimada = T[i] + t[origen][destino]
        
        apertura = Tw[destino][0]
        cierre = Tw[destino][1]
        
        ## T[i+1] = max(apertura, llegada_estimada)
        if apertura == -float('inf'):
            T[i+1] = llegada_estimada
        else:
            ## Hay que esperar
            T[i+1] = max(apertura, llegada_estimada)
            
        g = T[i+1] - cierre

        penalizacion_total += (max(0, g)** 2)
            
    ## El tiempo total final será el último valor registrado en nuestra lista T
    tiempo_total = T[-1]
    
    VFO = tiempo_total + (lambda_penal * penalizacion_total)
    
    return VFO


def remocion_abruptos_insercion(ruta, t, Tw, m=3, lambda_penal=10):

    hijo = ruta.copy()
    
    ## Identificamos cuántas ciudades cliente hay (excluyendo el depósito 0)
    num_ciudades_clientes = len(hijo) - 1 
    
    ## Recorre cada ciudad cliente (del 1 al n)
    for i in range(1, num_ciudades_clientes + 1):
        
        ## 1. SELECCIÓN ENTRE LAS CIUDADES MÁS CERCANAS
        ## Ordenamos los índices de la matriz de tiempo según su cercanía a la ciudad i
        idx_cercanos = np.argsort(t[i])
        
        # En Python d(i,i) es 0 y queda en la posición 0 del argsort. 
        # Tomamos de la posición 1 a la m+1 para obtener los m vecinos más cercanos.
        vecinos_candidatos = idx_cercanos[1:m+1]
        
        # Elegimos un vecino al azar de esos m candidatos
        vecino_elegido = random.choice(vecinos_candidatos)
        
        # 2. POSICIÓN DE INSERCIÓN
        # Buscamos en qué posición (índice) de la ruta actual está el vecino elegido
        idx_vecino = hijo.index(vecino_elegido)
        
        # Las dos opciones de inserción: antes del vecino (idx_vecino) o después (idx_vecino + 1)
        posiciones_insercion = [idx_vecino, idx_vecino + 1]
        
        # 3. ELIMINAR CIUDAD DE SU POSICIÓN ACTUAL
        ruta_temporal = hijo.copy()
        pos_remove = ruta_temporal.index(i)
        ruta_temporal.pop(pos_remove)  # Removemos la ciudad 'i' de la ruta
        
        # Ajustar las posiciones de inserción según la eliminación
        # Si la ciudad que borramos estaba antes que el vecino, las posiciones se recorren -1
        posiciones_ajustadas = []
        for pos in posiciones_insercion:
            if pos > pos_remove:
                posiciones_ajustadas.append(pos - 1)
            else:
                posiciones_ajustadas.append(pos)
                
        # 4. INSERTAR EL ELEMENTO EN LAS NUEVAS POSICIONES (Concatenación)
        # Opción 1: Insertar antes del vecino
        p1 = posiciones_ajustadas[0]
        ruta1 = ruta_temporal[:p1] + [i] + ruta_temporal[p1:]
        
        # Opción 2: Insertar después del vecino
        p2 = posiciones_ajustadas[1]
        ruta2 = ruta_temporal[:p2] + [i] + ruta_temporal[p2:]
        
        # 5. SELECCIONAR LA MEJOR RUTA (EL TORNEO)
        fo_hijo = evaluar_aptitud_tsptw(hijo, t, Tw, lambda_penal)
        fo_ruta1 = evaluar_aptitud_tsptw(ruta1, t, Tw, lambda_penal)
        fo_ruta2 = evaluar_aptitud_tsptw(ruta2, t, Tw, lambda_penal)
        
        # Al igual que en MATLAB, comparamos las 3 opciones y nos quedamos con la menor
        candidatos = [hijo, ruta1, ruta2]
        fo_valores = [fo_hijo, fo_ruta1, fo_ruta2]
        
        idx_mejor = np.argmin(fo_valores)
        hijo = candidatos[idx_mejor].copy()
        
    return hijo

##--ALGORITMO GENÉTICO HÍBRIDO--##

# Paso 1: Generar la población inicial (50 individuos con 10 ciudades cliente)
poblacion_inicial = generar_poblacion_inicial(50, 10)

# Paso 1.2 y Paso 2: Evaluar y Aplicar Remoción de Abruptos a toda la población
poblacion_paso2 = []
vfo_iniciales = []
vfo_optimizados = []

for ind in poblacion_inicial:
    # Evaluamos la aptitud original del individuo (Paso 1.2)
    vfo_orig = evaluar_aptitud_tsptw(ind, t, Tw)
    vfo_iniciales.append(vfo_orig)
    
    # Aplicamos la heurística local de inserción vecinal (Paso 2)
    ind_optimizado = remocion_abruptos_insercion(ind, t, Tw, m=3)
    poblacion_paso2.append(ind_optimizado)
    
    # Evaluamos el VFO de la ruta ya optimizada
    vfo_opt = evaluar_aptitud_tsptw(ind_optimizado, t, Tw)
    vfo_optimizados.append(vfo_opt)

# Mostramos una muestra pequeña de los primeros 5 resultados para validar
print("\nMuestra de resultados (Primeros 5 individuos de la población):")
for i in range(5):
    print(f"Individuo {i+1}:")
    print(f"  Ruta Inicial:    {poblacion_inicial[i]} | VFO: {vfo_iniciales[i]:.2f}")
    print(f"  Ruta Optimizada: {poblacion_paso2[i]} | VFO: {vfo_optimizados[i]:.2f}")