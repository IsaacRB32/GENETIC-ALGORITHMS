## Práctica 2: Problema 1.
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd # Para la tabla final, es más limpio

def genPoblacionReal(Np, Num_var, li, ls):

    ##Aquí vivirá toda la población
    poblacion = []

    for i in range(Np):
        ##Estamos en el individuo actual 

        ##Lista temporal para guardar (x1,x2,...)
        individuo = []

        for j in range(Num_var):
            ##Variable actual
            
            ##Genera un random real entre los límites
            x = random.uniform(li[j], ls[j])

            ##Gurda el valor de la variable actual
            individuo.append(x)

        ##Guarda el individuo actual en la población
        poblacion.append(individuo)

    return poblacion

def cruzamientoSBX(padres, Pc, li, ls, Nc=20):

    ##Calculamos cuantos padres hay
    Np = len(padres)
    ##Cuántas variables tiene los padres
    Num_var = len(padres[0])

    ##Se guardarán los hijos
    hijos = []

    ##Bucle para seleccionar parejas, avanza de 2 en 2
    for i in range(0, Np-1, 2):

        ##Sacamos un random y si es menor o igual a Pc se cruza
        if random.random() <= Pc:
            ##Se genera el valor random de U (0-1)
            U = random.random()

            hijo1 = []
            hijo2 = []

            for j in range(Num_var):
                ##Extraemos el valor de la varible actual de ambos padres
                P1 = padres[i][j]
                P2 = padres[i+1][j]

                ##Se agregó esta "protección" para evitar divir entre cero
                if abs(P2 - P1) < 1e-14:
                    hijo1.append(P1)
                    hijo2.append(P2)
                    continue
                ##Cálculo de beta
                beta = 1 + (2/(P2-P1))*min((P1-li[j]),(ls[j]-P2))
                ##Cálculo de alpha
                alpha = 2 - abs(beta)**(-(Nc+1))

                ## Cálculo del factor de beta_c basado en la distribución polinomial
                if U <= 1/alpha:
                    beta_c = (U*alpha)**(1/(Nc+1))
                else:
                    beta_c = (1/(2-U*alpha))**(1/(Nc+1))

                h1 = 0.5*((P1+P2) - beta_c*abs(P2-P1))
                h2 = 0.5*((P1+P2) + beta_c*abs(P2-P1))

                ##Se mete cada valor en el hijo correspondiente
                hijo1.append(h1)
                hijo2.append(h2)
        ## Si no simplemente se clonan
        else:

            hijo1 = padres[i].copy()
            hijo2 = padres[i+1].copy()
        ##Se meten los hijos a la lista de hijos
        hijos.append(hijo1)
        hijos.append(hijo2)

    return hijos

def mutacionPolinomial(Hijos, Pm, li, ls, Nm=70):
    ##Cantidad de hijos
    Np = len(Hijos)
    #Numero de variables por hijo
    Num_var = len(Hijos[0])

    for i in range(Np):
        ##Hijo actual
        for j in range(Num_var):
            ##Variable actual

            ##Si el random es menor o igual que Pm se muta
            if random.random() <= Pm:
                ##Numero random (0-1)
                r = random.random()

                ##Se calcula delta
                delta = min((ls[j] - Hijos[i][j]), (Hijos[i][j] - li[j])) / (ls[j] - li[j])

                ##Dependiendo del valor de r, calculamos delta_q que dictará cuánto se moverá el valor
                if r <= 0.5:

                    deltaq = (2*r + (1 - 2*r)*(1 - delta)**(Nm+1))**(1/(Nm+1)) - 1

                else:

                    deltaq = 1 - (2*(1 - r) + 2*(r - 0.5)*(1 - delta)**(Nm+1))**(1/(Nm+1))
                
                ##Se aplica la mutación
                Hijos[i][j] = Hijos[i][j] + deltaq*(ls[j] - li[j])

    return Hijos

def evaluar(poblacion):
    ## Lista para guardar cada valor de la función penalizada Fp
    valores = []
    
    ## Parámetro lambda de penalización
    lambda_p = 100000
    
    for individuo in poblacion:
        ## Evaluación de [x,y] en f(x,y)
        f_val = funcion_objetivo(individuo)
        
        ## Función de penalización P(x,y)
        P_val = funcion_penalizacion(individuo)
        
        ## Función penalizada Fp(x,y) = f(x,y) + lambda_p * P(x,y)
        Fp = f_val + (lambda_p * P_val)
        
        valores.append(Fp)
        
    return valores

def seleccionPadresTorneo(poblacion, aptitud):
    ##Tamaño de la población
    Np = len(poblacion)
    ##Lista donde se guardan los ganadores
    padres = []

    ##Permuta las posiciones de cada población
    perm1 = np.random.permutation(Np)
    perm2 = np.random.permutation(Np)

    for i in range(Np):
        ##Torneo actual
        c1 = perm1[i]
        c2 = perm2[i]
        
        ##Queremos al más pequeño (min)
        if aptitud[c1] < aptitud[c2]:
            padres.append(poblacion[c1].copy())
        else:
            padres.append(poblacion[c2].copy())

    return padres

def sustitucionElitismo(poblacion, hijos, aptitud):

    ## Sacamos el índice del individio con la aptitud más pequeña
    mejor_idx = np.argmin(aptitud)

    ##Sacamos al individuo correspiende a la aptitud más pequeña
    mejor_individuo = poblacion[mejor_idx]

    ## Sale el primer hijo y se rempalza por el ganador 
    hijos[0] = mejor_individuo.copy()

    return hijos


def funcion_objetivo(x):
    ## Rendimientos: BB(20%), LOP(42%), ILI(100%), HEAL(50%), QUI(46%), AUA(30%)
    Z = 0.20*x[0] + 0.42*x[1] + 1.00*x[2] + 0.50*x[3] + 0.46*x[4] + 0.30*x[5]
    ## Se retorna negativo porque se busca minimizar
    return -Z


def funcion_penalizacion(x):
    Rd = np.array([]) 
    Ri = np.array([sum(x) - 1.0])
    
    suma_Rd = np.sum(np.maximum(Rd, 0)**2)
    suma_Ri = np.sum(Ri**2)
    
    P = suma_Rd + suma_Ri
    
    return P

li = [0.0] * 6  ## Límite inferior de 0% para las 6 acciones
ls = [0.4] * 6  ## Límite superior de 40% (0.4) para las 6 acciones
Num_var = 6     ## Son 6 acciones (BB, LOP, ILI, HEAL, QUI, AUA)

##INPUTS 
tamanio_poblacion = 300
total_generaciones = 200
prob_cruzamiento = 0.4
prob_mutacion = 0.3

ejecuciones = 10
tabla_metricas = []
mejor_solucion_global = None
mejor_apt_global = float('inf')

## Datos para el análisis de riesgo y retorno (Tablas de Lidia)
rendimientos = np.array([0.20, 0.42, 1.00, 0.50, 0.46, 0.30])
S = np.array([
    [ 0.032,  0.005,  0.030, -0.031, -0.027,  0.010],
    [ 0.005,  0.100,  0.085, -0.070, -0.050,  0.020],
    [ 0.030,  0.085,  0.333, -0.110, -0.020,  0.042],
    [-0.031, -0.070, -0.110,  0.125,  0.050, -0.060],
    [-0.027, -0.050, -0.020,  0.050,  0.065, -0.020],
    [ 0.010,  0.020,  0.042, -0.060, -0.020,  0.080]
])

print(f"Iniciando {ejecuciones} ejecuciones para el Problema 1...")

for e in range(1, ejecuciones + 1):
    
    ## 1. Generación de población inicial
    poblacion = genPoblacionReal(tamanio_poblacion, Num_var, li, ls)

    ## 2. Evaluación inicial en la FO
    aptitud = evaluar(poblacion)

    for gen in range(1, total_generaciones + 1):
        
        # 3 Selección
        padres = seleccionPadresTorneo(poblacion, aptitud)

        # 4 Cruzamiento
        hijos = cruzamientoSBX(padres, prob_cruzamiento, li, ls)

        # 5 Mutación
        hijos = mutacionPolinomial(hijos, prob_mutacion, li, ls)

        # 6 Evaluación de descendientes
        aptitud_hijos = evaluar(hijos)

        # 7 Sustitución
        poblacion = sustitucionElitismo(poblacion, hijos, aptitud)

        # actualizar aptitud de la nueva población
        aptitud = evaluar(poblacion)

    ## CÁLCULO DE MÉTRICAS DE LA EJECUCIÓN ACTUAL
    mejor_apt = np.min(aptitud)
    peor_apt = np.max(aptitud)
    promedio_apt = np.mean(aptitud)
    desviacion_apt = np.std(aptitud)
    
    idx_mejor = np.argmin(aptitud)
    mejor_ind_e = poblacion[idx_mejor]
    
    ## Guardamos los resultados de esta corrida
    tabla_metricas.append([e, mejor_apt, promedio_apt, peor_apt, desviacion_apt])
    
    ## Guardamos al mejor de los mejores (Global)
    if mejor_apt < mejor_apt_global:
        mejor_apt_global = mejor_apt
        mejor_solucion_global = mejor_ind_e.copy()
        
    print(f" > Ejecución {e} terminada. Mejor aptitud: {mejor_apt:.4f}")

## Dibujamos una tabla con el reporte estadístico
print("\n" + "="*80)
print(f"{'Ejecución':<12} | {'Mejor Aptitud':<15} | {'Promedio':<12} | {'Peor':<12} | {'Desv. Est.':<12}")
print("-" * 80)
for fila in tabla_metricas:
    print(f"{fila[0]:<12} | {fila[1]:<15.6f} | {fila[2]:<12.6f} | {fila[3]:<12.6f} | {fila[4]:<12.6f}")
print("="*80)

## ARCHIVO DE LA SOLUCIÓN ÓPTIMA GLOBAL
pesos = np.array(mejor_solucion_global)
retorno_total = np.dot(pesos, rendimientos)
## Se hace la traspuesta para que sea compatible en la mul. de matrices (1x6)(6x6)(6x1)
riesgo_total = np.dot(pesos.T, np.dot(S, pesos))

print("\n*** SOLUCIÓN ÓPTIMA OBTENIDA (MEJOR DE TODAS LAS CORRIDAS) ***")
acciones = ["BB", "LOP", "ILI", "HEAL", "QUI", "AUA"]
for i in range(len(acciones)):
    print(f"Peso {acciones[i]}: {pesos[i]*100:.2f}%")

print(f"\nRETORNO TOTAL DE LA CARTERA: {retorno_total*100:.2f}%")
print(f"RIESGO ASOCIADO (VARIANZA): {riesgo_total:.6f}")
print("*"*60)