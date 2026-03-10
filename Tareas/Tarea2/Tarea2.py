import numpy as np
import random
import math

def longitudCadena(rango, pres):
    return int(math.log2((rango[1] - rango[0]) * (10**pres)) + 0.9)

def genPoblacion(longitudTotal, tamanioPoblacion):
    ##Aquí se guardará cada individio 
    poblacion = []

    for i in range(tamanioPoblacion):
        ##Individio actual
        individuo = []
        ##Recorremos cada bit dada la longitudTotal
        for j in range(longitudTotal):
            ##De manera aleatorea deicimos si es 0 o 1 el bit actual
            bit = random.randint(0,1)
            individuo.append(bit)
        ##Agregamos el individuo actual a la población
        poblacion.append(individuo)
    return poblacion

def binarioADecimal(bits):
    ##Acomulador 
    decimal = 0
    ##Cantidad de bits de la cadena
    L = len(bits)
    
    for i in range(L):
        ##Bit actual
        ##Al valor actual (0 o 1) lo multiplicamos por 
        ##la potencia de 2 que le corresponde a esa posición
        decimal += bits[i] * (2**(L - 1 - i))
    
    return decimal

def decodificar(poblacion, longitud, rango):
    ##Lista de la población en valor real
    poblacionReal = []
    ##Extraemos los límetes sup. e inf. del rango
    li = rango[0]
    ls = rango[1]
    
    for individuo in poblacion:
        ##Individuo actual
        
        ##Separamos al individuo actual a la mita pq es (x,y)
        bits_x1 = individuo[:longitud]
        bits_x2 = individuo[longitud:]
        
        ##Ahora, hacemos el cambio a decimal de cada mitad
        x1_dec = binarioADecimal(bits_x1)
        x2_dec = binarioADecimal(bits_x2)
        
        ##Comberción de binario a números reales
        x1_real = li + (x1_dec * (ls - li)) / (2**longitud - 1)
        x2_real = li + (x2_dec * (ls - li)) / (2**longitud - 1)
        
        ##Metemos a la nueva lista el indivio actual en versión real
        poblacionReal.append([x1_real, x2_real])
    
    return poblacionReal

def evaluar(poblacionReal):
    ##Lista para guardar cada valor de cada individuo de la población.
    valores = []
    
    for individuo in poblacionReal:
        ##Indivio actual
        valor = funcionRastrigin(individuo)
        valores.append(valor)
    return valores

def seleccionPadresTorneo(poblacion, fitness):
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
        if fitness[c1] < fitness[c2]:
            padres.append(poblacion[c1].copy())
        else:
            padres.append(poblacion[c2].copy())

    return padres

def cruzamiento(padres, Pc):
    ##Lista donde se guardan la nueva generación
    hijos = []

    Np = len(padres) ##Número de padres
    ##Tomamos la primer lista de la lista de listas que es padres
    Nbits = len(padres[0]) ##Cantidad de bits

    ##Sato de dos en dos
    for i in range(0, Np-1, 2):
        ##Padre en la posición par 
        p1 = padres[i]
        ##Padre en la posición impar
        p2 = padres[i+1]

        ## Genera un número decimal entre 0 y 1 para la probabilidad
        rand = random.random()
        ## Si el azar es menor o igual a la Pc
        if rand <= Pc:
            ## Elige el primer sitio para cortar
            pto1 = random.randint(1, Nbits-2)
            ## Elige el segundo sitio (más adelante que el primero)
            pto2 = random.randint(pto1+1, Nbits-1)

            ##Concatenación
            hijo1 = (
                p1[:pto1] +     ## Cabeza del Padre 1
                p2[pto1:pto2] + ## Cuerpo del Padre 2
                p1[pto2:]       ## Cola del Padre 1
            )

            hijo2 = (
                p2[:pto1] +     ## Cabeza del Padre 2
                p1[pto1:pto2] + ## Cuerpo del Padre 1
                p2[pto2:]       ## Cola del Padre 2
            )

        else:
            ## Los padres son los hijos
            hijo1 = p1.copy()
            hijo2 = p2.copy()

        hijos.append(hijo1)
        hijos.append(hijo2)

    return hijos

def sustitucionElitismo(poblacion, hijos, aptitud):

    ## Sacamos el índice del individio con la aptitud más pequeña
    mejor_idx = np.argmin(aptitud)

    ##Sacamos al individuo correspiende a la aptitud más pequeña
    mejor_individuo = poblacion[mejor_idx]

    ## Sale el primer hijo y se rempalza por el ganador 
    hijos[0] = mejor_individuo.copy()

    return hijos

def mutacion(poblacion, prob):
    Np = len(poblacion) ##Cantidade de individuos
    Nbit = len(poblacion[0]) ##Longitud de cada individuo

    for i in range(Np):
        ##Indivio actual
        ## Genera un número decimal entre 0 y 1 para la probabilidad
        rand = random.random()

        if rand <= prob:
            # Generar una posición aleatoria para mutar (un solo bit)
            pto = random.randint(0, Nbit - 1)
            # Cambiar el valor del bit (bit flip)
            poblacion[i][pto] = 1 - poblacion[i][pto]
            
    return poblacion


def funcionRastrigin(x):
    # x[0] es la variable "x", x[1] es la variable "y"
    term1 = x[0]**2 - 10 * math.cos(2 * math.pi * x[0])
    term2 = x[1]**2 - 10 * math.cos(2 * math.pi * x[1])
    return 20 + term1 + term2

rango = [-5.12, 5.12]
precision = 5

##INPUTS
tamanio_poblacion = 200
total_generaciones = 200
prob_cruzamiento = 0.8
prob_mutacion = 0.1

num_ejecuciones = 10
## Lista para almacenar el mejor de cada una de las ejecuciones
resultados_ejecuciones = []

##Calculamos la longitud
long_bits_variable = longitudCadena(rango, precision)

##Como son dos variables la total es * 2
longitud_total = long_bits_variable * 2 

## 10 EJECUCIONES
for ejec in range(1, num_ejecuciones + 1):
    print(f"\nEjecución {ejec}")
    ## 1. Generación de población inicial
    poblacion = genPoblacion(longitud_total, tamanio_poblacion)

    ## 2. Decodificación inicial
    poblacion_real = decodificar(poblacion, long_bits_variable, rango)

    ## 3. Evaluación inicial en la FO
    aptitud = evaluar(poblacion_real)

    for gen in range(1, total_generaciones + 1):
        
        ## 4. Selección de padres
        padres = seleccionPadresTorneo(poblacion, aptitud)
        
        ## 5. Cruzamiento (Dos puntos)
        hijos = cruzamiento(padres, Pc=prob_cruzamiento)
        
        ## 6. Mutación
        hijos = mutacion(hijos, prob=prob_mutacion)
        
        ## 7. Sustitución
        poblacion = sustitucionElitismo(poblacion, hijos, aptitud)
        
        ## 8. Decodificación de la nueva población
        poblacion_real = decodificar(poblacion, long_bits_variable, rango)
        
        ## 9. Evaluación de la nueva población
        aptitud = evaluar(poblacion_real)
        
        ## 10. Selección del mejor individuo de esta generación
        idx_mejor = np.argmin(aptitud)
        mejor_valor = aptitud[idx_mejor]
        mejor_individuo = poblacion_real[idx_mejor]

        ##Solo multiplos de 10
        if gen % 10 == 0 or gen == 1:
            print(f"Gen {gen}: Mejor FO = {mejor_valor} en x={mejor_individuo}")

    resultados_ejecuciones.append(mejor_valor)

print("\n**** REPORTE ESTADÍSTICO FINAL ****\n")

print(f"Mejor Solución Global: {np.min(resultados_ejecuciones):.10f}")
print(f"Peor Solución:         {np.max(resultados_ejecuciones):.10f}")
print(f"Mediana:                {np.median(resultados_ejecuciones):.10f}")
print(f"Desviación Estándar:    {np.std(resultados_ejecuciones):.10f}")