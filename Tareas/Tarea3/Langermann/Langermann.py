## ROJAS BARRÓN ISAAC - TAREA 2 Algoritmo genético básico
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import imageio

def graficar_poblacion(poblacion, mejor, gen, li, ls):
    x = np.linspace(li[0], ls[0], 100)
    y = np.linspace(li[1], ls[1], 100)
    X, Y = np.meshgrid(x, y)

    Z = np.zeros_like(X)

    Z = np.array([
        [funcionLangermann([X[j,i], Y[j,i]]) for i in range(len(x))]
        for j in range(len(y))
    ])

    plt.figure(figsize=(6,5))
    plt.contourf(X, Y, Z, levels=50, cmap='jet')

    pop = np.array(poblacion)

    plt.scatter(pop[:,0], pop[:,1], color='white', s=10, alpha=0.6, label='Población')
    plt.scatter(mejor[0], mejor[1], color='red', s=40, label='Mejor')

    plt.title(f"Generación {gen}")
    plt.xlim(li[0], ls[0])
    plt.ylim(li[1], ls[1])

    plt.legend()

    filename = f"frame_{gen}.png"
    plt.savefig(filename)
    plt.close()

    return filename

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

##Se optó por un Nc=2 (Bajo) para favorecer la exploración
def cruzamientoSBX(padres, Pc, li, ls, Nc=2):

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

##Se optó por un Nm=20 (Bajo) para genera valores más lejanos al origen.
def mutacionPolinomial(Hijos, Pm, li, ls, Nm=20):
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
    ##Lista para guardar cada valor de cada individuo de la población.
    valores = []
    
    for individuo in poblacion:
        ##Indivio actual
        valor = funcionLangermann(individuo)
        valores.append(valor)
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


def funcionLangermann(x):

    a = [3,5,2,1,7]
    b = [5,2,1,4,9]
    c = [1,2,5,2,3]

    suma = 0

    for i in range(5):

        term = (x[0]-a[i])**2 + (x[1]-b[i])**2

        suma += c[i] * math.cos(math.pi * term) * math.exp(-term/math.pi)

    return -suma

li = [0,0]
ls = [10,10]
Num_var = 2

##INPUTS
tamanio_poblacion = 200
total_generaciones = 200
prob_cruzamiento = 0.8
prob_mutacion = 0.3

frames = []

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

    idx_mejor = np.argmin(aptitud)
    mejor_valor = aptitud[idx_mejor]
    mejor_individuo = poblacion[idx_mejor]
    ##Solo multiplos de 5
    if gen % 5 == 0 or gen == 1:
        print(f"Gen {gen}: Mejor FO = {mejor_valor} en x={mejor_individuo}")
    
    if gen % 5 == 0:
        frame = graficar_poblacion(poblacion, mejor_individuo, gen, li, ls)
        frames.append(frame)
    
imagenes = [imageio.v2.imread(f) for f in frames]
imageio.mimsave("convergencia.gif", imagenes, fps=2)
