# Calcular la regresion lineal de un conjunto de datos con el metodo de gradiente descendente
# Autor: Mauricio Tumalan Castillo
# Matricula: A01369288

import numpy as np
import matplotlib.pyplot as plt

def graficaDatos(x, y, theta):
    '''
    Grafica los datos de entrada y salida
    
    Parametros:
    x -- vector de entrada
    y -- vector de salida
    theta -- vector de parametros
    
    Salida:
    Grafica de los datos
    '''
    plt.plot(x, y, 'rx')
    plt.plot(x, theta[0] + theta[1] * x)
    plt.show()

def gradienteDescendiente(x, y, theta = np.array([0, 0], dtype=float), alpha = 0.01, iteraciones = 1500):
    '''
    Calcula la regresion lineal de un conjunto de datos con el metodo de gradiente descendente
    
    Parametros:
    x -- vector de entrada
    y -- vector de salida
    theta -- vector de parametros
    alpha -- tasa de aprendizaje
    iteraciones -- numero de iteraciones
    
    Salida:
    theta -- vector de parametros
    '''
    m = len(y) # Numero de datos
    for i in range(iteraciones):
        # Calcular la hipótesis
        h = theta[0] + theta[1] * x
        
        # Actualizar theta
        theta[0] = theta[0] - alpha * (1/m) * np.sum(h - y)
        theta[1] = theta[1] - alpha * (1/m) * np.sum((h - y) * x)
    
    return theta


def calculaCosto(x, y, theta):
    '''
    Calcula el costo de la regresion lineal
    
    Parametros:
    x -- vector de entrada
    y -- vector de salida
    theta -- vector de parametros
    
    Salida:
    j -- costo
    '''
    m = len(y)
    h = theta[0] + theta[1] * x
    j = (1/(2*m)) * np.sum(np.square(h - y))
    return j