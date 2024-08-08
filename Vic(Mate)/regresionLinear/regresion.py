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
    plt.plot(x[:, 1], y, 'rx')
    plt.plot(x[:, 1], np.dot(x, theta))
    plt.show()

def gradienteDescendiente(x, y, theta = np.array([0, 0], dtype=float), alpha = 0.01, iteraciones = 1500):
    '''
    Calcula la regresion lineal de un conjunto de datos
    
    Parametros:
    x -- vector de entrada
    y -- vector de salida
    theta -- vector de parametros
    alpha -- tasa de aprendizaje
    iteraciones -- numero de iteraciones
    
    Salida:
    theta -- vector de parametros
    '''
    m = len(y)
    for i in range(iteraciones):
        h = np.dot(x, theta)
        theta = theta - alpha * (1/m) * np.dot(x.T, h - y)
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
    h = np.dot(x, theta)
    j = (1/(2*m)) * np.sum(np.square(h - y))
    return j