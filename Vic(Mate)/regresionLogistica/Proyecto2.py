# Calcular la regresion logistica para un conjunto de datos
# Autor: Mauricio Tumalan Castillo
# Matricula: A01369288

import numpy as np
import matplotlib.pyplot as plt

def graficaDatos(x, y, theta):
    '''
    Grafica los datos y la línea de decisión
    
    Parametros:
    x -- vector de entrada
    y -- vector de salida
    theta -- vector de parametros
    '''
    admitidos = (y == 1)
    no_admitidos = (y == 0)
    
    # Crear la gráfica
    plt.figure(figsize=(8, 6))
    plt.scatter(x[admitidos, 1], x[admitidos, 2], c='b', marker='x', label='Admitido')
    plt.scatter(x[no_admitidos, 1], x[no_admitidos, 2], c='r', marker='o', label='No admitido')
    
    if theta is not None:
        # Graficar la línea de decisión
        plot_x = np.array([np.min(x[:, 1]) - 2, np.max(x[:, 1]) + 2])
        plot_y = (-1 / theta[2]) * (theta[1] * plot_x + theta[0])
        plt.plot(plot_x, plot_y, label='Línea de decisión')
    
    plt.xlabel('Examen 1')
    plt.ylabel('Examen 2')
    plt.legend()

    plt.show()

def sigmoidal(z):
    '''
    Calcula la funcion sigmoidal
    
    Parametros:
    z -- valor
    
    Salida:
    g -- valor de la funcion sigmoidal
    '''
    g = 1 / (1 + np.exp(-z))
    return g

def funcionCosto(theta, x, y):
    '''
    Calcula el costo de la regresion logistica
    
    Parametros:
    theta -- vector de parametros
    x -- vector de entrada
    y -- vector de salida
    
    Salida:
    j -- costo
    grad -- gradiente
    '''
    m = len(y)
    h = sigmoidal(x @ theta)
    h = np.clip(h, 1e-10, 1 - 1e-10)  # Evitar valores extremos
    j = (1 / m) * np.sum(-y * np.log(h) - (1 - y) * np.log(1 - h)) # Función de costo
    grad = (1 / m) * x.T @ (h - y) # Gradiente
    
    return j, grad

def aprende(theta, x, y, iteraciones, alpha = 0.01):
    '''
    Aprende los parametros de la regresion logistica
    
    Parametros:
    theta -- vector de parametros
    x -- vector de entrada
    y -- vector de salida
    iteraciones -- numero de iteraciones
    
    Salida:
    theta -- vector de parametros
    '''
    for _ in range(iteraciones):
        j, grad = funcionCosto(theta, x, y)
        theta = theta - alpha * grad # Actualizar theta con funcion delta
    return theta

def predice(theta, x):
    '''
    Predice los valores de salida
    
    Parametros:
    theta -- vector de parametros
    x -- vector de entrada
    
    Salida:
    probabilidad -- vector de probabilidades
    '''
    probabilidad = sigmoidal(np.dot(x, theta))
    return (probabilidad >= 0.5).astype(int)

# def main():
#     # Cargar los datos
#     data = np.loadtxt('Vic(Mate)/regresionLogistica/ex2data1.txt', delimiter=',')
#     x = data[:, 0:2]
#     y = data[:, 2]
    
#     # Normalizar los datos
#     x = (x - np.mean(x, axis=0)) / np.std(x, axis=0)

#     # Agregar una columna de unos a x
#     x = np.hstack((np.ones((x.shape[0], 1)), x))
    
#     # Inicializar theta
#     theta = np.zeros(x.shape[1])
    
#     # Entrenar el modelo
#     thetaf = aprende(theta, x, y, 1500)
    
#     # Graficar los datos y la línea de decisión
#     graficaDatos(x, y, thetaf)

# if __name__ == '__main__':
#     main()