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
    plt.plot(x, theta[0] + theta[1] * x + theta[2] * x**2 + theta[3] * x**3)
    plt.show()

def gradienteDescendiente(x, y, theta = np.array([0, 0, 0, 0], dtype=float), alpha = 0.0001, iteraciones = 1500):
    '''
    Calcula la regresion polinomial de grado 3 de un conjunto de datos con el metodo de gradiente descendente
    
    Parametros:
    x -- vector de entrada
    y -- vector de salida
    theta -- vector de parametros [theta0, theta1, theta2, theta3]
    alpha -- tasa de aprendizaje
    iteraciones -- numero de iteraciones
    
    Salida:
    theta -- vector de parametros actualizados
    '''
    m = len(y)  # Numero de datos
    for i in range(iteraciones):
        # Calcular la hipótesis para polinomio de grado 3
        h = theta[0] + theta[1] * x + theta[2] * x**2 + theta[3] * x**3
        
        # Actualizar theta
        theta[0] = theta[0] - alpha * (1/m) * np.sum(h - y)
        theta[1] = theta[1] - alpha * (1/m) * np.sum((h - y) * x)
        theta[2] = theta[2] - alpha * (1/m) * np.sum((h - y) * x**2)
        theta[3] = theta[3] - alpha * (1/m) * np.sum((h - y) * x**3)
    
    return theta

def calculaCosto(x, y, theta):
    '''
    Calcula el costo de la regresion polinomial de grado 3
    
    Parametros:
    x -- vector de entrada
    y -- vector de salida
    theta -- vector de parametros
    
    Salida:
    j -- costo
    '''
    m = len(y)
    h = theta[0] + theta[1] * x + theta[2] * x**2 + theta[3] * x**3
    j = (1/(2*m)) * np.sum(np.square(h - y))
    return j

# Escalar los datos de entrada (opcional pero recomendado)
x = np.array([2, 3, 5, 4, 3, 6, 1])
y = np.array([4, 6, 8, 4, 2, 3, 1])

# Parámetros iniciales para theta
theta_inicial = np.array([0, 0, 0, 0], dtype=float)

# Aplicar gradiente descendente
alpha = 0.0001  # Reducir la tasa de aprendizaje
iteraciones = 10000  # Número de iteraciones

theta_final = gradienteDescendiente(x, y, theta_inicial, alpha, iteraciones)

# Imprimir los parámetros finales
print("Parámetros finales:", theta_final)
print("Valor de theta_2:", theta_final[2])