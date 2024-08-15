# Algoritmo de aprendizaje para perceptrón simple que clasifique las entradas de una tabla de verdad de OR
# Autor: Mauricio Tumalan Castillo
# Matricula: A01369288

import numpy as np

def perceptron(alpha=0.1, iteraciones=1500):
    '''
    Algoritmo de aprendizaje para perceptrón simple que devuelva los pesos de la función de activación
    '''

    # Tabla de verdad de OR
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 1])

    # Inicializar peso
    weight = np.array([1.5, 0.5, 1.5])
    print('Pesos iniciales:', weight)

    # Agregar columna de unos a x para incluir el bias
    x = np.column_stack((np.ones(len(x)), x))
    print('Datos de entrada con bias:', x)

    i = 0
    # Algoritmo de aprendizaje
    while i < iteraciones:
        total_error = 0
        for j in range(len(y)):
            # Calcular la hipótesis
            d = np.dot(x[j], weight)
            # Aplicar función de activación escalón
            output = 1 if d >= 0 else 0
            # Calcular error
            error = y[j] - output
            # Actualizar pesos
            weight += alpha * error * x[j]
            # Sumar error absoluto para saber cuándo parar
            total_error += abs(error)
        
        print('Iteración:', i, 'Error total:', total_error, 'Pesos:', weight)
        
        # Si el error total es menor o igual a un umbral pequeño, detener
        epsilon = 1e-3
        if total_error <= epsilon:
            print('Convergencia alcanzada en iteración:', i)
            break
        i += 1
    
    return weight

# Entrenamiento del perceptrón
weights = perceptron()

# convertir pesos a float
weights = [float(w) for w in weights]

print('Pesos finales:', weights)