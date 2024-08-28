from Proyecto2 import *

# Cargar datos
data = np.loadtxt('Vic(Mate)/regresionLogistica/ex2data1.txt', delimiter=',')
X = data[:, 0:2]  # Asignar primeras dos columnas a X
y = data[:, 2]    # Asignar tercera columna a y
m = len(y)        # Número de ejemplos de entrenamiento

# Normalización de datos (opcional pero recomendado)
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# Añadir columna de unos a X para el término de sesgo
X = np.hstack((np.ones((X.shape[0], 1)), X))

# Inicializar theta
theta_inicial = np.zeros(X.shape[1]) # Inicializar theta con ceros [0, 0, 0]

# Parámetros
iteraciones = 1500
alpha = 0.01  # Ajusta el valor de alpha aquí

# Test de la función de regresión logística
def main():
    # Entrenar el modelo
    theta_final = aprende(theta_inicial, X, y, iteraciones)
    
    # Verificar el costo final
    costo_final, _ = funcionCosto(theta_final, X, y)
    print(f'Costo final: {costo_final:.3f}')
    print(f'Theta final: {theta_final}')
    
    # Predicción
    predicciones = predice(theta_final, X)
    
    # Precisión
    precision = np.mean(predicciones == y) * 100
    print(f'Precisión: {precision:.2f}%')

    # Verificar predicción para un estudiante específico
    estudiante = np.array([1, 45, 85])  # Incluye el término de sesgo (1)
    probabilidad = sigmoidal(estudiante @ theta_final)
    prediccion = predice(theta_final, estudiante.reshape(1, -1))
    print(f'Probabilidad de admisión para el estudiante [45, 85]: {probabilidad:.3f}')
    print(f'Predicción para el estudiante [45, 85]: {prediccion[0]}')

    # Graficar datos y la línea de decisión
    graficaDatos(X, y, theta_final)

if __name__ == '__main__':
    main()