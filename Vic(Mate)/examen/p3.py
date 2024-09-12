import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

X = np.array([[2], [3], [5], [4], [3], [6], [1]])  # Valores de x
y = np.array([4, 6, 8, 4, 2, 3, 1])  # Valores de y

# Crear características polinomiales de grado 3
poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X)

# Ajustar el modelo de regresión lineal
model = LinearRegression()
model.fit(X_poly, y)

# Obtener los coeficientes del modelo
coefficients = model.coef_
intercept = model.intercept_

# Imprimir el valor de theta_2 (coeficiente del término cuadrático)
print("Coeficientes del modelo:", coefficients)
print("Intercepto:", intercept)
print("Valor de theta_2 (coeficiente cuadrático):", coefficients[2])
