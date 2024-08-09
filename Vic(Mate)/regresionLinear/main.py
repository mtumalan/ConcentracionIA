# Usado para testing

from Proyecto1 import *
import csv
import os
import numpy as np

# Datos de entrada

def fileData():
    path = os.getcwd()
    filepath = r"\Vic(Mate)\regresionLinear\ex1data1.txt"
    path2file = path + filepath

    x = []
    y = []
    with open(path2file, 'r') as f:
        csvreader = csv.reader(f)
        for row in csvreader:
            x.append(float(row[0]))
            y.append(float(row[1]))

    x = np.array(x)  # Mantener solo la columna original de valores de x
    y = np.array(y)

    return x, y

# Datos de entrada para pruebas
x, y = fileData()

# Inicializar theta
theta = np.array([0, 0], dtype=float)

# Aplicar el método de gradiente descendiente
theta = gradienteDescendiente(x, y, theta)

# Calcular el costo final
cost = calculaCosto(x, y, theta)
print(f"Costo final: {cost}")

# Graficar los datos y la línea de regresión
graficaDatos(x, y, theta)