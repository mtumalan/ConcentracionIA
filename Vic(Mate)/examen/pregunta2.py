import numpy as np

a = 5
b = 3
c = 2

u = b * c                # u = b * c
v = 2 * a + u**2         # v = 2a + u^2
J = 3 * v**2             # J = 3v^2

dJ_dv = 6 * v
dv_du = 2 * u
du_db = c

# Regla de la cadena para obtener dJ/db
dJ_db = dJ_dv * dv_du * du_db

# Resultados
print("Derivada de J con respecto a b (dJ/db):", dJ_db)