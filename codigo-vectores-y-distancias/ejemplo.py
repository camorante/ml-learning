import numpy as np
from toy import T                                   # los 12 camiones de la versión sencilla
from distancias import minkowski, estandarizar

K = T[:, [2, 1]]                                    # km/día y paradas/día
Z, media, desv = estandarizar(K)                    # media y desviación de los 12 camiones
print(np.round(media, 3), np.round(desv, 3))        # [368.333 7.708] [200.076 4.25]
ref, otros = 1, [0, 7, 3]                           # camión 2 frente a los camiones 1, 8 y 4
for nombre, X in (("km y paradas", K), ("escalados", Z)):
    for j in otros:
        print(nombre, j + 1, np.round(X[j] - X[ref], 3),
              [round(float(minkowski(X[ref], X[j], p)), 3) for p in (1, 2, np.inf)])
