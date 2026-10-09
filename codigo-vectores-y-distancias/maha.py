import numpy as np
from toy import T
from distancias import mahalanobis, matriz_distancias, estandarizar

X = T[:, [2, 1, 0]]                                   # km, paradas, horas de los 12 camiones
S = np.cov(X, rowvar=False, bias=True)                # covarianza (tema posterior)
print(np.round(np.corrcoef(X, rowvar=False)[0], 2))   # correlación de los km con km, paradas, horas
VI = np.linalg.inv(S)
for nombre, D in (("estandarizada", matriz_distancias(estandarizar(X)[0])), ("Mahalanobis", mahalanobis(X, X, VI))):
    np.fill_diagonal(D, np.inf)
    print(f"{nombre:13s}", (D.argmin(1) + 1).tolist())
