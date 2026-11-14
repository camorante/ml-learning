import numpy as np
from neurona import sigmoide

# 3 jornadas × 3 entradas: frenadas/100 km, horas de ralentí, % del tiempo por encima del límite
X = np.array([[6.8, 1.9, 4.0], [1.2, 0.8, 0.5], [4.1, 2.2, 9.0]])
# 2 neuronas (filas de W): «revisar conducción» y «revisar velocidad»
W = np.array([[0.83, 1.28, 0.00],
              [0.10, 0.00, 0.60]])
b = np.array([-6.0, -4.0])

Z = X @ W.T + b                      # (3×3)·(3×2) + b: todas las neuronas y jornadas a la vez
bucle = np.array([[sum(W[k, j] * X[i, j] for j in range(3)) + b[k] for k in range(2)] for i in range(3)])
print("Z =\n", Z.round(3))
print("diferencia con el bucle neurona a neurona:", np.max(np.abs(Z - bucle)))
print("salidas σ(Z) =\n", sigmoide(Z).round(3))
