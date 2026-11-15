"""Una convolución 1D es una capa densa cuya matriz tiene casi todo ceros y el mismo filtro repetido en cada fila
(una matriz de Toeplitz). Y el gradiente de cada peso compartido es la suma de los gradientes de sus copias."""
import numpy as np
from conv import conv1d, conv1d_atras

w = np.array([1.0, 2.0, -1.0])                      # filtro de 3 pesos
x = np.array([0.5, -1.0, 2.0, 0.0, 1.5, -0.5, 1.0, 3.0])
L, k = len(x), len(w)
T = np.zeros((L - k + 1, L))
for i in range(L - k + 1):
    T[i, i:i + k] = w                               # fila i: el filtro, desplazado i columnas
print(T.astype(int))
y, cache = conv1d(x[None, None], w[None, None], np.zeros(1))
print("T @ x       =", T @ x)
print("conv1d(x, w) =", y[0, 0])
print(f"la matriz tiene {T.size} números, {np.count_nonzero(T)} distintos de cero y solo {k} valores distintos")
dy = np.array([1.0, -2.0, 0.5, 0.0, 1.0, 3.0])      # gradiente que llega de arriba
dT = np.outer(dy, x)                                # si cada casilla de T fuera un peso libre (capa densa)
suma = [sum(dT[i, i + j] for i in range(L - k + 1)) for j in range(k)]
print("suma de los gradientes de las copias de cada peso:", np.round(suma, 4))
print("∂W de conv1d_atras:                               ", conv1d_atras(dy[None, None], cache)[1][0, 0])
print(f"con la ventana del tema (64 lecturas, filtro de 5): matriz {60}×{64} = {60 * 64} números frente a 5 pesos")
