import numpy as np
from datos import carga_consumo
from red import Densa, MLP

X, y = carga_consumo()                               # (30, 2): carga en t y consumo en L/100 km
W1 = np.array([[-1.1, 2.0],                          # neurona A: «gasta de más»
               [1.1, -2.0]])                         # neurona B: «gasta de menos»
b1 = np.array([-53.0, 43.0])
w2, b2 = np.array([[10.0, 10.0]]), np.array([-5.0])  # salida: basta un voto

# un viaje: vector (2,) → oculta (2,) → salida (1,)
x = X[4]
z1 = W1 @ x + b1; h = 1 / (1 + np.exp(-z1)); z2 = w2 @ h + b2; p = 1 / (1 + np.exp(-z2))
print("x =", x, " z1 = W1 x + b1 =", z1.round(3), " h =", h.round(4), " p =", p.round(4))

# todo el lote a la vez: filas = viajes
Z1 = X @ W1.T + b1                                   # (30, 2) @ (2, 2) + (2,) → (30, 2)
H = 1 / (1 + np.exp(-Z1))
P = 1 / (1 + np.exp(-(H @ w2.T + b2)))               # (30, 2) @ (2, 1) + (1,) → (30, 1)
print("formas:", X.shape, "→", Z1.shape, "→", P.shape, "  fila 5 del lote == viaje 5:", np.allclose(P[4], p))
print("errores:", int(np.sum((P[:, 0] > 0.5) != y)), "de", len(y))

# lo mismo con la clase del módulo, y el recuento de parámetros
red = MLP([Densa(W1, b1), Densa(w2, b2)])
print("MLP:", red.tamanos, "parámetros:", red.n_param, " max |diferencia| =", np.max(np.abs(red.forward(X) - P)))
