import numpy as np
from sklearn.linear_model import LogisticRegression
from datos import carga_consumo, XOR
from red import Densa

X, y = carga_consumo()
oculta = Densa([[-1.1, 2.0], [1.1, -2.0]], [-53.0, 43.0])       # la capa oculta de la red a mano
_, H = oculta(X)

# una neurona de salida (regresión logística casi sin regularizar) sobre x o sobre h
for nombre, E in (("entradas x (carga, consumo)", X), ("espacio oculto h = (h_A, h_B)", H)):
    m = LogisticRegression(C=1e6, max_iter=10000).fit(E, y)
    print(f"logística sobre {nombre:30s}: errores {int(np.sum(m.predict(E) != y))} de 30")

# en el espacio oculto los grupos quedan en esquinas distintas
for t, s in ((0, "normales"), (1, "raros")):
    print(f"{s:9s}: h_A en [{H[y == t, 0].min():.3f}, {H[y == t, 0].max():.3f}]"
          f"  h_B en [{H[y == t, 1].min():.3f}, {H[y == t, 1].max():.3f}]"
          f"  h_A + h_B en [{H[y == t].sum(1).min():.3f}, {H[y == t].sum(1).max():.3f}]")

# XOR: los dos «sí» caen en el MISMO punto del espacio oculto
_, Hx = Densa([[10, 10], [10, 10]], [-5, -15])(XOR[0])
for x, h in zip(XOR[0].astype(int), Hx):
    print("XOR", x, "→ h =", h.round(4), "  h_A − h_B − 0.5 =", round(h[0] - h[1] - 0.5, 4))
