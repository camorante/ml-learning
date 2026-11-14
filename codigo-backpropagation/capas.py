"""La culpa que llega a cada capa: red de 10 capas ocultas de 8 neuronas sobre los 30 viajes (estandarizados),
retropropagación con retro.py. Tamaño medio |∂L/∂w| de los pesos de cada capa, media de 50 inicializaciones
(Glorot para sigmoide y tanh, He para ReLU), y la media de la pendiente φ'(z) que multiplica en cada capa."""
import json
import numpy as np
from datos import carga_consumo
from retro import Densa, MLP, ACT

X, y = carga_consumo(); X = (X - X.mean(0)) / X.std(0)
NC, M, S = 10, 8, 50
res = {}
for act in ("sigmoide", "tanh", "relu"):
    g, pend = np.zeros(NC + 1), np.zeros(NC)
    for s in range(S):
        rng = np.random.default_rng(s)
        tam = [2] + [M] * NC + [1]
        sd = lambda a, b: np.sqrt(2 / a) if act == "relu" else np.sqrt(2 / (a + b))
        red = MLP([Densa(rng.normal(0, sd(a, b), (b, a)), np.zeros(b), act if i < NC else "identidad")
                   for i, (a, b) in enumerate(zip(tam, tam[1:]))], "bce")
        _, gr = red.gradientes(X, y)
        g += np.array([np.mean(np.abs(dW)) for dW, _ in gr]) / S
        pend += np.array([np.mean(ACT[act][1](c.Z, c.H)) for c in red.capas[:NC]]) / S
    res[act] = {"g": (g / g[-1]).tolist(), "pend": pend.tolist()}
    print(f"{act:8s} culpa de cada capa / la de la salida:  " + "  ".join(f"{v:.1e}" for v in g[[0, 2, 4, 6, 8, 10]] / g[-1])
          + f"   (capas 1, 3, 5, 7, 9 y salida)")
    print(f"{'':8s} pendiente media φ'(z): {pend.min():.3f}–{pend.max():.3f}")
json.dump(res, open("capas.json", "w", encoding="utf-8"))
