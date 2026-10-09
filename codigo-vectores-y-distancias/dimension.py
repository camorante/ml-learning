import numpy as np
from scipy.spatial.distance import pdist, cdist

rng = np.random.default_rng(0)
print(" d    media   desv   desv/media  cerca/lejos")
for d in (2, 10, 100, 1000, 10000):
    X = rng.random((500, d))                       # 500 puntos al azar en el cubo [0, 1]^d
    D = pdist(X)                                   # las 124 750 distancias entre pares
    Q = cdist(X[:50], X[50:])                      # 50 consultas contra los otros 450
    ratio = (Q.min(1) / Q.max(1)).mean()           # más cerca / más lejos, media de las 50
    print(f"{d:5d} {D.mean():8.3f} {D.std():6.3f} {D.std() / D.mean():9.3f} {ratio:10.3f}")
print("teoría: media ≈ sqrt(d/6), desv → sqrt(7/120) =", round(np.sqrt(7 / 120), 3))
