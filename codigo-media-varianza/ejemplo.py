import numpy as np
from resumen import media, mediana, varianza, desviacion, cv, Welford

x = np.array([32, 35, 30, 33, 37, 31, 34, 32], float)          # L/100 km de 8 viajes
d = x - media(x); print(media(x), d, d.sum(), d**2, (d**2).sum())
print(varianza(x, 0), round(varianza(x, 1), 4), round(desviacion(x, 0), 4), round(desviacion(x, 1), 4))
print([36 + 8 * (33 - c) ** 2 for c in (32, 33, 34)], ((x - 32) ** 2).sum())   # Σ(x − c)² = 36 + 8(33 − c)²
print(mediana(x), round(cv(x) * 100, 2))
y = x.copy(); y[4] = 370                                         # el 37 guardado como 370
print(media(y), mediana(y), round(desviacion(y), 2), round((370 - media(y)) / desviacion(y), 4), round(7 / np.sqrt(8), 4))
r = np.array([12, 25, 8, 15, 40, 10, 18, 16], float)            # minutos de ralentí de los mismos viajes
print(media(r), (r - media(r)).tolist(), ((r - media(r)) ** 2).sum(), round(desviacion(r), 3), round(cv(r) * 100, 1))
print(round((37 - 33) / desviacion(x), 3), round((40 - 18) / desviacion(r), 3))
w = Welford()
for v in x:
    w.add(v); print(w.n, round(w.m, 4), round(w.M2, 4))
