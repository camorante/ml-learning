import numpy as np
from producto import matriz_coseno, top_k, pearson, coseno

rng = np.random.default_rng(7)
# horas/semana en: ciudad, carretera, autopista, ralentí
patrones = {"reparto urbano": [0.45, 0.25, 0.15, 0.15],
            "regional":       [0.25, 0.40, 0.25, 0.10],
            "larga distancia":[0.10, 0.30, 0.52, 0.08]}
tipos, X = [], []
for k, (nombre, p) in enumerate(patrones.items()):
    for _ in range(30):
        horas = rng.uniform(5, 70)                        # cuánto trabaja ese camión
        mezcla = np.clip(np.array(p) + rng.normal(0, 0.07, 4), 0.01, None)
        X.append(horas * mezcla / mezcla.sum()); tipos.append(k)
X, tipos = np.array(X), np.array(tipos)

D = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1)); np.fill_diagonal(D, np.inf)
S = matriz_coseno(X); np.fill_diagonal(S, -np.inf)
v_eu, v_cos = D.argmin(1), S.argmax(1)
print("vecino del mismo tipo: euclídea", (tipos[v_eu] == tipos).mean().round(3), " coseno", (tipos[v_cos] == tipos).mean().round(3))
# un camión en el que la euclídea se equivoca de tipo y el coseno no
i = int(np.flatnonzero((tipos[v_eu] != tipos) & (tipos[v_cos] == tipos))[0])
nom = list(patrones)
print(i, nom[tipos[i]], X[i].round(1))
print("  euclídea ->", v_eu[i], nom[tipos[v_eu[i]]], X[v_eu[i]].round(1), round(D[i, v_eu[i]], 1))
print("  coseno   ->", v_cos[i], nom[tipos[v_cos[i]]], X[v_cos[i]].round(1), round(S[i, v_cos[i]], 3))
tot = X.sum(1)
print("diferencia media de horas totales con el vecino: euclídea", np.abs(tot - tot[v_eu]).mean().round(1), " coseno", np.abs(tot - tot[v_cos]).mean().round(1))
# k-NN con k=5
idx_eu = np.argsort(D, 1)[:, :5]; idx_c, _ = top_k(X, X, 6); idx_c = idx_c[:, 1:]
print("5 vecinos del mismo tipo: euclídea", (tipos[idx_eu] == tipos[:, None]).mean().round(3), " coseno", (tipos[idx_c] == tipos[:, None]).mean().round(3))
# Pearson = coseno centrado, y diferencia con coseno sin centrar
a, b = X[0], X[1]
print("cos", round(coseno(a, b), 4), "pearson", round(pearson(a, b), 4), "np.corrcoef", round(np.corrcoef(a, b)[0, 1], 4))
