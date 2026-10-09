import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from distancias import estandarizar, vecinos, knn_predecir
np.set_printoptions(suppress=True)

def flota(n=1200, seed=0):
    """Camiones simulados: km/día, paradas/día, horas/día, carga media (kg) y tipo de servicio."""
    r = np.random.default_rng(seed)
    tipos = ["urbano", "lanzadera", "regional", "larga"]
    #           (media, desviación) de km, paradas y horas
    par = [((220, 60), (13, 2.0), (8.0, 1.0)), ((240, 60), (4, 1.5), (6.0, 1.0)),
           ((380, 70), (7, 1.5), (8.5, 1.0)), ((650, 80), (2, 1.0), (10.0, 1.0))]
    y = r.integers(0, 4, n); X = np.zeros((n, 4))
    for c, ((km, sk), (pa, sp), (h, sh)) in enumerate(par):
        i = y == c
        X[i, 0] = r.normal(km, sk, i.sum())
        X[i, 1] = np.clip(r.normal(pa, sp, i.sum()), 0, None)
        X[i, 2] = r.normal(h, sh, i.sum())
    X[:, 3] = r.uniform(2000, 18000, n)            # carga: no depende del tipo de servicio
    return X, y, tipos

X, y, tipos = flota()
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
print(np.round(Xtr.std(0), 1))
Ztr, media, desv = estandarizar(Xtr)               # media y desviación SOLO del entrenamiento
Zte, _, _ = estandarizar(Xte, media, desv)

for cols, nombre in (([0, 1, 2, 3], "km, paradas, horas, carga"), ([0, 1, 2], "km, paradas, horas")):
    a = (knn_predecir(Xtr[:, cols], ytr, Xte[:, cols], k=5) == yte).mean()
    b = (knn_predecir(Ztr[:, cols], ytr, Zte[:, cols], k=5) == yte).mean()
    print(f"{nombre:26s} sin escalar {a:.3f}   escalado {b:.3f}")

pipe = make_pipeline(StandardScaler(), KNeighborsClassifier(5)).fit(Xtr, ytr)
print("sklearn (StandardScaler + KNN):", round(pipe.score(Xte, yte), 3))

# El vecino más cercano de un camión de la prueba (una lanzadera)
i = 6
for nombre, A, B in (("sin escalar", Xtr, Xte), ("escalado", Ztr, Zte)):
    j, d = vecinos(B[i:i + 1], A, k=1)
    j = int(j[0, 0])
    print(f"{nombre:11s}", tipos[yte[i]], np.round(Xte[i], 1), "->", np.round(Xtr[j], 1), tipos[ytr[j]])
