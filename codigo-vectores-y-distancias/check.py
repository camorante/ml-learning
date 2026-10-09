import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from distancias import *

rng = np.random.default_rng(0)
rel = lambda a, b: np.abs(a - b).max() / np.abs(b).max()          # error relativo máximo
e = dict(norma=0.0, minkowski=0.0, ponderada=0.0, sklearn=0.0, rapida=0.0, mahalanobis=0.0)
ok_vec = ok_knn = 0
for t in range(200):
    n, m, d = rng.integers(5, 120), rng.integers(5, 120), rng.integers(1, 40)
    X, Y = rng.normal(size=(n, d)) * rng.uniform(0.1, 50), rng.normal(size=(m, d))
    for p in (1, 1.5, 2, 3, np.inf):
        e["norma"] = max(e["norma"], rel(norma(X, p), np.linalg.norm(X, ord=p, axis=1)))
        ref = cdist(X, Y, "chebyshev") if p == np.inf else cdist(X, Y, "minkowski", p=p)
        e["minkowski"] = max(e["minkowski"], rel(matriz_distancias(X, Y, p), ref))
    w = rng.uniform(0.1, 3, d)
    e["ponderada"] = max(e["ponderada"], rel(matriz_distancias(X, Y, 3, w), cdist(X, Y, "minkowski", p=3, w=w)))
    for met, p in (("manhattan", 1), ("euclidean", 2), ("chebyshev", np.inf)):
        e["sklearn"] = max(e["sklearn"], rel(matriz_distancias(X, Y, p), pairwise_distances(X, Y, metric=met)))
    e["rapida"] = max(e["rapida"], rel(euclidea_rapida(X, Y), cdist(X, Y)))
    A = rng.normal(size=(d, d)); VI = np.linalg.inv(A @ A.T + d * np.eye(d))
    e["mahalanobis"] = max(e["mahalanobis"], rel(mahalanobis(X, Y, VI), cdist(X, Y, "mahalanobis", VI=VI)))
    k = int(rng.integers(1, 6))
    i1, _ = vecinos(Y, X, k)
    _, i2 = NearestNeighbors(n_neighbors=k).fit(X).kneighbors(Y)
    ok_vec += np.array_equal(i1, i2)
    ytr = rng.integers(0, 3, n)
    ok_knn += np.array_equal(knn_predecir(X, ytr, Y, k), KNeighborsClassifier(n_neighbors=k).fit(X, ytr).predict(Y))
for k, v in e.items():
    print(f"{k:12s} {v:.1e}")
print("vecinos = NearestNeighbors:", ok_vec, "/ 200 · KNN = KNeighborsClassifier:", ok_knn, "/ 200")
