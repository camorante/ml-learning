import time
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import euclidean_distances
from distancias import euclidea_rapida, matriz_distancias

# Posiciones GPS en coordenadas UTM (metros, zona 30T): dos lecturas separadas «sep» metros
for sep in (10.0, 1.0, 0.1, 0.01):
    X = np.array([[440_250.0, 4_474_830.0], [440_250.0 + 0.6 * sep, 4_474_830.0 + 0.8 * sep]])
    print(f"{sep:5}", f"{cdist(X, X)[0, 1]:.6f}", f"{euclidea_rapida(X)[0, 1]:.6f}",
          f"{euclidean_distances(X)[0, 1]:.6f}", f"{euclidea_rapida(X - X.mean(0))[0, 1]:.6f}")
#   sep   resta directa   truco   scikit-learn   truco tras centrar
print(np.spacing(np.float32(4_474_830.0)))          # resolución de float32 a 4,47 millones

rng = np.random.default_rng(0); A = rng.normal(size=(5000, 20))
for nombre, f in (("difusión", lambda: matriz_distancias(A, A)), ("truco", lambda: euclidea_rapida(A, A)), ("cdist", lambda: cdist(A, A))):
    t = time.perf_counter(); f(); print(nombre, round(time.perf_counter() - t, 2), "s")
