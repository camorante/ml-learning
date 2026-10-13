"""Idea 5: rellenar huecos de una tabla camión × ruta (consumo, L/100 km) con una sola capa."""
import numpy as np

RUTAS = ["urbana", "polígono", "regional", "nacional", "autovía", "montaña"]


def tabla(seed=4):
    rng = np.random.default_rng(seed)
    base = np.array([34.0, 31.0, 29.0, 28.0, 26.0, 38.0])
    fac = np.round(rng.uniform(0.86, 1.16, 8), 3)
    verdad = np.outer(fac, base) + rng.normal(0, 0.5, (8, 6))
    verdad = np.round(verdad, 1)
    # 16 huecos: cada camión conserva al menos 3 rutas y cada ruta al menos 4 camiones
    while True:
        M = np.ones((8, 6), bool); idx = rng.choice(48, 16, replace=False); M.ravel()[idx] = False
        if M.sum(1).min() >= 3 and M.sum(0).min() >= 4:
            break
    return verdad, M


def capa1(X, M, it=200):
    """Una capa (rango 1) ajustada solo a las casillas conocidas, por mínimos cuadrados alternos."""
    t = np.array([X[i, M[i]].mean() for i in range(X.shape[0])]) / 30.0
    for _ in range(it):
        r = np.array([(t[M[:, j]] @ X[M[:, j], j]) / (t[M[:, j]] @ t[M[:, j]]) for j in range(X.shape[1])])
        t = np.array([(r[M[i]] @ X[i, M[i]]) / (r[M[i]] @ r[M[i]]) for i in range(X.shape[0])])
    c = r.mean() / 30.0
    return t * c, r / c                          # factor de camión ≈ 1, consumo base de cada ruta


if __name__ == "__main__":
    V, M = tabla(); t, r = capa1(V, M); P = np.outer(t, r)
    print(V); print(M.astype(int)); print("camión", t.round(3)); print("ruta", r.round(2))
    h = ~M
    print("error medio huecos (1 capa):", np.abs(P - V)[h].mean().round(2), " máximo", np.abs(P - V)[h].max().round(2))
    cm = np.array([V[M[:, j], j].mean() for j in range(6)])
    print("error medio huecos (media de la ruta):", np.abs(cm[None, :] - V)[h].mean().round(2))
    print("error en casillas conocidas:", np.abs(P - V)[M].mean().round(2))
