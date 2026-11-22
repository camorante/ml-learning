"""¿Cuántos números en el código? El mismo autoencoder con k = 1, 2, 3, 4, 6 y 8, frente al PCA con k componentes,
y un autoencoder sin embudo (64 → 128 → 64 → 128 → 64). Para cada uno: error en validación normal y averías que
caza con el umbral del percentil 99 de validación."""
import json
import numpy as np
from sklearn.decomposition import PCA
from datos import X_TR, X_VA, X_TE, Y_TE, TIPOS
from modelos import entrenar, error, reconstruir

EJ = [int(np.flatnonzero(Y_TE == k)[0]) for k in range(5)]     # la primera ventana de cada tipo, para los dibujos
res, rec = {}, {}
print("   k   PCA val    AE val   AE: umbral · atascado · pico · deriva · vibración (de 50) · normales con aviso (de 2000)")
for k in [1, 2, 3, 4, 6, 8, 64]:
    if k < 64:
        p = PCA(k).fit(X_TR)
        e_pca = float(((p.inverse_transform(p.transform(X_VA)) - X_VA) ** 2).mean())
        net, _ = entrenar(X_TR, k=k, seed=0)
    else:
        e_pca = 0.0
        net, _ = entrenar(X_TR, k=64, m=128, seed=0)
    e_va, e_te = error(net, X_VA), error(net, X_TE)
    u = float(np.quantile(e_va, 0.99))
    caza = [int((e_te[Y_TE == c] > u).sum()) for c in range(1, 5)]
    res[k] = {"pca": e_pca, "ae": float(e_va.mean()), "umbral": u, "caza": caza, "fa": int((e_te[Y_TE == 0] > u).sum())}
    rec[k] = reconstruir(net, X_TE[EJ]).round(4).tolist()
    print(f"{k:4d}  {e_pca:8.4f}  {e_va.mean():8.4f}   {u:.4f} · " + " · ".join(f"{c:2d}" for c in caza) + f" · {res[k]['fa']}")
json.dump({"res": res, "ej": EJ, "rec": rec}, open("tamano.json", "w", encoding="utf-8"))
