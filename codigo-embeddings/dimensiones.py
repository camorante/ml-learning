"""Barrido de la dimensión del embedding (1, 2, 3, 4, 8 y 16) frente a la tabla one-hot, con tres semillas, y
la mejor pérdida posible (las frecuencias de 200 000 visitas simuladas). Guarda dimensiones.json."""
import json, os
import numpy as np
import torch
from sklearn.cluster import KMeans
from datos import V, NOMBRES, SISTEMA, P_TR, P_TE, visitas, pares
from modelos import Emb, Tabla, entrenar, n_param, pureza

os.chdir(os.path.dirname(os.path.abspath(__file__)))
DIMS = [1, 2, 3, 4, 8, 16]
res, Es = {}, {}
print("modelo        parámetros  pérdida entren.  pérdida prueba (3 semillas)   top-3 prueba  vecinos del mismo sistema")
for d in DIMS + ["one-hot"]:
    fila = []
    for seed in (0, 1, 2):
        torch.manual_seed(seed)
        m = Tabla(V) if d == "one-hot" else Emb(V, d)
        h, _ = entrenar(m, P_TR, P_TE, epocas=60, lr=0.02, lote=64, seed=seed)
        pu = pureza(m.E.weight.detach().numpy(), SISTEMA)[0] if d != "one-hot" else None
        fila.append((h[-1][1], h[-1][3], h[-1][4], pu, min(r[3] for r in h)))
        if seed == 0 and d != "one-hot":
            Es[d] = m.E.weight.detach().numpy()
    f = np.array([r[:3] for r in fila]); pus = [r[3] for r in fila]
    nombre = f"embedding d={d}" if d != "one-hot" else "one-hot (tabla)"
    print(f"{nombre:15s} {n_param(m):6d}  {f[:, 0].mean():15.3f}  {f[0, 1]:.3f} {f[1, 1]:.3f} {f[2, 1]:.3f} (media {f[:, 1].mean():.3f})"
          f"  {f[:, 2].mean():11.1%}  " + (" · ".join(f"{p}" for p in pus) + " de 60" if pus[0] is not None else "—"))
    res[str(d)] = {"param": n_param(m), "tr": round(f[:, 0].mean(), 4), "te": [round(v, 4) for v in f[:, 1]],
                   "top3": round(f[:, 2].mean(), 4), "pureza": pus, "mejor_te": [round(r[4], 4) for r in fila]}
print("con parada temprana (la mejor época en prueba), one-hot: "
      + ", ".join(f"{v:.3f}" for v in res["one-hot"]["mejor_te"]))

# la referencia: las probabilidades de verdad, estimadas con 200 000 visitas simuladas
C = np.zeros((V, V))
for a, b in pares(visitas(200_000, 99)):
    C[a, b] += 1
Pv = C / C.sum(1, keepdims=True)
ref = -np.mean(np.log(Pv[P_TE[:, 0], P_TE[:, 1]]))
marg = np.bincount(P_TR[:, 1], minlength=V) / len(P_TR)
frec = marg[P_TE[:, 1]] / (1 - marg[P_TE[:, 0]])          # solo la frecuencia de cada código, sin el visto
print(f"referencias en prueba: las probabilidades de verdad {ref:.3f} · solo la frecuencia de cada código {-np.mean(np.log(frec)):.3f}"
      f" · al azar (ln 19) {np.log(19):.3f}")

# d = 16: 16 coordenadas no se pueden dibujar; PCA las proyecta en las 2 direcciones de más variación
E16 = Es[16]; Z = E16 - E16.mean(0)
U, s, Vt = np.linalg.svd(Z, full_matrices=False)
pca = Z @ Vt[:2].T
var = s ** 2 / (s ** 2).sum()
print(f"\nd=16, semilla 0: PCA explica {var[0]:.1%} + {var[1]:.1%} = {var[:2].sum():.1%} de la variación")
Nn = E16 / np.linalg.norm(E16, axis=1, keepdims=True)
km = KMeans(4, n_init=10, random_state=0).fit(Nn)
for c in range(4):
    print(f"  grupo {c + 1} de k-means: " + ", ".join(NOMBRES[i] for i in range(V) if km.labels_[i] == c))
json.dump({"res": res, "ref": round(ref, 4), "E": {str(d): np.round(Es[d], 4).tolist() for d in DIMS},
           "pca16": np.round(pca, 4).tolist(), "var16": [round(v, 4) for v in var[:2]], "km16": km.labels_.tolist()},
          open("dimensiones.json", "w", encoding="utf-8"))
