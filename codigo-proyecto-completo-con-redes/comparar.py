"""La tabla de comparación: referencias, la red completa y la red sin cada rama (ablación), tres semillas.
Métricas en el examen (600 camiones nuevos × 4 meses futuros = 2 400 fotos), vistas una sola vez."""
import json, os
import numpy as np
import torch
from multiprocessing import Pool

CONF = [("red completa", ("seq", "emb", "tab"), None), ("sin secuencia", ("emb", "tab"), None),
        ("sin embeddings", ("seq", "tab"), None), ("sin tabla", ("seq", "emb"), None),
        ("CNN en vez de GRU", ("cnn", "emb", "tab"), None), ("sin pesos de clase", ("seq", "emb", "tab"), 1.0)]


def tarea(a):
    nombre, ramas, w, seed = a
    from comun import trozos
    from red import entrenar, prob
    _, F, E, _ = trozos()
    m, info = entrenar(E["tr"], E["va"], seed=seed, ramas=ramas, w=w)
    if nombre == "red completa" and seed == 0:
        torch.save(m.state_dict(), "modelo.pt")
    return a, info, prob(m, E["va"], info["w"]), prob(m, E["te"], info["w"])


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    from comun import trozos
    from base import logistica, boosting
    from metricas import pr_auc, roc_auc, coste
    D, F, E, _ = trozos()
    yva, yte = F["va"]["y"], F["te"]["y"]
    with Pool(4) as pool:
        R = pool.map(tarea, [(n, r, w, s) for n, r, w in CONF for s in (0, 1, 2)])
    filas, probs = [], {}
    met = lambda y, p: [pr_auc(y, p), roc_auc(y, p), coste(y, p)]
    yv0 = np.zeros_like(yte)
    filas.append(("nunca avisar", [[yte.mean(), 0.5, coste(yte, yv0)]] * 3, None))
    filas.append(("avisar a todos", [[yte.mean(), 0.5, coste(yte, yv0 + 1)]] * 3, None))
    pva, pte = logistica(F["tr"], F["va"], F["te"])
    filas.append(("regresión logística", [met(yte, pte)] * 3, [pr_auc(yva, pva)] * 3)); probs["logistica"] = pte
    pva, pte, k = boosting(F["tr"], F["va"], F["te"])
    filas.append(("boosting de árboles", [met(yte, pte)] * 3, [pr_auc(yva, pva)] * 3)); probs["boosting"] = pte
    print(f"boosting: {k} árboles elegidos con la validación")
    curvas = {}
    for n, _, _ in CONF:
        rs = [r for r in R if r[0][0] == n]
        filas.append((n, [met(yte, r[3]) for r in rs], [pr_auc(yva, r[2]) for r in rs]))
        for r in rs:
            probs[f"{n}|{r[0][3]}"] = r[3]
            curvas[f"{n}|{r[0][3]}"] = dict(curva=r[1]["curva"], mejor=r[1]["mejor"], w=r[1]["w"])
            if n == "red completa":
                probs[f"va|{r[0][3]}"] = r[2]
    print(f"{'modelo':22s} {'PR-AUC examen':>22s} {'ROC-AUC':>8s} {'€ por camión y mes':>20s}  PR-AUC validación")
    for n, ms, vs in filas:
        ms = np.array(ms)
        ap = " ".join(f"{v:.3f}" for v in ms[:, 0]) if vs else f"{ms[0, 0]:.3f} (= tasa de averías)"
        v = "" if vs is None else f"{np.mean(vs):.3f}"
        print(f"{n:22s} {ap:>22s} {ms[:, 1].mean():8.3f} {ms[:, 2].mean():20.1f}  {v}")
    np.savez_compressed("probs.npz", **probs)
    json.dump(dict(filas=[(n, np.round(ms, 4).tolist(), None if vs is None else np.round(vs, 4).tolist()) for n, ms, vs in filas],
                   curvas=curvas), open("comparar.json", "w", encoding="utf-8"), separators=(",", ":"))
