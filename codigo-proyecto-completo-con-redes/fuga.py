"""La trampa de mirar antes de separar. Un analista con prisa calcula para cada camión sus «averías al año» con
TODA la tabla (los 2 años, incluidos los meses que luego quiere predecir), la pone en lugar del historial de los
12 meses anteriores y reparte las fotos AL AZAR (70/15/15) entre los 2 400 camiones de entrenamiento y validación
(meses 0–19). Su validación promete. El examen de verdad (600 camiones nuevos, meses 20–23) solo puede usar el
pasado: las averías de los 12 meses anteriores. Se compara con el reparto al azar sin la trampa y con el correcto."""
import json, os
import numpy as np
from multiprocessing import Pool


def tarea(a):
    nombre, seed = a
    from datos import cargar, reparto, filas
    from rasgos import preparar
    from red import entrenar, prob
    from metricas import pr_auc, coste
    D, R = cargar(), reparto()
    cam = np.sort(np.r_[R["tr"][0], R["va"][0]])
    todo = filas(D, "x", dict(x=(cam, np.arange(20))))
    if nombre == "tasa de toda la tabla":
        f = D["fallos"]; S0 = D["S"][0]
        anual = np.bincount(f[f[:, 1] >= S0, 0], minlength=len(D["mod"])) / 2   # averías en los 2 años / 2
        todo["tab"] = todo["tab"].copy(); todo["tab"][:, 4] = anual[todo["cam"]]
    rng = np.random.default_rng(100 + seed)
    o = rng.permutation(len(todo["y"])); n = len(o)
    parte = {"tr": o[:int(.7 * n)], "va": o[int(.7 * n):int(.85 * n)], "pr": o[int(.85 * n):]}
    sub = {k: {c: v[i] for c, v in todo.items()} for k, i in parte.items()}
    E = {}
    E["tr"], esc = preparar(sub["tr"])
    for k in ("va", "pr"):
        E[k] = preparar(sub[k], esc)[0]
    E["te"] = preparar(filas(D, "te", R), esc)[0]          # en el examen, el historial solo puede ser el pasado
    m, info = entrenar(E["tr"], E["va"], seed=seed)
    ppr, pte = prob(m, E["pr"], info["w"]), prob(m, E["te"], info["w"])
    return a, [pr_auc(E["pr"]["y"], ppr), coste(E["pr"]["y"], ppr)], [pr_auc(E["te"]["y"], pte), coste(E["te"]["y"], pte)]


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    with Pool(6) as pool:
        R = pool.map(tarea, [(n, s) for n in ("tasa de toda la tabla", "historial del pasado") for s in (0, 1, 2)])
    from comun import trozos
    from metricas import pr_auc, coste
    _, F, _, _ = trozos()
    P = np.load("probs.npz")
    bien = [[[pr_auc(F["va"]["y"], P[f"va|{s}"]), coste(F["va"]["y"], P[f"va|{s}"])],
             [pr_auc(F["te"]["y"], P[f"red completa|{s}"]), coste(F["te"]["y"], P[f"red completa|{s}"])]] for s in (0, 1, 2)]
    out = {}
    print(f"{'':40s} {'promete (PR-AUC, €)':>22s}   {'examen real (PR-AUC, €)':>24s}")
    for n, filas in [("al azar, averías de toda la tabla", [(r[1], r[2]) for r in R if r[0][0] == "tasa de toda la tabla"]),
                     ("al azar, averías solo del pasado", [(r[1], r[2]) for r in R if r[0][0] == "historial del pasado"]),
                     ("por camión y por tiempo", [tuple(b) for b in bien])]:
        a = np.array(filas)                                                  # (semilla, promesa/real, métrica)
        out[n] = np.round(a, 4).tolist()
        print(f"{n:40s} {a[:, 0, 0].mean():10.3f} {a[:, 0, 1].mean():9.1f} €   {a[:, 1, 0].mean():12.3f} {a[:, 1, 1].mean():9.1f} €")
        print(f"{'':40s} semillas PR-AUC: " + " · ".join(f"{x:.3f}→{y:.3f}" for x, y in zip(a[:, 0, 0], a[:, 1, 0])))
    json.dump(out, open("fuga.json", "w", encoding="utf-8"), separators=(",", ":"))
