"""Desde cero en NumPy: curva precisión-exhaustividad, precisión media (PR-AUC), área ROC, coste en euros, umbral por
costes y bootstrap por camión. check.py los compara con scikit-learn."""
import numpy as np

C_REV, C_AVE = 400.0, 4000.0          # revisión preventiva (mecánico + un día parado) y avería en ruta, en €
UMBRAL = C_REV / C_AVE                # avisar si p · 4 000 > 400 → p > 0,10


def curva_pr(y, p):
    """Recorre los umbrales de mayor a menor (uno por cada valor distinto de p). Devuelve exhaustividad, precisión
    y umbral en cada corte."""
    o = np.argsort(-p, kind="mergesort")
    y, p = np.asarray(y, float)[o], np.asarray(p)[o]
    corte = np.r_[np.flatnonzero(np.diff(p)), len(p) - 1]        # último índice de cada grupo de empates
    tp = np.cumsum(y)[corte]
    fp = corte + 1 - tp
    return tp / y.sum(), tp / (tp + fp), p[corte]


def pr_auc(y, p):
    """Precisión media: suma de (salto de exhaustividad) × (precisión) en cada umbral."""
    r, pr, _ = curva_pr(y, p)
    return float(np.sum(np.diff(np.r_[0, r]) * pr))


def roc_auc(y, p):
    """Probabilidad de que una avería al azar puntúe más que una foto sana al azar (empates cuentan ½), por rangos."""
    y = np.asarray(y).astype(bool)
    o = np.argsort(p, kind="mergesort"); ps = np.asarray(p)[o]
    rango = np.empty(len(p)); i = 0
    while i < len(p):                                            # rango medio en los empates
        j = i
        while j + 1 < len(p) and ps[j + 1] == ps[i]:
            j += 1
        rango[o[i:j + 1]] = (i + j) / 2 + 1; i = j + 1
    n1, n0 = y.sum(), (~y).sum()
    return float((rango[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def coste(y, p, u=UMBRAL):
    """Coste medio por camión y mes: 400 € por cada aviso y 4 000 € por cada avería sin aviso."""
    a = np.asarray(p) > u
    return float(np.mean(a * C_REV + (~a) * np.asarray(y) * C_AVE))


def matriz(y, p, u=UMBRAL):
    a, y = np.asarray(p) > u, np.asarray(y).astype(bool)
    return dict(vp=int((a & y).sum()), fp=int((a & ~y).sum()), fn=int((~a & y).sum()), vn=int((~a & ~y).sum()))


def bootstrap(cam, y, p, f, B=2000, seed=0):
    """Remuestrea camiones enteros (con sus 4 meses) con reemplazo, B veces, y devuelve f en cada réplica."""
    rng = np.random.default_rng(seed)
    u, inv = np.unique(cam, return_inverse=True)
    grupos = [np.flatnonzero(inv == k) for k in range(len(u))]
    out = np.empty(B)
    for b in range(B):
        idx = np.concatenate([grupos[k] for k in rng.integers(0, len(u), len(u))])
        out[b] = f(y[idx], p[idx])
    return out
