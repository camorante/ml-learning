"""Comprobaciones: las métricas desde cero frente a scikit-learn, el umbral por costes frente a la fuerza bruta, el
bootstrap frente a la fórmula del error típico, y la red entera (GRU incluida) en NumPy frente a PyTorch."""
import json
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve
from comun import trozos
from metricas import pr_auc, roc_auc, curva_pr, coste, bootstrap, UMBRAL, C_REV, C_AVE
from red import Red

D, F, E, _ = trozos()
P = np.load("probs.npz")
y = F["te"]["y"]

# 1. PR-AUC, ROC-AUC y la curva PR, frente a scikit-learn, en las tres listas de probabilidades del examen
e_ap = max(abs(pr_auc(y, P[k]) - average_precision_score(y, P[k])) for k in ("red completa|0", "boosting", "logistica"))
e_roc = max(abs(roc_auc(y, P[k]) - roc_auc_score(y, P[k])) for k in ("red completa|0", "boosting", "logistica"))
r, pr, u = curva_pr(y, P["boosting"])
ps, rs, us = precision_recall_curve(y, P["boosting"])            # sklearn: de menor a mayor umbral, con un punto extra
e_cur = max(np.abs(pr - ps[:-1][::-1]).max(), np.abs(r - rs[:-1][::-1]).max(), np.abs(u - us[::-1]).max())
print(f"PR-AUC frente a average_precision_score: {e_ap:.1e}")
print(f"ROC-AUC frente a roc_auc_score:          {e_roc:.1e}")
print(f"curva PR frente a precision_recall_curve: {e_cur:.1e} ({len(r)} umbrales)")

# 2. el umbral 0,10 minimiza el coste esperado según las probabilidades: fuerza bruta sobre todos los cortes posibles
p = P["red completa|0"]
esperado = lambda a: np.sum(np.where(a, C_REV, p * C_AVE))     # € esperados si avisamos a los marcados en a
reglas = np.r_[0, np.sort(p)]
mejor = min(esperado(p > t) for t in reglas)
print(f"coste esperado con p > {UMBRAL:.2f}: {esperado(p > UMBRAL):.4f} €; mejor corte por fuerza bruta: {mejor:.4f} €"
      f" (diferencia {esperado(p > UMBRAL) - mejor:.1e})")

# 3. bootstrap por filas del coste medio frente al error típico de la fórmula s/√n
c = np.where(p > UMBRAL, C_REV, y * C_AVE)
a = bootstrap(np.arange(len(y)), c, c, lambda yy, pp: yy.mean(), B=4000, seed=3)
se = c.std(ddof=1) / np.sqrt(len(c))
print(f"error típico del coste medio: bootstrap {a.std():.3f} €, fórmula {se:.3f} € (diferencia relativa {abs(a.std() / se - 1):.1%})")
b = bootstrap(F["te"]["cam"], c, c, lambda yy, pp: yy.mean(), B=4000, seed=3)
print(f"remuestreando camiones enteros: {b.std():.3f} € (los 4 meses de un camión no son independientes)")

# 4. la corrección del peso de clase: σ(z − ln w) es lo mismo que p / (p + w·(1 − p))
w = json.load(open("comparar.json", encoding="utf-8"))["curvas"]["red completa|0"]["w"]
z = np.linspace(-6, 6, 1001); q = 1 / (1 + np.exp(-z))
print(f"σ(z − ln w) frente a p/(p + w(1 − p)): {np.abs(1 / (1 + np.exp(-(z - np.log(w)))) - q / (q + w * (1 - q))).max():.1e}")

# 5. la red completa, desde cero en NumPy (GRU, embeddings, tabla, cabeza), frente a PyTorch en float64
m = Red(); m.load_state_dict(torch.load("modelo.pt")); m = m.double().eval()
W = {k: v.detach().numpy() for k, v in m.state_dict().items()}
sig = lambda v: 1 / (1 + np.exp(-v))


def gru_np(x):
    """x: (n, 28, 6). Ecuaciones de PyTorch: puertas r, z y candidata n; h ← (1 − z)·n + z·h."""
    Wi, Wh, bi, bh = W["gru.weight_ih_l0"], W["gru.weight_hh_l0"], W["gru.bias_ih_l0"], W["gru.bias_hh_l0"]
    H = Wh.shape[1]; h = np.zeros((x.shape[0], H))
    for t in range(x.shape[1]):
        gi, gh = x[:, t] @ Wi.T + bi, h @ Wh.T + bh
        r, zz = sig(gi[:, :H] + gh[:, :H]), sig(gi[:, H:2 * H] + gh[:, H:2 * H])
        n = np.tanh(gi[:, 2 * H:] + r * gh[:, 2 * H:])
        h = (1 - zz) * n + zz * h
    return h


def red_np(e):
    h = np.concatenate([gru_np(e["seq"].transpose(0, 2, 1).astype(np.float64)), W["e_mod.weight"][e["mod"]],
                        W["e_dep.weight"][e["dep"]], e["tab"]], 1)
    return np.maximum(0, h @ W["h1.weight"].T + W["h1.bias"]) @ W["out.weight"][0] + W["out.bias"][0]


with torch.no_grad():
    zt = m({k: torch.as_tensor(v).double() if v.dtype.kind == "f" else torch.as_tensor(v) for k, v in E["te"].items()}).numpy()
zn = red_np(E["te"])
print(f"logits de la red en NumPy frente a PyTorch (2 400 fotos del examen): {np.abs(zn - zt).max():.1e}")
print(f"y la probabilidad corregida frente a la guardada (float32): {np.abs(sig(zn - np.log(w)) - p).max():.1e}")
