import numpy as np
from sklearn import metrics as M
from imblearn.over_sampling import SMOTE
from imb import *
rng = np.random.default_rng(0); e = dict(metricas=0, roc=0, auc=0, pr=0, ap=0)
for t in range(200):
    n = int(rng.integers(30, 3000)); y = rng.random(n) < rng.uniform(0.01, 0.3); y[0] = True; y[1] = False
    s = np.round(rng.normal(y * rng.uniform(0, 2), 1), int(rng.integers(0, 3)))
    th = float(np.quantile(s, rng.uniform(0.5, 0.99))); a = s >= th
    c = matriz(y, s, th); m = metricas(**c, beta=2.0)
    ref = [M.accuracy_score(y, a), M.precision_score(y, a, zero_division=0), M.recall_score(y, a), M.fbeta_score(y, a, beta=2.0, zero_division=0),
           M.balanced_accuracy_score(y, a), M.matthews_corrcoef(y, a)]
    mine = [m["acierto"], m["precision"], m["exhaustividad"], m["f"], m["acierto_equilibrado"], m["mcc"]]
    e["metricas"] = max(e["metricas"], np.abs(np.array(mine) - ref).max())
    fx, ty, _ = curva_roc(y, s); rf, rt, _ = M.roc_curve(y, s, drop_intermediate=False)
    e["roc"] = max(e["roc"], np.abs(fx - rf).max() + np.abs(ty - rt).max())
    e["auc"] = max(e["auc"], abs(area(fx, ty) - M.roc_auc_score(y, s)))
    p, r, _ = curva_pr(y, s); rp, rr, _ = M.precision_recall_curve(y, s, drop_intermediate=False)
    e["pr"] = max(e["pr"], np.abs(p - rp[:-1][::-1]).max() + np.abs(r - rr[:-1][::-1]).max())
    e["ap"] = max(e["ap"], abs(precision_media(y, s) - M.average_precision_score(y, s)))
print({k: float(f"{v:.1e}") for k, v in e.items()})
# SMOTE: propiedades (cada sintético en un segmento entre dos positivos) y tamaño igual que imblearn
X = rng.normal(size=(400, 3)); y = rng.random(400) < 0.08
Xs, ys = smote(X, y, k=5, seed=1); Xi, yi = SMOTE(k_neighbors=5, random_state=1).fit_resample(X, y)
P = X[y]; N = Xs[len(X):]
dist = max(np.min([np.linalg.norm(np.cross(q - P[a], P[b] - P[a])) / (np.linalg.norm(P[b] - P[a]) + 1e-12) for a in range(len(P)) for b in range(len(P)) if a != b]) for q in N[:20])
print("SMOTE: positivos tras remuestrear propio", int(ys.sum()), "imblearn", int(yi.sum()), "| distancia máx. de un sintético a su segmento", f"{dist:.1e}")
