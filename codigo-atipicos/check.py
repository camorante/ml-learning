import numpy as np
from scipy import stats
from scipy.spatial.distance import mahalanobis as mh
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import OLSInfluence
from sklearn.ensemble import IsolationForest
from sklearn.ensemble._iforest import _average_path_length
from outl import *
from sim import viajes
rng = np.random.default_rng(0); e = dict(rz=0, iqr=0, huber=0, cook=0, mah=0, cn=0)
for t in range(100):
    n = int(rng.integers(20, 300)); x = rng.uniform(0, 20, n); y = 3 + 0.8 * x + rng.standard_t(2, n)
    e["rz"] = max(e["rz"], np.abs(z_robusto(y) - (y - np.median(y)) / stats.median_abs_deviation(y, scale="normal")).max())
    lo, hi = vallas_iqr(y); q1, q3 = np.quantile(y, [.25, .75]); e["iqr"] = max(e["iqr"], abs(lo - (q1 - 1.5 * (q3 - q1))))
    b, _ = huber_irls(x, y); r = sm.RLM(y, sm.add_constant(x), M=sm.robust.norms.HuberT(1.345)).fit(conv="coefs", tol=1e-12, maxiter=500)
    e["huber"] = max(e["huber"], np.abs(b - r.params).max())
    c, h = cook(x, y); e["cook"] = max(e["cook"], np.abs(c - OLSInfluence(sm.OLS(y, sm.add_constant(x)).fit()).cooks_distance[0]).max())
    X = np.c_[x, y]; S = np.cov(X.T); mu = X.mean(0)
    e["mah"] = max(e["mah"], np.abs(mahalanobis(X, mu, S) - np.array([mh(p, mu, np.linalg.inv(S)) for p in X])).max())
ns = np.arange(1, 2000); e["cn"] = float(np.abs(c_n(ns) - _average_path_length(ns)).max())
print({k: float(f"{v:.1e}") for k, v in e.items()})
d = viajes(); X = d[["carga_t", "consumo"]].values
mine = isolation_forest(X, 200, 256, 0); sk = -IsolationForest(n_estimators=200, max_samples=256, random_state=0).fit(X).score_samples(X)
from sklearn.metrics import roc_auc_score
print("iForest propio vs sklearn: Spearman", round(stats.spearmanr(mine, sk)[0], 3),
      "AUC errores propio", round(roc_auc_score(d.es_error, mine), 3), "sklearn", round(roc_auc_score(d.es_error, sk), 3))
