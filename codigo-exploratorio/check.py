import numpy as np, pandas as pd
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from eda import quantile, skew, kurt, pearson, rangos, spearman, kendall_b, vif, centinelas, descomponer_covarianza
from sim import viajes
rng = np.random.default_rng(0); err = dict(q=0, sk=0, ku=0, p=0, sp=0, kt=0, rk=0)
for t in range(200):
    n = int(rng.integers(8, 300)); x = np.round(rng.lognormal(0, 1, n), int(rng.integers(0, 3))); y = np.round(x * rng.normal(1, .5, n) + rng.normal(0, 1, n), 1)
    for q in (0.1, 0.25, 0.5, 0.9): err["q"] = max(err["q"], abs(quantile(x, q) - np.quantile(x, q)))
    s = pd.Series(x)
    err["sk"] = max(err["sk"], abs(skew(x) - s.skew())); err["ku"] = max(err["ku"], abs(kurt(x) - s.kurt()))
    err["p"] = max(err["p"], abs(pearson(x, y) - stats.pearsonr(x, y)[0]))
    err["sp"] = max(err["sp"], abs(spearman(x, y) - stats.spearmanr(x, y)[0]))
    err["rk"] = max(err["rk"], np.abs(rangos(x) - stats.rankdata(x)).max())
    if n < 150: err["kt"] = max(err["kt"], abs(kendall_b(x, y) - stats.kendalltau(x, y)[0]))
print({k: float(f"{v:.1e}") for k, v in err.items()})
d = viajes(problemas=False); C = ["distancia_km", "km_gps", "vel_media", "duracion_h", "carga_t", "ralenti_pct", "temp_motor"]
X = d[C].values; Xc = add_constant(X)
ref = np.array([variance_inflation_factor(Xc, j + 1) for j in range(len(C))])
print("VIF propio  ", np.round(vif(X), 1)); print("VIF statsmod", np.round(ref, 1), "dif máx", float(np.abs(vif(X) - ref).max()))
print("centinelas temp", centinelas(viajes().temp_motor), "vel", centinelas(viajes().vel_media))
de, en, tot = descomponer_covarianza(d.vel_media, d.consumo, d.tipo)
print("cov dentro", round(de, 2), "entre", round(en, 2), "total", round(tot, 2), "suma", round(de + en, 2))
