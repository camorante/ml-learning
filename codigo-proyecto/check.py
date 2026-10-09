import numpy as np, pandas as pd
from scipy import stats
from monitor import psi, ks, ks_p, cortes
rng = np.random.default_rng(0); e = dict(ks=0, psi=0, ks_p=0)
for k in range(200):
    n, m = int(rng.integers(30, 3000)), int(rng.integers(30, 3000))
    a = np.round(rng.normal(0, 1, n), int(rng.integers(0, 3))); b = np.round(rng.normal(rng.uniform(-0.5, 0.5), rng.uniform(0.7, 1.5), m), int(rng.integers(0, 3)))
    e["ks"] = max(e["ks"], abs(ks(a, b) - stats.ks_2samp(a, b).statistic))           # con empates incluidos
    D = ks(a, b); en = np.sqrt(n * m / (n + m))
    e["ks_p"] = max(e["ks_p"], abs(ks_p(D, n, m) - stats.kstwobign.sf((en + 0.12 + 0.11 / en) * D)))
    a2 = rng.normal(0, 1, n); b2 = rng.normal(0.3, 1.2, m); bins = cortes(a2)    # PSI frente a una versión con pandas.cut
    ea = pd.Series(pd.cut(a2, bins, right=False)).value_counts(normalize=True, sort=False).values
    ab = pd.Series(pd.cut(b2, bins, right=False)).value_counts(normalize=True, sort=False).values
    ea = np.clip(ea, 1e-4, None); ab = np.clip(ab, 1e-4, None)
    e["psi"] = max(e["psi"], abs(psi(a2, b2) - np.sum((ab - ea) * np.log(ab / ea))))
print({k: float(f"{v:.1e}") for k, v in e.items()})
