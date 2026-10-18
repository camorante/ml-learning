"""Compara ic.py con scipy en 300 problemas aleatorios."""
import numpy as np
from scipy import stats
from ic import *

rng = np.random.default_rng(0); err = {}
def up(k, v): err[k] = max(err.get(k, 0.0), float(v))
for t in range(300):
    n = int(rng.integers(3, 120)); conf = float(rng.choice([0.8, 0.9, 0.95, 0.99]))
    x = rng.lognormal(3.4, float(rng.uniform(0.05, 0.6)), n)
    up("t_ppf vs stats.t.ppf", abs(t_ppf((1 + conf) / 2, n - 1) - stats.t.ppf((1 + conf) / 2, n - 1)))
    lo, hi = ic_media(x, conf); ref = stats.t.interval(conf, n - 1, x.mean(), stats.sem(x))
    up("ic_media (t) vs stats.t.interval", max(abs(lo - ref[0]), abs(hi - ref[1])))
    lo, hi = ic_media(x, conf, sigma=2.0); ref = stats.norm.interval(conf, x.mean(), 2.0 / np.sqrt(n))
    up("ic_media (z) vs stats.norm.interval", max(abs(lo - ref[0]), abs(hi - ref[1])))
    k = int(rng.integers(0, min(n, 15) + 1)); bt = stats.binomtest(k, n)
    for f, m in [(ic_wilson, "wilson"), (ic_clopper_pearson, "exact")]:
        lo, hi = f(k, n, conf); ci = bt.proportion_ci(conf, method=m)
        up(f"{f.__name__} vs binomtest().proportion_ci('{m}')", max(abs(lo - ci.low), abs(hi - ci.high)))
    y = rng.normal(x.mean() * 1.05, x.std() * 1.5, int(rng.integers(3, 120)))
    lo, hi = ic_diferencia(x, y, conf); ci = stats.ttest_ind(x, y, equal_var=False).confidence_interval(conf)
    up("ic_diferencia (Welch) vs ttest_ind(equal_var=False)", max(abs(lo - ci.low), abs(hi - ci.high)))
    if t < 60:                                   # bootstrap: mismos remuestreos que scipy con la misma semilla
        for est, nom in [(np.mean, "media"), (np.median, "mediana")]:
            for met, sm in [("percentil", "percentile"), ("basico", "basic"), ("BCa", "BCa")]:
                mine = bootstrap(x, est, B=2999, conf=conf, metodo=met, seed=t)
                ref = stats.bootstrap((x,), est, n_resamples=2999, confidence_level=conf, method=sm, rng=np.random.default_rng(t)).confidence_interval
                up(f"bootstrap {met} ({nom}) vs stats.bootstrap", max(abs(mine[0] - ref.low), abs(mine[1] - ref.high)))
w = max(len(k) for k in err)
for k, v in err.items(): print(f"# {k:{w}s}  {v:.1e}")
