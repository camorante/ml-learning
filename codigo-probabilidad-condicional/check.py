import numpy as np, pandas as pd
from scipy import stats
from scipy.stats.contingency import association
from cond import tabla, conjunta, marginales, condicional, esperada, chi2, cramer_v, bayes, prob_total

rng = np.random.default_rng(0)
err = dict(tabla=0, conj=0, filas=0, cols=0, marg=0, esp=0, chi=0, p=0, yates=0, v=0, bayes=0)
for it in range(200):
    r, c = rng.integers(2, 6, 2); n = int(rng.integers(50, 5000))
    a = rng.integers(0, r, n); b = (a + rng.integers(0, c, n) * rng.integers(0, 2, n)) % c
    T, _, _ = tabla(a, b); X = pd.crosstab(a, b)
    err["tabla"] = max(err["tabla"], np.abs(T - X.values).max())
    err["conj"] = max(err["conj"], np.abs(conjunta(T) - pd.crosstab(a, b, normalize="all").values).max())
    err["filas"] = max(err["filas"], np.abs(condicional(T, "filas") - pd.crosstab(a, b, normalize="index").values).max())
    err["cols"] = max(err["cols"], np.abs(condicional(T, "columnas") - pd.crosstab(a, b, normalize="columns").values).max())
    pa, pb = marginales(T)
    err["marg"] = max(err["marg"], np.abs(pa - pd.Series(a).value_counts(normalize=True).sort_index().values).max())
    s = stats.chi2_contingency(T, correction=False)
    err["esp"] = max(err["esp"], np.abs(esperada(T) - s.expected_freq).max())
    e, g, p = chi2(T)
    err["chi"] = max(err["chi"], abs(e - s.statistic) / s.statistic); err["p"] = max(err["p"], abs(p - s.pvalue) / max(s.pvalue, 1e-300))
    err["v"] = max(err["v"], abs(cramer_v(T) - association(T, method="cramer")))
    if r == 2 and c == 2:
        s2 = stats.chi2_contingency(T, correction=True)
        err["yates"] = max(err["yates"], abs(chi2(T, yates=True)[0] - s2.statistic))
    # Bayes: P(fila | columna j) a partir de P(fila) y P(columna j | fila)
    j = 0; post = bayes(pa, condicional(T, "filas")[:, j])
    err["bayes"] = max(err["bayes"], np.abs(post - condicional(T, "columnas")[:, j]).max())
    err["bayes"] = max(err["bayes"], abs(prob_total(pa, condicional(T, "filas")[:, j]) - pb[j]))
for k, v in err.items():
    print(f"{k:6s} {v:.1e}")
