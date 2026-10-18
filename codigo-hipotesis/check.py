"""Compara pruebas.py con scipy en 300 problemas aleatorios."""
import numpy as np
from scipy import stats
from fractions import Fraction
from itertools import combinations
from pruebas import *

rng = np.random.default_rng(0); err = {}
def up(k, v): err[k] = max(err.get(k, 0.0), float(v))
ALT = {"bilateral": "two-sided", "mayor": "greater", "menor": "less"}
for it in range(300):
    n1, n2 = int(rng.integers(3, 60)), int(rng.integers(3, 60)); alt = str(rng.choice(list(ALT)))
    x = np.round(rng.normal(32, rng.uniform(0.5, 3), n1), 1); y = np.round(rng.normal(32 - rng.uniform(-1, 2), rng.uniform(0.5, 3), n2), 1)
    w = welch(x, y, alt); r = stats.ttest_ind(x, y, equal_var=False, alternative=ALT[alt])
    up("welch: t, gl y p vs ttest_ind(equal_var=False)", max(abs(w["t"] - r.statistic), abs(w["gl"] - r.df), abs(w["p"] - r.pvalue)))
    ci = stats.ttest_ind(x, y, equal_var=False).confidence_interval(); w2 = welch(x, y)
    up("welch: intervalo vs .confidence_interval()", max(abs(w2["ic"][0] - ci.low), abs(w2["ic"][1] - ci.high)))
    s = student(x, y, alt); r = stats.ttest_ind(x, y, alternative=ALT[alt])
    up("student vs ttest_ind", max(abs(s["t"] - r.statistic), abs(s["p"] - r.pvalue)))
    m = min(n1, n2); e = emparejada(x[:m], y[:m], alt); r = stats.ttest_rel(x[:m], y[:m], alternative=ALT[alt])
    up("emparejada vs ttest_rel", max(abs(e["t"] - r.statistic), abs(e["p"] - r.pvalue)))
    u = mann_whitney(x, y, alt); r = stats.mannwhitneyu(x, y, alternative=ALT[alt], method="asymptotic")
    up("mann_whitney: U y p vs mannwhitneyu(method='asymptotic')", max(abs(u["U"] - r.statistic), abs(u["p"] - r.pvalue)))
    N1, N2 = int(rng.integers(20, 3000)), int(rng.integers(20, 3000)); k1, k2 = int(rng.integers(1, N1)), int(rng.integers(1, N2))
    T = [[k1, N1 - k1], [k2, N2 - k2]]; c = chi2_tabla(T); r = stats.chi2_contingency(T, correction=False)
    up("chi2_tabla 2×2 vs chi2_contingency(correction=False)", max(abs(c["chi2"] - r.statistic) / r.statistic, abs(c["p"] - r.pvalue)))
    z = z_proporciones(k1, N1, k2, N2); up("z_proporciones: z² − χ² y p", max(abs(z["z"]**2 - c["chi2"]) / c["chi2"], abs(z["p"] - c["p"])))
    T3 = rng.integers(5, 400, (int(rng.integers(2, 5)), int(rng.integers(2, 5)))); c = chi2_tabla(T3); r = stats.chi2_contingency(T3, correction=False)
    up("chi2_tabla r×c vs chi2_contingency", max(abs(c["chi2"] - r.statistic) / r.statistic, abs(c["p"] - r.pvalue)))
    F = rng.integers(0, 30, (2, 2)).tolist(); up("fisher_2x2 vs fisher_exact", abs(fisher_2x2(F)["p"] - stats.fisher_exact(F).pvalue))
    ps = rng.random(int(rng.integers(2, 40))) ** 2
    up("bh vs false_discovery_control(method='bh')", np.max(np.abs(bh(ps) - stats.false_discovery_control(ps, method="bh"))))
    o = np.argsort(ps); naive = np.empty(len(ps)); run = 0
    for i, j in enumerate(o): run = max(run, (len(ps) - i) * ps[j]); naive[j] = min(1, run)
    up("holm vs definición paso a paso", np.max(np.abs(holm(ps) - naive)))
    if it < 40:
        d, sg, a, b = rng.uniform(0.2, 2), rng.uniform(0.5, 3), int(rng.integers(2, 80)), int(rng.integers(2, 80)); al = float(rng.choice([0.01, 0.05, 0.1]))
        gl = a + b - 2; lam = d / (sg * np.sqrt(1 / a + 1 / b)); tc = stats.t.ppf(1 - al / 2, gl)
        ref = stats.nct.sf(tc, gl, lam) + stats.nct.cdf(-tc, gl, lam)
        up("potencia_t vs scipy.stats.nct", abs(potencia_t(d, sg, a, b, al) - ref))
    if it < 25:                                 # datos sin redondear (sin empates) frente a scipy
        a, b = rng.normal(32, 1, int(rng.integers(3, 8))), rng.normal(31, 1, int(rng.integers(3, 8)))
        f, al2 = (lambda u, v: abs(np.mean(u) - np.mean(v)), "greater") if alt == "bilateral" else (lambda u, v: np.mean(u) - np.mean(v), ALT[alt])
        ref = stats.permutation_test((a, b), f, permutation_type="independent", n_resamples=np.inf, alternative=al2).pvalue
        up("permutacion exacta vs permutation_test(n_resamples=inf)", abs(permutacion(a, b, alternativa=alt)["p"] - ref))
        a, b = np.round(a, 1), np.round(b, 1)      # con empates: frente a la cuenta exacta con fracciones
        A, B = [Fraction(round(v * 10), 10) for v in a], [Fraction(round(v * 10), 10) for v in b]; Z = A + B; n1 = len(A)
        obs = abs(sum(A) / n1 - sum(B) / len(B)); cuenta = [abs(sum(Z[i] for i in c) / n1 - (sum(Z) - sum(Z[i] for i in c)) / len(B)) >= obs
                                                            for c in combinations(range(len(Z)), n1)]
        up("permutacion exacta con empates vs recuento con fracciones", abs(permutacion(a, b)["p"] - sum(cuenta) / len(cuenta)))
w_ = max(len(k) for k in err)
for k, v in err.items(): print(f"{k:{w_}s}  {v:.1e}")
