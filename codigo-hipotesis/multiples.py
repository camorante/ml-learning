import numpy as np
from scipy import stats
from sim import METRICAS
from pruebas import bonferroni, holm, bh

rng = np.random.default_rng(13); R, m = 20_000, 20
# un curso inútil, 20 métricas, 30 camiones por grupo: todas las H0 son ciertas
x, y = rng.normal(0, 1, (R, m, 30)), rng.normal(0, 1, (R, m, 30))
P = stats.ttest_ind(x, y, axis=2).pvalue
print(f"1 − 0,95^20 = {1 - 0.95**20:.4f}")
for nom, f in [("sin corregir", lambda p: p), ("Bonferroni", bonferroni), ("Holm", holm), ("Benjamini-Hochberg", bh)]:
    adj = np.apply_along_axis(f, 1, P)
    print(f"{nom:19s} P(al menos un «hallazgo») = {np.mean((adj < 0.05).any(1)):.4f}")
ej = P[0]; o = np.argsort(ej)
print("el primer curso:", [(METRICAS[i], round(float(ej[i]), 3)) for i in o[:3]])
# 20 métricas, 5 con efecto real (d = 0,8): potencia y tasa de falsos descubrimientos
efecto = np.r_[np.full(5, 0.8), np.zeros(15)]
x, y = rng.normal(0, 1, (R, m, 30)), rng.normal(0, 1, (R, m, 30)) - efecto[None, :, None]
P = stats.ttest_ind(x, y, axis=2).pvalue
for nom, f in [("sin corregir", lambda p: p), ("Bonferroni", bonferroni), ("Holm", holm), ("Benjamini-Hochberg", bh)]:
    rej = np.apply_along_axis(f, 1, P) < 0.05; V = rej[:, 5:].sum(1); S = rej[:, :5].sum(1)
    fdr = np.mean(np.where(V + S > 0, V / np.maximum(V + S, 1), 0))
    print(f"{nom:19s} reales detectados {S.mean() / 5:.3f}   FWER {np.mean(V > 0):.3f}   FDR {fdr:.3f}")
