import math
import numpy as np
from scipy import stats
from sim import SIN_CURSO as a, CON_CURSO as b, euros_anio

r = stats.ttest_ind(a, b, equal_var=False)               # Welch: no supone varianzas iguales
ee = math.sqrt(a.var(ddof=1) / 10 + b.var(ddof=1) / 10)
print(f"s = {a.std(ddof=1):.3f} y {b.std(ddof=1):.3f}   error estándar de la diferencia = {ee:.4f}")
print(f"Welch:   t = {r.statistic:.4f}  gl = {r.df:.2f}  p = {r.pvalue:.4f}")
ci = r.confidence_interval(0.95); print(f"         IC 95 % de la diferencia: [{ci.low:.3f}, {ci.high:.3f}]")
s = stats.ttest_ind(a, b); print(f"Student: t = {s.statistic:.4f}  gl = {s.df:.0f}  p = {s.pvalue:.4f}")
u = stats.ttest_ind(a, b, equal_var=False, alternative="greater"); print(f"unilateral (H1: el curso ahorra): p = {u.pvalue:.4f}")
print(f"aproximación normal (z = t): p = {2 * stats.norm.sf(r.statistic):.4f}")
# dualidad con el intervalo: el IC del (1 − α) excluye el 0 exactamente cuando p < α
for conf in (0.90, 0.95, 0.96, 0.97, 0.99):
    c = r.confidence_interval(conf); print(f"IC {conf:.0%}: [{c.low:6.3f}, {c.high:.3f}]  ¿excluye 0? {str(c.low > 0):5s}  ¿p < {1 - conf:.2f}? {r.pvalue < 1 - conf}")
d = (a.mean() - b.mean()) / math.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
print(f"tamaño del efecto (d de Cohen) = {d:.2f}")
print(f"en euros: {euros_anio(1.6):.0f} €/camión/año (IC: {euros_anio(ci.low):.0f} a {euros_anio(ci.high):.0f} €)")
