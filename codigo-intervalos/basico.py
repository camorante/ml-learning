import math
import numpy as np
from scipy import stats
from sim import DIEZ as x, SIGMA

n, m, s = len(x), x.mean(), x.std(ddof=1)
se = s / math.sqrt(n)
print(f"n = {n}  media = {m:.2f}  s = {s:.4f}  error estándar = {se:.4f}")
print("z con σ = 4,5 conocida:", np.round(stats.norm.interval(0.95, m, SIGMA / math.sqrt(n)), 3))
print("1,96 con s (mal):      ", np.round([m - 1.96 * se, m + 1.96 * se], 3))
print("t con n − 1 = 9 g.l.:  ", np.round(stats.t.interval(0.95, n - 1, m, se), 3), " t =", round(stats.t.ppf(0.975, n - 1), 4))
print("scipy.stats.ttest_1samp(x, 32).confidence_interval():", np.round(stats.ttest_1samp(x, 32).confidence_interval(0.95), 3))
for c in (0.80, 0.90, 0.95, 0.99):
    print(f"nivel {c:.0%}: z = {stats.norm.ppf((1 + c) / 2):.3f}   t(9) = {stats.t.ppf((1 + c) / 2, 9):.3f}   t(29) = {stats.t.ppf((1 + c) / 2, 29):.3f}")
print("n para un margen de ±0,5 con σ = 4,5:", math.ceil((1.959964 * 4.5 / 0.5) ** 2))
lo, hi = [math.sqrt((n - 1) * s**2 / stats.chi2.ppf(q, n - 1)) for q in (0.975, 0.025)]
print(f"intervalo para σ con la χ²(9): [{lo:.2f}, {hi:.2f}]")
