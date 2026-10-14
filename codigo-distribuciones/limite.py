import numpy as np
from scipy import stats

lam = 2.0; k = np.arange(0, 60)
po = stats.poisson.pmf(k, lam)
for n in (5, 10, 20, 50, 200, 1000):
    bi = stats.binom.pmf(k, n, lam / n)
    tv = 0.5 * np.abs(bi - po).sum()                 # distancia de variación total
    print(f"n = {n:4d}  p = {lam / n:.4f}  P(0): binomial {bi[0]:.4f}  Poisson {po[0]:.4f}   d_TV = {tv:.4f}   cota n·p² = {n * (lam / n) ** 2:.4f}")
