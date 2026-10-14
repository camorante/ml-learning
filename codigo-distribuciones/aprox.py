import numpy as np
from scipy import stats

for n, p, k in [(20, 0.10, 6), (300, 0.10, 40), (200, 0.30, 70)]:
    mu, sd = n * p, np.sqrt(n * p * (1 - p))
    exacta = stats.binom.sf(k - 1, n, p)
    normal = stats.norm.sf((k - 0.5 - mu) / sd)               # corrección de continuidad: P(X ≥ k) ≈ P(Y > k − 0.5)
    sin_corr = stats.norm.sf((k - mu) / sd)
    poisson = stats.poisson.sf(k - 1, mu)
    print(f"n={n:3d} p={p:.2f} P(X ≥ {k:2d})  exacta {exacta:.5f}  normal+corr. {normal:.5f}  normal sin corr. {sin_corr:.5f}"
          f"  Poisson {poisson:.5f}   n·p·(1−p) = {n * p * (1 - p):.1f}")
