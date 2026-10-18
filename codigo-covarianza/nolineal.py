import numpy as np
from cor import pearson, spearman, kendall_tau_b, covarianza
from sim import velocidad

x = [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5]                     # cuarteto de Anscombe (1973)
A = {"I": (x, [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68]),
     "II": (x, [9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74]),
     "III": (x, [7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73]),
     "IV": ([8, 8, 8, 8, 8, 8, 8, 19, 8, 8, 8], [6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 12.50, 5.56, 7.91, 6.89])}
for k, (a, b) in A.items():
    print(f"Anscombe {k:3s} medias {np.mean(a):.2f} {np.mean(b):.2f}  var {np.var(a, ddof=1):.2f} {np.var(b, ddof=1):.3f}"
          f"  r {pearson(a, b):.3f}  Spearman {spearman(a, b):.3f}  Kendall {kendall_tau_b(a, b):.3f}")
v, c = velocidad()
print(f"velocidad y consumo (U): r {pearson(v, c):+.3f}  Spearman {spearman(v, c):+.3f}  r(|v − 50|, c) {pearson(np.abs(v - 50), c):+.3f}")
u = np.random.default_rng(4).uniform(-1, 1, 100_000)
print(f"X ~ U(−1, 1), Y = X²: cov {covarianza(u, u**2):+.4f}  r {pearson(u, u**2):+.4f}  (Y depende por completo de X)")
