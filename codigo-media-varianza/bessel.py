import numpy as np

rng = np.random.default_rng(7)
pobl = 29 + rng.gamma(4.0, 1.0, 2000)               # los 2 000 viajes del año de un camión (L/100 km), sesgados
mu, s2 = pobl.mean(), pobl.var()                    # verdad: toda la población, dividiendo por N
print(f"población: media {mu:.3f}, varianza {s2:.3f}, desviación {np.sqrt(s2):.3f}")
print("  n   E[÷n]/σ²  E[÷(n−1)]/σ²  (n−1)/n   E[S]/σ   E[ΣΔ² con μ] / E[ΣΔ² con x̄]")
for n in (2, 3, 5, 10, 30):
    X = rng.choice(pobl, size=(100_000, n), replace=True)
    ss = ((X - X.mean(1, keepdims=True)) ** 2).sum(1)          # respecto a la media de la muestra
    ssmu = ((X - mu) ** 2).sum(1)                                # respecto a la media verdadera
    print(f"{n:3d}   {np.mean(ss / n) / s2:8.3f}  {np.mean(ss / (n - 1)) / s2:11.3f}  {(n - 1) / n:7.3f}"
          f"  {np.mean(np.sqrt(ss / (n - 1))) / np.sqrt(s2):7.3f}   {ssmu.mean() / ss.mean():8.3f}")
