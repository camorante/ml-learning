import math
import numpy as np
from scipy import stats
from sim import SIN_CURSO as a, CON_CURSO as b
from pruebas import permutacion, bonferroni, holm, bh

ma, mb = a.sum() / 10, b.sum() / 10
print("1. sumas", round(a.sum(), 1), round(b.sum(), 1), " medias", ma, round(mb, 2), " diferencia", round(ma - mb, 2))
da, db = a - ma, b - mb
print("2. d² sin curso", np.round(da**2, 2).tolist(), " Σ", round(np.sum(da**2), 2))
print("   d² con curso", np.round(db**2, 2).tolist(), " Σ", round(np.sum(db**2), 2))
va, vb = np.sum(da**2) / 9, np.sum(db**2) / 9
print("   s² =", round(va, 4), round(vb, 4), " s =", round(math.sqrt(va), 4), round(math.sqrt(vb), 4))
ee = math.sqrt(va / 10 + vb / 10); print("3. s²/10:", round(va / 10, 5), round(vb / 10, 5), " suma", round(va / 10 + vb / 10, 5), " ee", round(ee, 4))
t = (ma - mb) / ee; print("4. t =", round(t, 4))
gl = (va / 10 + vb / 10) ** 2 / ((va / 10) ** 2 / 9 + (vb / 10) ** 2 / 9)
print("5. numerador", round((va / 10 + vb / 10) ** 2, 6), " denominador", round((va / 10) ** 2 / 9 + (vb / 10) ** 2 / 9, 6), " gl", round(gl, 2))
p = 2 * stats.t.sf(t, gl); print("6. P(T > t) =", round(stats.t.sf(t, gl), 5), " p bilateral", round(p, 4), " con la normal", round(2 * stats.norm.sf(t), 4))
r = permutacion(a, b); print("7. C(20,10) =", math.comb(20, 10), " extremos", r["b"], " p", round(r["p"], 4))
tc = stats.t.ppf(0.975, gl); print("8. t crítico", round(tc, 4), " margen", round(tc * ee, 3), " IC [", round(ma - mb - tc * ee, 3), ",", round(ma - mb + tc * ee, 3), "]",
                                    " en €:", round((ma - mb) * 1650), round((ma - mb - tc * ee) * 1650), round((ma - mb + tc * ee) * 1650))
e0 = 1.6 * math.sqrt(2 / 10); lam = 1.0 / e0
print("9. ee con σ = 1,6:", round(e0, 4), " λ = 1/ee =", round(lam, 4), " λ − 1,96 =", round(lam - 1.96, 4), " potencia ≈ Φ(...) =", round(stats.norm.cdf(lam - 1.96), 4),
      " exacta", round(stats.nct.sf(stats.t.ppf(0.975, 18), 18, lam) + stats.nct.cdf(-stats.t.ppf(0.975, 18), 18, lam), 4))
print("   n = 2·(1,96 + 0,8416)²·1,6²/1² =", round(2 * (1.96 + 0.8416) ** 2 * 1.6**2, 2))
ps = np.array([0.004, 0.0322, 0.041])                     # frenazos, consumo, velocidad media
print("10. Bonferroni", np.round(bonferroni(ps), 4).tolist(), " Holm", np.round(holm(ps), 4).tolist(), " BH", np.round(bh(ps), 4).tolist())
