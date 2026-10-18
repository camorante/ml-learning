import math
import numpy as np
from scipy import stats
from sim import SIGMA, EFECTO

def pot_normal(d, s, n, a=0.05):
    l = d / (s * math.sqrt(2 / n)); z = stats.norm.ppf(1 - a / 2); return stats.norm.sf(z - l) + stats.norm.cdf(-z - l)
def pot_exacta(d, s, n, a=0.05):                   # t no central con gl = 2n − 2 y λ = δ / ee
    gl = 2 * n - 2; l = d / (s * math.sqrt(2 / n)); tc = stats.t.ppf(1 - a / 2, gl); return stats.nct.sf(tc, gl, l) + stats.nct.cdf(-tc, gl, l)

rng = np.random.default_rng(11); R = 20_000
print(f"ahorro real δ = {EFECTO} L/100 km, σ = {SIGMA}, α = 0,05 bilateral")
print(" n/grupo  normal  exacta(nct)  simulada")
for n in (5, 10, 20, 41, 80):
    a, b = rng.normal(32, SIGMA, (R, n)), rng.normal(32 - EFECTO, SIGMA, (R, n))
    sim = np.mean(stats.ttest_ind(a, b, axis=1).pvalue < 0.05)
    print(f"{n:6d}   {pot_normal(EFECTO, SIGMA, n):.4f}   {pot_exacta(EFECTO, SIGMA, n):.4f}      {sim:.4f}")
z = stats.norm.ppf(0.975) + stats.norm.ppf(0.80)
print(f"n para 80 %: 2·(z_0,975 + z_0,80)²·σ²/δ² = 2·{z:.4f}²·{SIGMA**2:.2f}/1 = {2 * z**2 * SIGMA**2:.2f} → 41")
n = next(n for n in range(2, 200) if pot_exacta(EFECTO, SIGMA, n) >= 0.8); print("con la t exacta:", n, f"(potencia {pot_exacta(EFECTO, SIGMA, n):.4f})")
for d in (0.5, 2.0): print(f"para δ = {d}: n = {math.ceil(2 * z**2 * SIGMA**2 / d**2)} por grupo")
mde = z * SIGMA * math.sqrt(2 / 10); print(f"efecto mínimo detectable con 10 por grupo (80 %): {mde:.2f} L/100 km")
# la maldición del ganador: con potencia baja, las pruebas «significativas» exageran el efecto
for n in (10, 41):
    a, b = rng.normal(32, SIGMA, (R, n)), rng.normal(32 - EFECTO, SIGMA, (R, n)); d = a.mean(1) - b.mean(1)
    sig = stats.ttest_ind(a, b, axis=1).pvalue < 0.05
    print(f"n = {n}: ahorro medio de las significativas = {d[sig].mean():.2f} (real {EFECTO}); con signo equivocado: {np.mean(d[sig] < 0):.2%}")
