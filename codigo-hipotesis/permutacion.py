import numpy as np
from scipy import stats
from sim import SIN_CURSO as a, CON_CURSO as b
from pruebas import permutacion

print("medias:", a.mean().round(2), b.mean().round(2), " ahorro observado:", round(a.mean() - b.mean(), 2), "L/100 km")
ex = permutacion(a, b)                                   # las C(20,10) = 184 756 particiones
print(f"exacto: {ex['b']} de {ex['total']} repartos dan |dif| ≥ 1,6  →  p = {ex['p']:.4f}")
print(f"unilateral (solo ahorros ≥ 1,6): p = {permutacion(a, b, alternativa='mayor')['p']:.4f}")
for m in (100, 1_000, 10_000, 100_000):
    mc = permutacion(a, b, m=m, seed=1)
    ee = np.sqrt(ex["p"] * (1 - ex["p"]) / m)            # error de Monte Carlo del p-valor
    print(f"Monte Carlo, {m:>6d} barajadas: b = {mc['b']:>5}, p = (b+1)/(m+1) = {mc['p']:.4f}  ± {ee:.4f}")
r = stats.permutation_test((a, b), lambda u, v: np.mean(u) - np.mean(v), permutation_type="independent",
                           n_resamples=9_999, alternative="two-sided", rng=np.random.default_rng(1))
print("scipy.stats.permutation_test, 9 999 barajadas:", round(r.pvalue, 4))
