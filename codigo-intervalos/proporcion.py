import numpy as np
from scipy import stats
from ic import ic_wald, ic_wilson, ic_clopper_pearson

def agresti_coull(k, n, z=1.959964):
    nt = n + z * z; pt = (k + z * z / 2) / nt; h = z * np.sqrt(pt * (1 - pt) / nt)
    return pt - h, pt + h

M = {"Wald": ic_wald, "Wilson": ic_wilson, "Agresti-Coull": agresti_coull, "Clopper-Pearson": ic_clopper_pearson}
for k, n in [(1, 40), (0, 40), (12, 40)]:
    print(f"{k} de {n}: " + "  ".join(f"{m} [{f(k, n)[0]:.4f}, {f(k, n)[1]:.4f}]" for m, f in M.items()))
print("scipy Wilson 1/40:", np.round(stats.binomtest(1, 40).proportion_ci(method="wilson"), 4), " exacto:", np.round(stats.binomtest(1, 40).proportion_ci(method="exact"), 4))

def cobertura(n, p, f, _cache={}):
    """Suma exacta: probabilidad de los k cuyo intervalo contiene p (los intervalos de cada k, calculados una vez)."""
    if (n, f) not in _cache:
        _cache[n, f] = np.array([f(k, n) for k in range(n + 1)])
    iv = _cache[n, f]; pk = stats.binom.pmf(np.arange(n + 1), n, p)
    return float(np.sum(pk[(iv[:, 0] <= p) & (p <= iv[:, 1])]))
print("cobertura exacta (n = 40):")
for p in (0.01, 0.02, 0.05, 0.10, 0.20, 0.50):
    print(f"  p = {p:.2f}  " + "  ".join(f"{m} {cobertura(40, p, f):.3f}" for m, f in M.items()))
ps = np.linspace(0.005, 0.5, 100)
for n in (40, 200):
    print(f"n = {n}: cobertura media en p ∈ [0,005, 0,5] y mínima  " + "  ".join(
        f"{m} {np.mean([cobertura(n, p, f) for p in ps]):.3f}/{min(cobertura(n, p, f) for p in ps):.3f}" for m, f in M.items()))
print("regla del 3 con 0 de 40:", 3 / 40, "  Clopper-Pearson unilateral 95 %:", round(1 - 0.05 ** (1 / 40), 4))
