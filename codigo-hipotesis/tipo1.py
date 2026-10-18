import numpy as np
from scipy import stats

rng = np.random.default_rng(3); R = 20_000
def tasa(n1, n2, s1, s2, metodo):
    """Fracción de pruebas con p < 0,05 cuando H0 es cierta (mismas medias)."""
    x, y = rng.normal(32, s1, (R, n1)), rng.normal(32, s2, (R, n2))
    if metodo == "mw": return np.mean(stats.mannwhitneyu(x, y, axis=1).pvalue < 0.05)
    return np.mean(stats.ttest_ind(x, y, axis=1, equal_var=(metodo == "student")).pvalue < 0.05)

print("error tipo I con α = 0,05 (20 000 pruebas A/A por fila)")
print("n1  n2   σ1   σ2   Student  Welch   Mann-Whitney")
for n1, n2, s1, s2 in [(10, 10, 1.6, 1.6), (10, 10, 1.0, 3.0), (5, 25, 3.0, 1.0), (25, 5, 3.0, 1.0), (50, 50, 1.6, 1.6)]:
    print(f"{n1:2d}  {n2:2d}  {s1:4.1f} {s2:4.1f}   {tasa(n1, n2, s1, s2, 'student'):.4f}   {tasa(n1, n2, s1, s2, 'welch'):.4f}  {tasa(n1, n2, s1, s2, 'mw'):.4f}")
p = stats.ttest_ind(rng.normal(32, 1.6, (R, 10)), rng.normal(32, 1.6, (R, 10)), axis=1).pvalue
print("bajo H0 el p-valor es uniforme: deciles", np.round(np.quantile(p, np.arange(0.1, 1, 0.1)), 3).tolist())
print("KS frente a U(0,1): p =", round(stats.kstest(p, "uniform").pvalue, 3), "  error de Monte Carlo de una tasa del 5 %: ±", round(1.96 * np.sqrt(0.05 * 0.95 / R), 4))
