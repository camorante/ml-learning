import numpy as np
from scipy import stats
from sim import ralenti

A, B = ralenti()                     # 2 000 trayectos sin aviso y 2 000 con aviso de ralentí en la app
k1, n1, k2, n2 = A.sum(), len(A), B.sum(), len(B)
print(f"sin aviso: {k1}/{n1} = {k1 / n1:.2%}   con aviso: {k2}/{n2} = {k2 / n2:.2%}")
p = (k1 + k2) / (n1 + n2); z = (k1 / n1 - k2 / n2) / np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
print(f"test z de dos proporciones: z = {z:.4f}  p = {2 * stats.norm.sf(abs(z)):.4f}")
T = [[k1, n1 - k1], [k2, n2 - k2]]
r = stats.chi2_contingency(T, correction=False)
print(f"χ² de independencia (tema 9): χ² = {r.statistic:.4f} (= z² = {z * z:.4f})  p = {r.pvalue:.4f}")
print("frecuencias esperadas si el aviso no cambia nada:", np.round(r.expected_freq, 1).tolist())
print(f"con corrección de Yates: p = {stats.chi2_contingency(T).pvalue:.4f}   Fisher exacto: p = {stats.fisher_exact(T).pvalue:.4f}")
d = k1 / n1 - k2 / n2; ee = np.sqrt(k1 / n1 * (1 - k1 / n1) / n1 + k2 / n2 * (1 - k2 / n2) / n2)
print(f"diferencia {d:.2%}  IC 95 % (Wald): [{d - 1.96 * ee:.2%}, {d + 1.96 * ee:.2%}]")
# con pocos datos, mejor el exacto: 30 trayectos por grupo con 9 y 3 casos de ralentí largo
T2 = [[9, 21], [3, 27]]
print(f"30 por grupo, 9 frente a 3: χ² p = {stats.chi2_contingency(T2, correction=False).pvalue:.4f}  Fisher p = {stats.fisher_exact(T2).pvalue:.4f}"
      f"  esperados mínimos = {stats.chi2_contingency(T2).expected_freq.min():.1f}")
