import numpy as np
from scipy import stats
from sim import SIN_CURSO as a, CON_CURSO as b

def resumen(x, y, nota):
    w = stats.ttest_ind(x, y, equal_var=False); u = stats.mannwhitneyu(x, y)       # method="auto"
    print(f"{nota:34s} medias {x.mean():.2f} − {y.mean():.2f}   Welch p = {w.pvalue:.4f}   "
          f"Mann-Whitney U = {u.statistic:.1f}  p = {u.pvalue:.4f}  AUC = U/(n·m) = {u.statistic / (len(x) * len(y)):.3f}")

resumen(a, b, "el juguete")
b2 = b.copy(); b2[0] = 41.0                       # un camión con curso tiene un inyector roto ese mes
resumen(a, b2, "con un camión averiado (41,0)")
print("rangos del juguete (20 = el que más gasta):", stats.rankdata(np.r_[a, b]).tolist())
print("con valores repetidos no hay tabla exacta: aproximación normal con corrección por empates, p =",
      stats.mannwhitneyu(a, b, method="asymptotic").pvalue.round(4))
