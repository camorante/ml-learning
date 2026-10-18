import numpy as np
from scipy import stats
from sim import euros_anio

rng = np.random.default_rng(21)
a, b = rng.normal(32, 1.6, 5000), rng.normal(31.9, 1.6, 5000)     # toda la flota: ahorro real 0,1 L/100 km
r = stats.ttest_ind(a, b, equal_var=False); ci = r.confidence_interval()
print(f"5 000 camiones por grupo: ahorro {a.mean() - b.mean():.3f}  p = {r.pvalue:.5f}  IC 95 % [{ci.low:.3f}, {ci.high:.3f}]")
print(f"en euros por camión y año: {euros_anio(a.mean() - b.mean()):.0f} € (IC {euros_anio(ci.low):.0f} a {euros_anio(ci.high):.0f} €)  frente a 300 € del curso")
print(f"d de Cohen = {(a.mean() - b.mean()) / 1.6:.3f}   ahorro que paga el curso: 300 / {euros_anio(1):.0f} = {300 / euros_anio(1):.3f} L/100 km")
