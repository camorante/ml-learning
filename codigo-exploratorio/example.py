import numpy as np
from eda import resumen, pearson, spearman, vif
vel = np.array([50, 60, 70, 80, 90, 255.]); cons = np.array([20, 22, 23, 26, 29, 21.])
r = resumen(vel)
print(round(r["media"], 1), r["mediana"], r["mad"], round(r["desv"], 1))          # 100.8 75.0 15.0 76.8
print(round((255 - r["media"]) / r["desv"], 1), round((255 - r["mediana"]) / r["mad_normal"], 1))   # 2.0 8.1
print(round(pearson(vel, cons), 3), round(pearson(vel[:5], cons[:5]), 3))       # -0.186 0.984
print(round(spearman(vel, cons), 3), round(spearman(vel[:5], cons[:5]), 3))     # 0.429 1.0
rng = np.random.default_rng(0); a = rng.normal(size=5000); b = 0.95 * a + np.sqrt(1 - 0.95**2) * rng.normal(size=5000)
print(np.round(vif(np.column_stack([a, b])), 1), round(1 / (1 - 0.95**2), 1))  # [10.2 10.2] 10.3  (la r de la muestra no es exactamente 0.95)
