import numpy as np
from outl import z_clasico, z_robusto, vallas_iqr, ols, mad
x = np.array([2, 6, 10, 14, 18, 12.]); y = np.array([24, 27, 31, 34, 38, 330.])
print(ols(x, y).round(2), ols(x[:5], y[:5]).round(3))                 # [40.26  3.91] [22.05  0.875]
print(round(z_clasico(y)[5], 2), round(5 / np.sqrt(6), 2))             # 2.04 2.04  (el tope)
print(np.median(y), mad(y), round(z_robusto(y)[5], 1))                 # 32.5 5.5 36.5
print(vallas_iqr(y))                                                   # (14.5, 50.5)
i, j = np.triu_indices(6, 1); b1 = np.median((y[j] - y[i]) / (x[j] - x[i])); b0 = np.median(y - b1 * x)
print(b1, b0)                                                          # 0.875 22.25
for b in (ols(x, y), ols(x[:5], y[:5]), (b0, b1)): print(round(b[0] + 20 * b[1], 2))   # 118.47 39.55 39.75
