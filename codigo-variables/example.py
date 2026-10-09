import numpy as np
from prep import ciclo, media_pasada
A, B, C = np.array([20, 4000]), np.array([180, 4100]), np.array([25, 9000])
s = np.array([60, 3000])                                   # desviaciones típicas de la flota
print(np.linalg.norm(A - B).round(1), np.linalg.norm(A - C).round(1))                 # 188.7 5000.0
print(np.linalg.norm((A - B) / s).round(3), np.linalg.norm((A - C) / s).round(3))     # 2.667 1.669
g, m = 25.0, 10
print(round((1 * 8.1 + m * g) / (1 + m), 2), round((185 * 22.68 + m * g) / (185 + m), 2))  # 23.46 22.8
h = ciclo([23, 1, 11], 24); print(h.round(4), np.linalg.norm(h[0] - h[1]).round(4), np.linalg.norm(h[0] - h[2]).round(4))
dias = [3, 10, 18, 25, 33]; cons = [28, 30, 27, 31, 29]
print(media_pasada(["C"] * 5, dias, cons, 30).round(2))     # [nan 28 29 28.33 29.33]
d = np.array(dias); c = np.array(cons); print(c[np.abs(d - 25) <= 15].mean())        # 29.25
