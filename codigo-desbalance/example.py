import numpy as np
from imb import metricas, umbral_coste, corregir_prior
m = metricas(vp=13, fp=78, fn=10, vn=899)
print({k: round(float(v), 3) for k, v in m.items()})
# {'acierto': 0.912, 'precision': 0.143, 'exhaustividad': 0.565, 'especificidad': 0.92, 'f': 0.228, 'acierto_equilibrado': 0.743, "mcc": 0.253}
print(10 * 4000 + 78 * 150, 23 * 4000, round(umbral_coste(150, 4000), 4))      # 51700 92000 0.0361
print(round(float(corregir_prior(0.40, 977 / 23)), 4))                            # 0.0155
z = 1.96; n = 23; p = 13 / 23; c = (p + z**2 / (2 * n)) / (1 + z**2 / n); h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
print(round(c - h, 3), round(c + h, 3))   # 0.368 0.744  (intervalo de Wilson del 95 % de la exhaustividad)
