import numpy as np
from sklearn.decomposition import PCA
from autovectores import estacionaria, tiempo_mezcla, pca

rng = np.random.default_rng(5)
ESTADOS = ["en ruta", "en taller", "parado"]
P = np.array([[0.90, 0.03, 0.07],
              [0.30, 0.60, 0.10],
              [0.50, 0.02, 0.48]])                 # la tabla «de verdad», desconocida para el analista

# 1) simular 200 camiones durante 180 días y estimar la tabla contando cambios
n_cam, n_dias = 200, 180
est = np.zeros((n_cam, n_dias), int); est[:, 0] = rng.choice(3, n_cam)
for t in range(1, n_dias):
    u = rng.random(n_cam)[:, None]
    est[:, t] = (u > np.cumsum(P[est[:, t - 1]], axis=1)).sum(1)
C = np.zeros((3, 3))
np.add.at(C, (est[:, :-1].ravel(), est[:, 1:].ravel()), 1)
Ph = C / C.sum(1, keepdims=True)
print("transiciones contadas:", int(C.sum()))
print("P estimada:\n", Ph.round(3))
pi_h = estacionaria(Ph); pi = estacionaria(P, "lineal")
ocup = np.bincount(est[:, 30:].ravel(), minlength=3) / est[:, 30:].size
print("estacionaria estimada  ", pi_h.round(4), "  de verdad", pi.round(4), "  ocupación observada (días 30+)", ocup.round(4))
w = np.sort(np.abs(np.linalg.eigvals(Ph.T)))[::-1]
print("|autovalores| de P estimada", w.round(4), "  semivida de una desviación:", round(np.log(2) / -np.log(w[1]), 2), "días")
print("tiempo de mezcla (TV < 1 %):", tiempo_mezcla(Ph), "días   (con la P de verdad:", tiempo_mezcla(P), ")")
print("días esperados en taller por visita, 1/(1 − P̂[1, 1]):", round(1 / (1 - Ph[1, 1]), 2), " retorno medio al taller 1/π:", round(1 / pi_h[1], 1), "días")

# 2) ¿qué pasa si el taller tarda más? (la diagonal 0.60 → 0.80)
P2 = P.copy(); P2[1] = [0.15, 0.80, 0.05]
pi2 = estacionaria(P2, "lineal"); w2 = np.sort(np.abs(np.linalg.eigvals(P2.T)))[::-1]
print("taller lento: estacionaria", pi2.round(4), " camiones en ruta de 200:", round(200 * pi2[0], 1),
      " |λ₂| =", round(w2[1], 4), " mezcla:", tiempo_mezcla(P2), "días")

# 3) PCA de telemetría: 600 viajes, 5 señales que dependen de 2 causas (carga y tipo de vía)
n = 600; carga = rng.uniform(0, 1, n); via = rng.uniform(0, 1, n)        # 0 = ciudad, 1 = autopista
X = np.c_[45 + 40 * via + rng.normal(0, 4, n),                           # velocidad media (km/h)
          1500 - 300 * via + 250 * carga + rng.normal(0, 60, n),         # rpm medias
          24 + 12 * carga - 4 * via + rng.normal(0, 1.2, n),             # consumo (L/100 km)
          88 + 6 * carga + rng.normal(0, 1.0, n),                        # temperatura del motor (°C)
          95 + 7 * carga + 0.5 * rng.normal(0, 1.0, n) + rng.normal(0, 1.0, n)]   # temperatura del aceite (°C)
Z = (X - X.mean(0)) / X.std(0, ddof=1)                                   # sin estandarizar mandarían las rpm
comp, var, _ = pca(Z); sk = PCA().fit(Z)
print("varianza explicada (desde cero):", (var / var.sum()).round(4))
print("varianza explicada (sklearn):   ", sk.explained_variance_ratio_.round(4))
print("componente 1:", comp[0].round(3), "\ncomponente 2:", comp[1].round(3))
print("correlación de la componente 1 con la carga:", round(abs(np.corrcoef(Z @ comp[0], carga)[0, 1]), 3),
      "  de la 2 con el tipo de vía:", round(abs(np.corrcoef(Z @ comp[1], via)[0, 1]), 3))
