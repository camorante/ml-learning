"""Incertidumbre en ML: la nota de un modelo con bootstrap por camiones, y la dispersión de un bosque."""
import numpy as np
from sklearn.ensemble import RandomForestRegressor

def flota(k, m, r):
    """k camiones × m trayectos: carga (t), velocidad (km/h), efecto del camión y consumo (L/100 km)."""
    carga = r.uniform(4, 26, (k, m)); vel = r.uniform(30, 90, (k, m)); cam = r.normal(0, 2.0, (k, 1))
    y = 24 + 0.5 * carga + 0.004 * (vel - 55)**2 + cam + r.normal(0, 2.0, (k, m))
    return np.column_stack([carga.ravel(), vel.ravel()]), y.ravel(), np.repeat(np.arange(k), m)

r = np.random.default_rng(0)
Xtr, ytr, _ = flota(60, 30, r)
rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=5, random_state=0).fit(Xtr, ytr)
Xbig, ybig, _ = flota(4000, 30, r); mae_real = np.mean(np.abs(rf.predict(Xbig) - ybig))
print(f"MAE «real» (4 000 camiones nuevos): {mae_real:.3f} L/100 km")

R, B = 300, 999; h_tr = h_cam = 0
for i in range(R):
    X, y, g = flota(20, 30, r); e = np.abs(rf.predict(X) - y)                 # conjunto de prueba: 20 camiones
    iv_t = np.quantile(e[r.integers(0, 600, (B, 600))].mean(1), [0.025, 0.975])
    ec = e.reshape(20, 30).mean(1); iv_c = np.quantile(ec[r.integers(0, 20, (B, 20))].mean(1), [0.025, 0.975])
    h_tr += iv_t[0] <= mae_real <= iv_t[1]; h_cam += iv_c[0] <= mae_real <= iv_c[1]
    if i == 0: print(f"una prueba: MAE {e.mean():.3f}  bootstrap por trayectos [{iv_t[0]:.3f}, {iv_t[1]:.3f}]  por camiones [{iv_c[0]:.3f}, {iv_c[1]:.3f}]")
print(f"cobertura del MAE real en {R} conjuntos de prueba: por trayectos {h_tr / R:.1%}  por camiones {h_cam / R:.1%}")

# ¿es la dispersión de los árboles un intervalo de predicción? No: mide la incertidumbre del modelo, no el ruido
Xte, yte, _ = flota(200, 30, r)
P = np.stack([t.predict(Xte) for t in rf.estimators_]); mu, sd = P.mean(0), P.std(0)
print(f"media ± 1,96·(desviación entre árboles) atrapa el consumo real: {np.mean(np.abs(yte - mu) <= 1.96 * sd):.1%}")
Xc, yc, _ = flota(40, 30, r); q = np.quantile(np.abs(rf.predict(Xc) - yc), 0.95)  # conformal partido
print(f"conformal partido (cuantil 95 % de |residuo| en calibración = ±{q:.2f}): atrapa {np.mean(np.abs(yte - mu) <= q):.1%}")
