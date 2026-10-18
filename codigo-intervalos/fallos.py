"""Cobertura real de intervalos del 95 % cuando el bootstrap ingenuo falla."""
import math
import numpy as np
from scipy import stats
from sim import MU
from ic import bootstrap_grupos, bootstrap_bloques, ic_media

R, B = 1000, 999
q = lambda v: np.quantile(v, [0.025, 0.975])
dentro = lambda iv, mu: iv[0] <= mu <= iv[1]

# 1) 8 camiones × 15 trayectos con un efecto por camión
r = np.random.default_rng(100); h = {"trayectos sueltos": 0, "camiones enteros": 0, "t sobre las 8 medias": 0, "t ingenuo (120 datos)": 0}
for _ in range(R):
    x = (MU + r.normal(0, 2.5, 8))[:, None] + r.normal(0, 3.0, (8, 15)); flat = x.ravel(); mc = x.mean(1)
    h["trayectos sueltos"] += dentro(q(flat[r.integers(0, 120, (B, 120))].mean(1)), MU)
    h["camiones enteros"] += dentro(q(mc[r.integers(0, 8, (B, 8))].mean(1)), MU)
    h["t sobre las 8 medias"] += dentro(ic_media(mc), MU)
    h["t ingenuo (120 datos)"] += dentro(ic_media(flat), MU)
print("camiones:", "  ".join(f"{k} {v / R:.1%}" for k, v in h.items()))
icc = 2.5**2 / (2.5**2 + 3.0**2); print(f"  correlación intraclase {icc:.3f}; efecto de diseño 1 + (15 − 1)·ICC = {1 + 14 * icc:.2f}; n efectivo = 120 / {1 + 14 * icc:.2f} = {120 / (1 + 14 * icc):.1f}")
x = (MU + np.random.default_rng(5).normal(0, 2.5, 8))[:, None] + np.random.default_rng(6).normal(0, 3.0, (8, 15))
print("  bootstrap_grupos (ic.py) en una flota:", np.round(bootstrap_grupos(x.ravel(), np.repeat(np.arange(8), 15), np.mean, seed=1), 2))

# 2) serie diaria con memoria: AR(1) con φ = 0,7, 120 días
def ar1(r, n=120, phi=0.7):
    e = r.normal(0, 1.5, n + 50); y = np.zeros(n + 50)
    for t in range(1, n + 50): y[t] = phi * y[t - 1] + e[t]
    return MU + y[50:]
r = np.random.default_rng(200); h = {"bootstrap simple": 0, "bloques L = 5": 0, "bloques L = 15": 0}
for _ in range(R):
    y = ar1(r); s = int(r.integers(1 << 30))
    h["bootstrap simple"] += dentro(q(y[r.integers(0, 120, (B, 120))].mean(1)), MU)
    h["bloques L = 5"] += dentro(bootstrap_bloques(y, np.mean, 5, B=B, seed=s), MU)
    h["bloques L = 15"] += dentro(bootstrap_bloques(y, np.mean, 15, B=B, seed=s), MU)
print("serie AR(1):", "  ".join(f"{k} {v / R:.1%}" for k, v in h.items()), f"  (varianza de la media inflada por (1 + φ)/(1 − φ) = {1.7 / 0.3:.2f})")

# 3) el máximo de una uniforme(78, 90): el percentil nunca llega a 90
r = np.random.default_rng(101); hit = 0
for _ in range(R):
    s = r.uniform(78, 90, 20); hit += dentro(q(s[r.integers(0, 20, (B, 20))].max(1)), 90)
print(f"máximo: cobertura {hit / R:.1%};  P(un remuestreo contiene el máximo) = 1 − (19/20)^20 = {1 - (19 / 20) ** 20:.3f}")
