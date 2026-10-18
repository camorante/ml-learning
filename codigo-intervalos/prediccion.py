import math
import numpy as np
from scipy import stats
from sim import DIEZ, MU, SIGMA, carga_consumo
from ic import ic_media, ic_prediccion

print("10 trayectos: confianza (la media)", np.round(ic_media(DIEZ), 2), "  predicción (el próximo trayecto)", np.round(ic_prediccion(DIEZ), 2))
r = np.random.default_rng(3); hc = hp = 0; R = 20000
for _ in range(R):
    x = r.normal(MU, SIGMA, 10); nuevo = r.normal(MU, SIGMA)
    lo, hi = ic_media(x); hc += lo <= nuevo <= hi
    lo, hi = ic_prediccion(x); hp += lo <= nuevo <= hi
print(f"¿atrapa el próximo trayecto? intervalo de confianza {hc / R:.1%}   intervalo de predicción {hp / R:.1%}")

# regresión consumo ~ carga: banda de confianza (la recta) y de predicción (un camión)
x, y = carga_consumo(); x, y = x[:60], y[:60]; n = len(x)               # los 60 primeros camiones
X = np.column_stack([np.ones(n), x]); beta, *_ = np.linalg.lstsq(X, y, rcond=None)
res = y - X @ beta; s = math.sqrt(res @ res / (n - 2)); t = stats.t.ppf(0.975, n - 2)
XtXi = np.linalg.inv(X.T @ X); Sxx = np.sum((x - x.mean())**2)
print(f"recta: consumo = {beta[0]:.3f} + {beta[1]:.4f}·carga   s = {s:.3f}   t(58) = {t:.4f}")
print("pendiente: IC", np.round(beta[1] + np.array([-1, 1]) * t * s * math.sqrt(XtXi[1, 1]), 4),
      " scipy.linregress:", np.round(beta[1] + np.array([-1, 1]) * t * stats.linregress(x, y).stderr, 4))
for x0 in (6, 15, 25):
    v = np.array([1, x0]); se_fit = s * math.sqrt(v @ XtXi @ v)
    se_f2 = s * math.sqrt(1 / n + (x0 - x.mean())**2 / Sxx)
    print(f"carga {x0:2d} t: ŷ = {v @ beta:.2f}  confianza ±{t * se_fit:.3f} (fórmula corta ±{t * se_f2:.3f})  predicción ±{t * math.sqrt(s**2 + se_fit**2):.3f}")
