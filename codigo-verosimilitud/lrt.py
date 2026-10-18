import math
import numpy as np
from scipy import stats
from sim import ALERTAS_A as x, alertas_conductores
from mv import ll_poisson, ll_negbin, ajustar_negbin, razon_verosimilitud, aic

# ¿da el camión A más alertas que la media de la flota (λ0 = 2)? Las tres pruebas clásicas
n, lh, l0 = len(x), x.mean(), 2.0
d, p = razon_verosimilitud(ll_poisson(l0, x), ll_poisson(lh, x))
zw = (lh - l0) / math.sqrt(lh / n); zs = (lh - l0) / math.sqrt(l0 / n)
print(f"razón de verosimilitudes: 2Δℓ = {d:.3f}  p = {p:.3f}")
print(f"Wald (ee en λ̂):  z = {zw:.3f}  p = {2 * stats.norm.sf(abs(zw)):.3f}")
print(f"puntuación (ee en λ0): z = {zs:.3f}  p = {2 * stats.norm.sf(abs(zs)):.3f}")

# 40 conductores × 30 días: ¿Poisson con un λ común o binomial negativa (λ que cambia)?
a, _ = alertas_conductores()
llp = ll_poisson(a.mean(), a)
r, mu, llnb = ajustar_negbin(a)
print(f"media {a.mean():.3f}  varianza {a.var(ddof=1):.3f}  (Poisson exige varianza = media)")
print(f"Poisson: ℓ = {llp:.2f}  AIC = {aic(llp, 1):.1f}")
print(f"binomial negativa: r = {r:.3f}  μ = {mu:.3f}  ℓ = {llnb:.2f}  AIC = {aic(llnb, 2):.1f}")
d = 2 * (llnb - llp)
print(f"2Δℓ = {d:.1f}  p (frontera: ½·χ²₁) = {0.5 * stats.chi2.sf(d, 1):.2e}")
