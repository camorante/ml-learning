"""Genera ../embed.json (datos de la versión sencilla) e imprime las cifras que se citan en el texto."""
import json, math
import numpy as np
from scipy import stats
from sim import *
from mv import *

out = {}
A, B = ALERTAS_A, ALERTAS_B
out["A"], out["B"] = A.tolist(), B.tolist()
for nom, x in (("A", A), ("B", B)):
    lh = mle_poisson(x); ll = ll_poisson(lh, x)
    print(f"juguete {nom}: {x.tolist()} suma {x.sum()} λ̂ = {lh:.4f}  ℓ(λ̂) = {ll:.4f}  producto = {math.exp(ll):.4e} = 10^{ll / math.log(10):.3f}")
    for lam in (0.5, 1, 1.5, 2, 2.5, 3, 4, 5):
        l = ll_poisson(lam, x)
        print(f"   λ = {lam}: ℓ = {l:.3f}  producto {math.exp(l):.3e}  el mejor es {math.exp(ll - l):.1f} veces más probable")
print("A: probabilidad de cada día con λ = 2,5:", [round(float(stats.poisson.pmf(k, 2.5)), 3) for k in A])
print("A: probabilidad de cada día con λ = 1:", [round(float(stats.poisson.pmf(k, 1)), 4) for k in A])
print("A: varianza de los 14 días (n−1):", A.var(ddof=1), " (Poisson: varianza = media)")

# idea 1: producto de muchas probabilidades
L = alertas_largo(); out["largo"] = L.tolist()
for n in (1, 2, 5, 14, 30, 60, 100, 200, 365, 500, 1000):
    x = L[:n]; lh = x.mean(); ll = ll_poisson(lh, x)
    prod = np.prod(stats.poisson.pmf(x, lh))
    print(f"idea 1: {n:4d} días λ̂ = {lh:.3f}  log10 producto = {ll / math.log(10):9.2f}  producto directo = {prod:.3e}")

# idea 3: misma proporción (10 %), más datos
for n in (10, 40, 200, 1000):
    k = n // 10; ll = lambda p: ll_binom(p, k, n)
    lo, hi = intervalo_perfil(ll, k / n, 1e-9, 1 - 1e-9)
    print(f"idea 3: {k}/{n}: intervalo de verosimilitud 95 % [{lo:.4f}, {hi:.4f}]  error estándar {se_binom(k / n, n):.4f}  Wald [{k / n - 1.96 * se_binom(k / n, n):.4f}, {k / n + 1.96 * se_binom(k / n, n):.4f}]")
print("   umbral de verosimilitud relativa: exp(−3,841/2) =", math.exp(-stats.chi2.ppf(0.95, 1) / 2))

# idea 4: mínimos cuadrados
c, y = consumo_carga(); out["carga"], out["cons"] = c.tolist(), y.tolist()
b, a = np.polyfit(c, y, 1); ss = float(np.sum((y - a - b * c) ** 2))
print(f"idea 4: recta MC: consumo = {a:.4f} + {b:.4f}·carga  suma de cuadrados {ss:.3f}  ℓ (σ = 2) = {-12 * math.log(2 * math.sqrt(2 * math.pi)) - ss / 8:.3f}")
for bb in (0.0, 0.3, 0.5, 0.7, 1.0):
    aa = y.mean() - bb * c.mean(); s2 = float(np.sum((y - aa - bb * c) ** 2))
    print(f"   pendiente {bb}: SC {s2:.2f}  ℓ {-12 * math.log(2 * math.sqrt(2 * math.pi)) - s2 / 8:.3f}")
print("   σ̂ por MV de los residuos:", math.sqrt(ss / 12), " con n − 2:", math.sqrt(ss / 10))

# idea 5: descarga
x = DESCARGA; out["desc"] = x.tolist()
print(f"idea 5: media {x.mean():.3f}  mediana {np.median(x)}  sin el 95: media {x[:-1].mean():.3f} mediana {np.median(x[:-1])}")
print(f"   normal: suma de cuadrados en la media {np.sum((x - x.mean()) ** 2):.2f}, en la mediana {np.sum((x - 20) ** 2):.2f}")
print(f"   Laplace: suma de |·| en la mediana {np.sum(np.abs(x - 20)):.2f}, en la media {np.sum(np.abs(x - x.mean())):.2f}")

# idea 6: logística
km, t = entregas(); out["km"], out["tarde"] = km.tolist(), t.tolist()
w = logistica((km[:, None] - 150) / 100, t)
incl = w[1]; medio = 150 - w[0] / w[1] * 100
z = incl * (km - medio) / 100; p = 1 / (1 + np.exp(-z))
ll = ll_bernoulli(p, t)
print(f"idea 6: {t.sum()} tarde de {t.size}  MV: punto medio {medio:.3f} km  inclinación {incl:.4f}  ℓ = {ll:.4f}  log loss = {-ll / t.size:.4f}")
out["lg"] = [round(medio, 3), round(incl, 4)]
pen = -(t * np.log(p) + (1 - t) * np.log(1 - p)); j = int(np.argmax(pen))
print(f"   peor explicada: km {km[j]} tarde {t[j]} p(tarde) {p[j]:.3f} castigo {pen[j]:.3f}")
from sklearn.metrics import log_loss
print("   sklearn log_loss:", log_loss(t, p), "  modelo 'siempre 50 %':", math.log(2), " modelo 'tasa media':", -ll_bernoulli(np.full(t.size, t.mean()), t) / t.size)
for (m_, b_) in ((170, 2.4), (100, 2), (250, 2), (163, 8), (163, 0.5)):
    pp = 1 / (1 + np.exp(-b_ * (km - m_) / 100)); print(f"   medio {m_} incl {b_}: log loss {-ll_bernoulli(pp, t) / t.size:.4f}")

# idea 7: pocos datos y prior
for k, n in ((0, 5), (1, 5), (0, 40), (3, 40), (16, 200)):
    print(f"idea 7: {k}/{n}: MV {k / n:.4f}  " + "  ".join(f"m={m}: {map_binom(k, n, m, 0.08):.4f}" for m in (0, 5, 10, 20, 50)))

# idea 8: subir la colina en (punto medio/100, inclinación) con ℓ media
def grad(u, bb):
    z = bb * (km / 100 - u); pp = 1 / (1 + np.exp(-z)); r = t - pp
    return float(np.mean(r * -bb)), float(np.mean(r * (km / 100 - u))), float(np.mean(t * np.log(pp) + (1 - t) * np.log(1 - pp)))
for paso in (0.5, 2.0, 8.0, 14.0):
    u, bb = 0.6, 0.6; hist = []
    for it in range(200):
        gu, gb, l = grad(u, bb); hist.append(l); u += paso * gu; bb += paso * gb
    print(f"idea 8: paso {paso}: tras 10 {hist[10]:.4f}, 30 {hist[30]:.4f}, 200 {hist[-1]:.4f}  final medio {u * 100:.1f} incl {bb:.3f}")

json.dump(out, open("../embed.json", "w"), separators=(",", ":"))
print("embed.json:", len(json.dumps(out)), "bytes")
