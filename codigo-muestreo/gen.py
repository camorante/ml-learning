"""Genera ../embed.json (la población de 12 000 paradas que usa la versión sencilla) y las cifras citadas."""
import json
import numpy as np
from scipy import stats
from sim import poblacion, RUTAS
from muestreo import n_efectivo_ar1, asignar

d, r, c = poblacion()
N = len(d); mu, sig = d.mean(), d.std()          # parámetros: desviación típica de la población (entre N)
json.dump(dict(dur=d.astype(int).tolist()), open("../embed.json", "w"), separators=(",", ":"))
print(f"N = {N}  μ = {mu:.2f}  σ = {sig:.2f}  mediana = {np.median(d):.0f}  máx = {d.max():.0f}  asimetría = {stats.skew(d):.2f}")
print(f"paradas > 60 min: {np.mean(d > 60):.1%}   > 200 min: {(d > 200).sum()}")
for h in range(3):
    x = d[r == h]; print(f"  {RUTAS[h]:8s} N_h = {len(x)}  media = {x.mean():.2f}  σ_h = {x.std():.2f}")
for n in [1, 4, 16, 30, 64, 100, 120, 400]:
    print(f"n = {n:3d}  σ/√n = {sig/np.sqrt(n):6.2f}  con fpc {sig/np.sqrt(n)*np.sqrt((N-n)/(N-1)):6.2f}")
rng = np.random.default_rng(0)
for n in [1, 4, 16, 64]:
    M = np.array([d[rng.integers(0, N, n)].mean() for _ in range(20000)])
    print(f"simulación n = {n:2d}: media de medias {M.mean():.2f}  desv. {M.std():.2f}  asimetría {stats.skew(M):.2f}")
# sesgo: solo se registran bien las paradas cortas
p = np.where(d <= 30, 1.0, np.exp(-(d - 30) / 40))
print(f"sesgo: media de las que se registran = {np.sum(p*d)/np.sum(p):.2f}  (se registra el {p.mean():.0%})")
# dependencia: 30 paradas seguidas de un mismo camión
M = []
for _ in range(20000):
    t = rng.integers(0, 60); s = rng.integers(0, 200 - 30); M.append(d[t*200+s:t*200+s+30].mean())
print(f"30 seguidas del mismo camión: desv. de la media {np.std(M):.2f} frente a σ/√30 = {sig/np.sqrt(30):.2f}")
lg = np.log(d)
ac = np.mean([np.corrcoef(d[t*200:t*200+199], d[t*200+1:t*200+200])[0, 1] for t in range(60)])
print(f"autocorrelación media de retardo 1 dentro de cada camión (minutos): {ac:.2f};  n_ef de 30 con φ = 0,6: {n_efectivo_ar1(30, 0.6):.1f}")
# diseños con n = 60
Nh = np.bincount(r); Sh = np.array([d[r == h].std() for h in range(3)])
for name, nh in [("proporcional", asignar(60, Nh)), ("Neyman", asignar(60, Nh, Sh))]:
    print(f"{name}: n_h = {nh.tolist()}")
def est(design):
    if design == 0:
        return d[rng.choice(N, 60, replace=False)].mean()
    if design == 1:
        nh = asignar(60, Nh); W = Nh / N
        return sum(W[h] * d[r == h][rng.choice(Nh[h], nh[h], replace=False)].mean() for h in range(3))
    t = rng.choice(60, 3, replace=False)
    return np.concatenate([d[k*200:(k+1)*200][rng.choice(200, 20, replace=False)] for k in t]).mean()
for k, nm in enumerate(["aleatorio simple", "estratificado", "3 camiones × 20"]):
    E = np.array([est(k) for _ in range(20000)]); print(f"{nm:17s}: media {E.mean():.2f}  error típico {E.std():.2f}")
nh = asignar(60, Nh, Sh); W = Nh / N
E = np.array([sum(W[h] * d[r == h][rng.choice(Nh[h], nh[h], replace=False)].mean() for h in range(3)) for _ in range(20000)])
print(f"Neyman           : media {E.mean():.2f}  error típico {E.std():.2f}")
print(f"teoría: simple {sig/np.sqrt(60):.2f}  proporcional {np.sqrt(np.sum(W*Sh**2)/60):.2f}  Neyman {np.sum(W*Sh)/np.sqrt(60):.2f}")
print(f"coste anual de las paradas: {N*mu*0.70:,.0f} €;  cada minuto de error en la media: {N*0.70:,.0f} €")
