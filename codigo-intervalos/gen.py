"""Genera ../embed.json (datos de la versión sencilla) e imprime las cifras que se citan en el texto."""
import json, math
import numpy as np
from scipy import stats
from sim import *
from ic import ic_media, ic_wald, ic_wilson, ic_diferencia, bootstrap, bootstrap_grupos

rd = lambda a, d=1: [round(float(v), d) for v in a]
out = {}
X = trayectos(); out["tray"] = rd(X)
print("población: media", MU, "desv", SIGMA, "| 1000 trayectos: media", X.mean().round(3), "s", X.std(ddof=1).round(3))
for n in (10, 30, 100, 1000):
    for c in (0.90, 0.95, 0.99):
        lo, hi = ic_media(X[:n], c); m = X[:n].mean()
        print(f"juguete n={n:4d} conf={c:.2f}: media {m:.3f} s {X[:n].std(ddof=1):.3f}  ±{(hi - lo) / 2:.3f}  [{lo:.2f}, {hi:.2f}]  ¿contiene 32? {lo <= MU <= hi}")
print("10 trayectos:", DIEZ.tolist(), "suma", DIEZ.sum())

# idea 1: con s = 4,5 y z
for n in (10, 25, 100, 400, 1000):
    for z in (1.645, 1.96, 2.576):
        print(f"idea 1: n={n:4d} z={z}: error estándar {4.5 / math.sqrt(n):.3f}  margen ±{z * 4.5 / math.sqrt(n):.3f}")
print("n para ±0,5 con z = 1,96:", (1.96 * 4.5 / 0.5) ** 2)

# idea 3: multiplicador t y lo que se acierta con 1,96 cuando se usa s
for n in (3, 4, 5, 10, 20, 30, 100):
    tc = stats.t.ppf(0.975, n - 1); cov = 1 - 2 * stats.t.sf(1.96, n - 1)
    print(f"idea 3: n={n:3d}: t = {tc:.3f}  cobertura usando 1,96 con s: {cov:.3%}")

# idea 4: entregas tarde
for k, n in [(1, 40), (0, 40), (3, 40), (10, 200), (0, 20)]:
    w, s = ic_wald(k, n), ic_wilson(k, n)
    print(f"idea 4: {k}/{n}: Wald [{w[0]:.4f}, {w[1]:.4f}]  Wilson [{s[0]:.4f}, {s[1]:.4f}]")
def cobertura_prop(n, p, f):
    k = np.arange(n + 1); pk = stats.binom.pmf(k, n, p)
    return sum(pk[i] for i in k if f(int(i), n)[0] <= p <= f(int(i), n)[1])
for n in (40, 100, 400):
    print(f"idea 4: cobertura real con p = 2 %, n = {n}: Wald {cobertura_prop(n, 0.02, ic_wald):.3f}  Wilson {cobertura_prop(n, 0.02, ic_wilson):.3f}")

# idea 5: bootstrap de 12 viajes
out["km12"], out["eur12"] = KM12.tolist(), EUR12.tolist()
ratio = lambda idx_or_none=None: None
for nombre, f in [("media", lambda e, k: e.mean(-1)), ("mediana", lambda e, k: np.median(e, -1)), ("€/km", lambda e, k: e.sum(-1) / k.sum(-1))]:
    th = f(EUR12, KM12)
    idx = np.random.default_rng(1).integers(0, 12, (10000, 12))
    tb = f(EUR12[idx], KM12[idx])
    print(f"idea 5: {nombre}: valor {th:.4f}  bootstrap percentil 95 % [{np.quantile(tb, .025):.4f}, {np.quantile(tb, .975):.4f}]  error estándar {tb.std(ddof=1):.4f}")
print("idea 5: intervalo t para la media", ic_media(EUR12), " suma €", EUR12.sum(), "suma km", KM12.sum())
print("   P(el viaje largo no sale en un remuestreo) = (11/12)^12 =", (11 / 12) ** 12)

# idea 6: cuando falla
x, g = camiones(); out["cam"] = [rd(x), g.tolist()]
lo, hi = ic_media(x); print(f"idea 6 camiones: media {x.mean():.3f}  t ingenuo [{lo:.2f}, {hi:.2f}]", "medias por camión", np.round([x[g == i].mean() for i in range(8)], 2))
idx = np.random.default_rng(2).integers(0, len(x), (2000, len(x))); tb = x[idx].mean(-1)
print(f"   bootstrap por trayectos [{np.quantile(tb, .025):.2f}, {np.quantile(tb, .975):.2f}]   por camiones {np.round(bootstrap_grupos(x, g, np.mean, B=2000, seed=2), 2)}")
# (la cobertura simulada de estos intervalos está en fallos.py)
pk = picos(); out["picos"] = rd(pk)
idx = np.random.default_rng(3).integers(0, 20, (2000, 20)); mb = pk[idx].max(-1)
print(f"idea 6 picos: máximo {pk.max()}  remuestreos con el máximo exacto {np.mean(mb == pk.max()):.1%}  (teoría 1 − (19/20)^20 = {1 - (19 / 20) ** 20:.1%})  percentil [{np.quantile(mb, .025)}, {np.quantile(mb, .975)}]")
cuatro = X[:4]; out["cuatro"] = rd(cuatro)
idx = np.random.default_rng(4).integers(0, 4, (2000, 4)); tb = cuatro[idx].mean(-1)
print(f"idea 6 cuatro: {cuatro.tolist()} media {cuatro.mean():.3f}  bootstrap [{np.quantile(tb, .025):.2f}, {np.quantile(tb, .975):.2f}]  t {np.round(ic_media(cuatro), 2)}  remuestreos distintos C(7,4) = {math.comb(7, 4)}")
# (la cobertura con 4, 10 y 30 datos está en cobertura.py)

# idea 7: dos depósitos
a, b = depositos(); out["dep"] = [rd(a), rd(b)]
for n in (10, 20, 30, 40, 60, 80, 100, 150, 200):
    lo, hi = ic_diferencia(b[:n], a[:n]); la, ha = ic_media(a[:n]); lb, hb = ic_media(b[:n])
    print(f"idea 7 n={n:3d}: Norte {a[:n].mean():.2f} [{la:.2f}, {ha:.2f}]  Sur {b[:n].mean():.2f} [{lb:.2f}, {hb:.2f}]  Sur − Norte {b[:n].mean() - a[:n].mean():+.2f} [{lo:+.2f}, {hi:+.2f}]  solapan {ha >= lb}")
json.dump(out, open("../embed.json", "w"), separators=(",", ":"))
print("embed.json", len(json.dumps(out, separators=(",", ":"))), "bytes")
