"""Datos y cifras de los juguetes de la versión sencilla (embed.json) y comprobaciones."""
import json, os, math
import numpy as np
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ---------- héroe: posición GPS de un camión en 30 minutos ----------
# velocidad (km/h) como suma de escalones suaves S(u) = 3u² − 2u³; posición (km) = ∫ v/60 dt (t en minutos)
STEPS = [(0, 3, 50), (8, 2, 40), (19, 1, -60), (23, 2, 50)]   # (inicio, duración, salto de velocidad)
S = lambda u: np.where(u <= 0, 0, np.where(u >= 1, 1, 3 * u**2 - 2 * u**3))
I = lambda u: np.where(u <= 0, 0, np.where(u >= 1, u - 0.5, u**3 - u**4 / 2))
vel = lambda t: sum(A * S((t - t0) / d) for t0, d, A in STEPS)
pos = lambda t: sum(A * d * I((t - t0) / d) for t0, d, A in STEPS) / 60
print("héroe: x(30) =", round(float(pos(30)), 3), "km")
for t, h in [(15, 10), (15, 5), (15, 1), (15, 0.1), (15, 0.01), (19.5, 10), (19.5, 1), (19.5, 0.1), (19.5, 0.01), (5, 10), (5, 1)]:
    sec = (pos(t + h) - pos(t)) / h * 60
    print(f"   t={t} h={h}: media {float(sec):.2f} km/h  velocímetro {float(vel(t)):.2f}")

# ---------- idea 1: 60 s de velocidad GPS a 1 Hz con una frenada brusca ----------
rng = np.random.default_rng(16)
ts = np.arange(61)
v1 = 72 + 0 * ts.astype(float)
v1 -= 50 * S((ts - 30) / 4.0)          # frenada de 72 a 22 km/h en 4 s
v1 += 18 * S((ts - 40) / 15.0)         # vuelve a acelerar suave
v1 = np.round(v1 + rng.normal(0, 0.5, ts.size), 1)
def acc(v, dt):
    a = np.full(v.size, np.nan)
    a[dt:] = (v[dt:] - v[:-dt]) / dt / 3.6       # m/s², diferencia hacia atrás
    return a
for dt in (1, 2, 3, 4, 5, 8, 10):
    a = acc(v1, dt); print(f"idea1 Δ={dt}s: mínimo {np.nanmin(a):.2f} m/s² en t={int(np.nanargmin(a))}  alerta(< −3): {int(np.sum(a < -3))} s")

# ---------- idea 3: coste por 100 km según la velocidad ----------
def coste(v, H=30, p=1.6):
    return p * (20 + 0.0016 * v**2) + H * 100 / v
def dcoste(v, H=30, p=1.6):
    return p * 0.0032 * v - H * 100 / v**2
for H in (15, 30, 60):
    vs = (H * 100 / (p := 1.6) / 0.0032) ** (1 / 3)
    print(f"idea3 H={H}: v* = {vs:.2f} km/h  coste {coste(vs, H):.2f} €/100 km  a 90: {coste(90, H):.2f}  a 70: {coste(70, H):.2f}")
for v in (60, 70, 80, 84, 90, 100):
    print(f"   v={v}: coste {coste(v):.2f}  pendiente {dcoste(v):+.3f} €/100km por km/h  combustible {1.6*(20+0.0016*v*v):.2f} tiempo {3000/v:.2f}")

# ---------- idea 4: recta del tema 4 (10 viajes) ----------
x = np.array([2.0, 4.5, 6.0, 8.5, 10.0, 12.5, 15.0, 17.5, 20.0, 23.5])
y = np.array([26.0, 29.2, 29.7, 28.5, 29.6, 30.5, 33.3, 33.7, 36.0, 34.2])
X = np.c_[np.ones_like(x), x / 10]          # pendiente en litros por cada 10 t
beta = np.linalg.lstsq(X, y, rcond=None)[0]
sse = lambda a, m: float(np.sum((y - a - m * x / 10) ** 2))
grad = lambda a, m: (-2 * float(np.sum(y - a - m * x / 10)), -2 * float(np.sum((y - a - m * x / 10) * x / 10)))
print("idea4 mínimo (altura, L por 10 t):", beta.round(4), "SSE", round(sse(*beta), 3))
H4 = 2 * X.T @ X; ev = np.linalg.eigvalsh(H4)
print("   hessiana", H4.round(3).tolist(), "autovalores", ev.round(3), "paso máx 2/λmax =", round(2 / ev[-1], 4))
for a, m in [(30, 1.0), (22, 6), (26.2, 4.08)]:
    print(f"   en ({a}, {m}): SSE {sse(a, m):.2f}  gradiente {np.round(grad(a, m), 3)}")
for eta in (0.01, 0.06, 0.075):
    a, m = 30.0, 1.0
    for k in range(40):
        g = grad(a, m); a -= eta * g[0]; m -= eta * g[1]
    print(f"   η={eta}: tras 40 pasos ({a:.3f}, {m:.3f}) SSE {sse(a, m):.3f}")

# ---------- idea 5: cadena km/h → L/h → €/h ----------
Lh = lambda v: (20 + 0.0016 * v**2) * v / 100
dLh = lambda v: 0.2 + 0.000048 * v**2
for v, p in [(80, 1.6), (60, 1.6), (90, 1.9)]:
    print(f"idea5 v={v} p={p}: L/h {Lh(v):.3f}  dL/h/dv {dLh(v):.4f}  €/h {p*Lh(v):.3f}  d€/dv {p*dLh(v):.4f}")
    for dv in (1, 5, 20):
        print(f"      +{dv}: real {p*(Lh(v+dv)-Lh(v)):.3f}  cadena {p*dLh(v)*dv:.3f}")

# ---------- idea 6: una neurona (atascos → llegar tarde) ----------
sig = lambda z: 1 / (1 + math.exp(-z))
xx, yy, w, b, eta = 2.0, 1, 0.5, -1.5, 0.5
for k in range(6):
    z = w * xx + b; p = sig(z); L = -math.log(p) if yy else -math.log(1 - p)
    dz = p - yy
    print(f"idea6 paso {k}: w={w:.4f} b={b:.4f} z={z:.4f} p={p:.4f} L={L:.4f} dL/dp={-1/p if yy else 1/(1-p):.4f} dp/dz={p*(1-p):.4f} dL/dz={dz:.4f} dL/dw={dz*xx:.4f} dL/db={dz:.4f}")
    w -= eta * dz * xx; b -= eta * dz

# ---------- idea 7: gradientes que se desvanecen ----------
for f in (0.25, 0.5, 0.9, 1.0, 1.1, 1.5):
    print(f"idea7 factor {f}: 10 capas {f**10:.3g}  20 capas {f**20:.3g}  30 capas {f**30:.3g}")

json.dump({"v1": v1.tolist(), "x": x.tolist(), "y": y.tolist(), "beta": beta.tolist()}, open("../embed.json", "w"))
