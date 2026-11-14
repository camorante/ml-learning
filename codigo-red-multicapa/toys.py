"""Números de los juguetes de la versión sencilla (misma aritmética que el JavaScript) y embed.json."""
import json, math
import numpy as np
from datos import carga_consumo, XOR, motor, perfil, KM
from red import MLP, Densa, sigmoide, n_param

# ---------- la franja del tema 1 ----------
C, y = carga_consumo()
r1 = json.load(open("../../r1/embed.json", encoding="utf-8"))
assert r1["cc"] == C[:, 0].tolist() and r1["cq"] == C[:, 1].tolist() and r1["cy"] == y.tolist()
print("viajes idénticos a los del tema 1:", len(y), "viajes,", y.sum(), "no cuadran",
      f"({int(np.sum((y == 1) & (C[:, 1] > 24 + 0.55 * C[:, 0])))} por encima, {int(np.sum((y == 1) & (C[:, 1] < 24 + 0.55 * C[:, 0])))} por debajo)")


def mejor_recta(P, t, n=3600):
    """Mínimo de errores de una sola recta (una neurona con escalón), probando direcciones y umbrales."""
    sc = np.array([22.0, 26.0]); Q = (P - P.mean(0)) / sc; best = 99
    for k in range(n):
        th = math.pi * k / n; u = Q @ np.array([math.cos(th), math.sin(th)])
        ts = t[np.argsort(u)]
        for j in range(len(u) + 1):
            e = ts[:j].sum() + (1 - ts[j:]).sum(); best = min(best, e, len(u) - e)
    return int(best)


print("mejor recta sola: errores", mejor_recta(C, y), "de 30 (como en el tema 1)")

# red 2-2-1 hecha a mano: A = «gasta de más» (por encima de 26,5 + 0,55·carga), B = «gasta de menos»
K = 2.0
W1 = np.array([[-0.55 * K, K], [0.55 * K, -K]]); b1 = np.array([-26.5 * K, 21.5 * K])
w2 = np.array([[10.0, 10.0]]); b2 = np.array([-5.0])
red = MLP([Densa(W1, b1), Densa(w2, b2)])
p, ((Z1, H), (Z2, _)) = red.forward(C, todo=True)
p = p[:, 0]
print(f"red 2-2-1 a mano: W1 = {W1.tolist()}, b1 = {b1.tolist()}, w2 = {w2[0].tolist()}, b2 = {b2[0]}")
print(f"   {red.n_param} parámetros = n_param([2, 2, 1]) = {n_param([2, 2, 1])};  errores: {int(np.sum((p > 0.5) != y))} de 30")
print(f"   normales: h_A ≤ {H[y == 0, 0].max():.3f}, h_B ≤ {H[y == 0, 1].max():.3f}, salida ≤ {p[y == 0].max():.3f}")
print(f"   raros:    max(h_A, h_B) ≥ {H[y == 1].max(1).min():.3f}, salida ≥ {p[y == 1].min():.3f}")
for i in (5, 3, 6):                                   # viajes 5 (encima), 3 (debajo) y 6 (normal) de la lista
    c, q = C[i - 1]
    print(f"   viaje {i}: carga {c} t, consumo {q} → z = ({Z1[i - 1, 0]:.2f}, {Z1[i - 1, 1]:.2f}),"
          f" h = ({H[i - 1, 0]:.4f}, {H[i - 1, 1]:.4f}), z_sal = {Z2[i - 1, 0]:.3f}, p = {p[i - 1]:.4f}")
# la misma red con «Y» en la salida (hacen falta los dos votos)
pY = sigmoide(H @ np.array([10.0, 10.0]) - 15.0)
print(f"   con «Y» (b2 = −15): avisa en {int(np.sum(pY > 0.5))} viajes, errores {int(np.sum((pY > 0.5) != y))}")

# ---------- XOR ----------
Xx, yx = XOR
rx = MLP([Densa([[10, 10], [10, 10]], [-5, -15]), Densa([[10, -10]], [-5])])
px, ((_, Hx), _) = rx.forward(Xx, todo=True)
for x, h, q in zip(Xx, Hx, px[:, 0]):
    print(f"XOR {x.astype(int)} → h = ({h[0]:.4f}, {h[1]:.4f}) → p = {q:.4f}")

# ---------- anchura: zona normal del motor (pesos entrenados en anchura.py) ----------
Xm, ym = motor()
AN = json.load(open("anchura.json", encoding="utf-8"))
print("zona normal, errores (de 120) y acierto en 2 000 nuevas por número de neuronas:",
      {m: (v["e"], v["a"]) for m, v in AN.items()})

# ---------- profundidad: la tira de papel doblada ----------
relu = lambda z: np.maximum(z, 0)
def sierra(u, L):
    """L capas de 2 neuronas ReLU; cada capa dobla [0, 1] por la mitad: t(u) = 2·relu(u) − 4·relu(u − 0,5)."""
    for _ in range(L):
        u = 2 * relu(u) - 4 * relu(u - 0.5)
    return u
uu = np.linspace(0, 1, 2 ** 14 + 1)
for L in range(1, 9):
    s = sierra(uu, L); pend = np.diff(s) / np.diff(uu)
    trozos = 1 + int(np.sum(np.abs(np.diff(pend)) > 1e-6))
    print(f"profundidad {L}: {trozos:3d} trozos rectos, {6 * L + 1:3d} parámetros;"
          f" una sola capa necesita {trozos - 1:3d} neuronas = {3 * (trozos - 1) + 1:4d} parámetros")

# ---------- aproximación universal: perfil de velocidad de una ruta ----------
v = perfil(KM)
def ajuste(n):
    c = (np.arange(n) + 0.5) * 100 / n; w = 0.35 * 100 / n
    B = np.c_[np.ones_like(KM), sigmoide((KM[:, None] - c) / w)]
    a, *_ = np.linalg.lstsq(B, v, rcond=None)
    return c, w, a, float(np.sqrt(np.mean((B @ a - v) ** 2))), float(np.max(np.abs(B @ a - v)))
AP = {}
for n in (1, 2, 4, 8, 16, 32, 64):
    c, w, a, e, emax = ajuste(n)
    AP[n] = {"w": round(w, 6), "a": np.round(a, 5).tolist(), "e": round(e, 3)}
    print(f"perfil con {n:2d} neuronas: error medio {e:6.2f} km/h, máximo {emax:6.2f} km/h, {3 * n + 1} números")

emb = {"cc": C[:, 0].tolist(), "cq": C[:, 1].tolist(), "cy": y.tolist(),
       "mr": Xm[:, 0].tolist(), "mt": Xm[:, 1].tolist(), "my": ym.tolist(), "an": AN,
       "v": np.round(v, 3).tolist(), "ap": AP}
json.dump(emb, open("../embed.json", "w", encoding="utf-8"), separators=(",", ":"))
print("embed.json:", len(json.dumps(emb, separators=(",", ":"))), "bytes")
json.dump({"cc": C[:, 0].tolist(), "cq": C[:, 1].tolist(), "cy": y.tolist()}, open("../tembed.json", "w", encoding="utf-8"), separators=(",", ":"))
