"""Datos de los dibujos (embed.json y tembed.json) y las cifras que enseñan, calculadas con bucles explícitos en el
mismo orden que el JavaScript de las páginas (que ejecuta el modelo de verdad en el navegador)."""
import json, math, os
import numpy as np
import torch
from datos import VOCAB, ESPECIALES, DEPOSITOS, CODIGOS, ID, datos
from modelo import Viajero, a_numpy

os.chdir(os.path.dirname(os.path.abspath(__file__)))
R = lambda a: np.round(np.asarray(a, dtype=float), 5).tolist()


def pesos(f):
    m = Viajero(con_posicion="sin" not in f); m.load_state_dict(torch.load(f))
    return m, {k: R(v) for k, v in a_numpy(m).items()}


m1, P1 = pesos("viajero.pt")
m0, P0 = pesos("viajero_sin_posicion.pt")
Xt, Lt = datos()[2]
EJ = 374
viaje = Xt[EJ, :Lt[EJ]].tolist()


def mat(X, W, b):
    out = []
    for x in X:
        fila = []
        for j in range(len(W[0])):
            s = 0.0
            for k in range(len(x)):
                s += x[k] * W[k][j]
            fila.append(s + b[j])
        out.append(fila)
    return out


def norma(X, g, b):
    out = []
    for x in X:
        mu = 0.0
        for v in x: mu += v
        mu /= len(x)
        var = 0.0
        for v in x: var += (v - mu) * (v - mu)
        var /= len(x)
        s = math.sqrt(var + 1e-5)
        out.append([g[j] * (x[j] - mu) / s + b[j] for j in range(len(x))])
    return out


def softmax(z):
    m = max(z); e = [math.exp(v - m) if v != -math.inf else 0.0 for v in z]; s = 0.0
    for v in e: s += v
    return [v / s for v in e]


def adelante(ids, P, T=1.0, causal=True, pos=True, h=2):
    """El mismo orden de cuentas que forward() en simple.js."""
    n, d = len(ids), len(P["E"][0]); dh = d // h
    X = []
    for i, t in enumerate(ids):
        fila = []
        for j in range(d):
            w = 1.0 / 10000 ** ((j - j % 2) / d)
            fila.append(P["E"][t][j] + ((math.sin(i * w) if j % 2 == 0 else math.cos(i * w)) if pos else 0.0))
        X.append(fila)
    A = norma(X, P["g1"], P["be1"])
    Q, K, V = mat(A, P["Wq"], P["bq"]), mat(A, P["Wk"], P["bk"]), mat(A, P["Wv"], P["bv"])
    W, C = [], [[0.0] * d for _ in range(n)]
    for c in range(h):
        Wc = []
        for i in range(n):
            s = []
            for j in range(n):
                if causal and j > i: s.append(-math.inf); continue
                a = 0.0
                for k in range(dh): a += Q[i][c * dh + k] * K[j][c * dh + k]
                s.append(a / math.sqrt(dh) / T)
            w = softmax(s); Wc.append(w)
            for k in range(dh):
                a = 0.0
                for j in range(n): a += w[j] * V[j][c * dh + k]
                C[i][c * dh + k] = a
        W.append(Wc)
    O = mat(C, P["Wo"], P["bo"])
    H = [[X[i][j] + O[i][j] for j in range(d)] for i in range(n)]
    F1 = [[max(0.0, v) for v in f] for f in mat(norma(H, P["g2"], P["be2"]), P["W1"], P["b1"])]
    F2 = mat(F1, P["W2"], P["b2"])
    Y = [[H[i][j] + F2[i][j] for j in range(d)] for i in range(n)]
    Z = mat(norma(Y, P["gf"], P["bef"]), P["Wout"], P["bout"])
    return W, [softmax(z) for z in Z]


def top(p, k=3):
    o = sorted(range(len(p)), key=lambda v: -p[v])[:k]
    return ", ".join(f"{VOCAB[v]} {p[v]:.1%}" for v in o)


with torch.no_grad():
    ref = m1(torch.tensor([viaje])).softmax(-1)[0].numpy()
W, Pr = adelante(viaje, P1)
print("viaje de ejemplo:", " ".join(f"{i}:{VOCAB[t]}" for i, t in enumerate(viaje)))
print(f"pesos redondeados a 5 decimales · probabilidades frente al modelo completo: error máximo {np.max(np.abs(np.array(Pr) - ref)):.1e}")
ir, it = viaje.index(ID["regreso"]), viaje.index(ID["taller"])
cods = [i for i in range(it) if VOCAB[viaje[i]] in CODIGOS]
print(f"juguete · desde «regreso» (pos. {ir}): peso en Valencia (pos. 1) cabeza 1 {W[0][ir][1]:.1%}, cabeza 2 {W[1][ir][1]:.1%}"
      f" · predice {top(Pr[ir])}")
print(f"juguete · desde «taller» (pos. {it}): cabeza 1 en {VOCAB[viaje[cods[1]]]} {W[0][it][cods[1]]:.1%} y en"
      f" {VOCAB[viaje[cods[0]]]} {W[0][it][cods[0]]:.1%}; cabeza 2 {W[1][it][cods[1]]:.1%} y {W[1][it][cods[0]]:.1%} · predice {top(Pr[it])}")
for T in (0.25, 0.5, 2.0, 4.0):
    Wt, Pt = adelante(viaje, P1, T)
    print(f"temperatura {T}: desde «regreso», peso en Valencia {Wt[0][ir][1]:.1%} / {Wt[1][ir][1]:.1%} · predice {top(Pt[ir], 2)}"
          f" · desde «taller», en {VOCAB[viaje[cods[1]]]} {Wt[0][it][cods[1]]:.1%} · predice {top(Pt[it], 2)}")
Wm, Pm = adelante(viaje, P1, causal=False)
print(f"sin máscara: desde «regreso», peso en Valencia {Wm[0][ir][1]:.1%} / {Wm[1][ir][1]:.1%}, en la Valencia del final (pos. {ir + 1})"
      f" {Wm[0][ir][ir + 1]:.1%} / {Wm[1][ir][ir + 1]:.1%} · predice {top(Pm[ir], 2)} · desde «taller» predice {top(Pm[it], 2)}")
ok = sum(np.argmax(Pm[i]) == viaje[i + 1] for i in range(len(viaje) - 1)); ok1 = sum(np.argmax(Pr[i]) == viaje[i + 1] for i in range(len(viaje) - 1))
print(f"fichas siguientes acertadas en el viaje: con máscara {ok1} de {len(viaje) - 1}, sin máscara {ok} de {len(viaje) - 1}")

# idea 5: el orden de los códigos
cambiado = viaje.copy(); cambiado[cods[0]], cambiado[cods[1]] = viaje[cods[1]], viaje[cods[0]]
for nombre, P, pos in (("con posición", P1, True), ("sin posición", P0, False)):
    for et, v in (("original", viaje), ("cambiado", cambiado)):
        _, p = adelante(v, P, pos=pos)
        print(f"idea 5 · {nombre}, orden {et}: tras «taller» P0299 {p[it][ID['P0299']]:.1%} · C0035 {p[it][ID['C0035']]:.1%}")

# idea 6: un viaje escrito a mano
mano = ["salida", "Sevilla", "carga", "P0217", "frenada", "C0035", "parada", "taller"]
_, p = adelante([ID[t] for t in mano], P1)
print("idea 6 ·", " ".join(mano), "→", top(p[-1]))
mano += ["C0035", "atasco", "regreso"]
_, p = adelante([ID[t] for t in mano], P1)
print("idea 6 ·", " ".join(mano), "→", top(p[-1]))
_, p = adelante([ID["salida"]], P1)
print("idea 6 · salida →", top(p[0], 4))

# idea 3: una pregunta frente a 10 claves para cada d
rng = np.random.default_rng(7)
muestra = {}
for d in (2, 8, 16, 64, 128, 512):
    q, K = rng.normal(size=d), rng.normal(size=(10, d))
    s = []
    for k in K:
        a = 0.0
        for j in range(d): a += q[j] * k[j]
        s.append(round(a, 4))
    muestra[d] = s
    p0, p1 = softmax(s), softmax([v / math.sqrt(d) for v in s])
    print(f"idea 3 · d = {d:3d}: puntuaciones de {min(s):.2f} a {max(s):.2f} · peso mayor sin dividir {max(p0):.1%}, dividiendo entre √d {max(p1):.1%}")

comp = json.load(open("comparar.json", encoding="utf-8"))
esc = json.load(open("escala.json", encoding="utf-8"))
datos_pag = {"vocab": VOCAB, "tipo": [0] * 4 + [1] * 4 + [2] * 4 + [3] * 8, "viaje": viaje, "P": P1, "P0": P0,
             "muestra": muestra, "escala": esc, "comp": {k: {"dep": v["dep"], "cod": v["cod"], "tramos_d": v["tramos_d"],
                                                              "seg": [round(s) for s in v["seg"]]} for k, v in comp.items()}}
json.dump(datos_pag, open("../embed.json", "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
json.dump({"vocab": VOCAB, "tipo": datos_pag["tipo"], "viaje": viaje, "P": P1}, open("../tembed.json", "w", encoding="utf-8"),
          ensure_ascii=False, separators=(",", ":"))
print("embed.json", os.path.getsize("../embed.json"), "bytes · tembed.json", os.path.getsize("../tembed.json"), "bytes")
