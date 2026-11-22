"""Datos de los dibujos: ../embed.json (sencilla) y ../tembed.json (técnica). Imprime las cifras que citan las
páginas. Antes hay que ejecutar comparar.py, bptt.py y prever.py (guardan las redes y los ecos)."""
import json, math, os
import numpy as np
import torch
from datos import HUECOS, L0, datos
from modelos import Lectora, ESCALA
import prever

os.chdir(os.path.dirname(os.path.abspath(__file__)))
r2 = lambda a: [round(float(v), 2) for v in np.ravel(a)]
r3 = lambda a: [round(float(v), 3) for v in np.ravel(a)]
R = json.load(open("comparar.json", encoding="utf-8"))
ECO = json.load(open("bptt.json", encoding="utf-8"))


def cargar(tipo, H, G):
    m = Lectora(tipo, H); m.load_state_dict(torch.load(f"red_{tipo}{H}_G{G}.pt")); m.eval(); return m


def recorrer(m, X):
    """Estados paso a paso de una red de 16 unidades: h_t (y c_t en la LSTM) para cada tramo de X."""
    P = {k: v.detach().numpy() for k, v in m.state_dict().items()}
    W, U, b = P["rec.weight_ih_l0"], P["rec.weight_hh_l0"], P["rec.bias_ih_l0"] + P["rec.bias_hh_l0"]
    x = np.asarray(X)[:, :, None] / ESCALA; n, T, _ = x.shape; H = U.shape[1]
    h, c, hs, cs = np.zeros((n, H)), np.zeros((n, H)), [], []
    sg = lambda z: 1 / (1 + np.exp(-z))
    for t in range(T):
        z = x[:, t] @ W.T + h @ U.T + b
        if m.tipo == "RNN":
            h = np.tanh(z)
        else:
            c = sg(z[:, H:2 * H]) * c + sg(z[:, :H]) * np.tanh(z[:, 2 * H:3 * H]); h = sg(z[:, 3 * H:]) * np.tanh(c)
        hs.append(h); cs.append(c)
    hs, cs = np.stack(hs, 1), np.stack(cs, 1)
    p = 1 / (1 + np.exp(-(hs @ P["out.weight"][0] + P["out.bias"][0])))      # la respuesta leída en cada paso
    return hs, cs, p


# ---------- aciertos (comparar.py) ----------
ACC = {t: [next(r["prueba"] for r in R if r["tipo"] == t and r["H"] == 16 and r["G"] == G and r["seed"] == 0 and not r["extra"])
           for G in HUECOS] for t in ("RNN", "LSTM", "GRU")}
print("aciertos semilla 0:", {t: [f"{a:.1%}" for a in v] for t, v in ACC.items()})

# ---------- el juguete: un tramo con frenada brusca y otro con frenada suave por hueco ----------
TR, RUN = {}, {}
for G in HUECOS:
    (_, _, _), (Xv, Yv, _), (Xt, Yt, LIM) = datos(G)
    ib, isv = int(np.argmax(Yt == 1)), int(np.argmax(Yt == 0))            # el primero de cada clase en la prueba
    TR[G] = {"brusca": r2(Xt[ib]), "suave": r2(Xt[isv])}
    RUN[G] = {}
    for tipo in ("RNN", "LSTM"):
        m = cargar(tipo, 16, G)
        hv, cv, _ = recorrer(m, Xv)
        fin = (hv if tipo == "RNN" else cv)[:, -1]
        cor = [abs(np.corrcoef(fin[:, k], Yv)[0, 1]) if fin[:, k].std() > 1e-9 else 0 for k in range(16)]
        k = int(np.argmax(cor))                                           # la unidad que más «recuerda» la respuesta
        hs, cs, p = recorrer(m, np.array([TR[G]["brusca"], TR[G]["suave"]]))
        mem = hs if tipo == "RNN" else cs
        for j, nm in enumerate(("brusca", "suave")):
            RUN[G].setdefault(nm, {})[tipo] = {"p": r3(p[j]), "m": r3(mem[j, :, k])}
        print(f"hueco {G:3d} {tipo:4s}: unidad de memoria {k:2d} (|correlación| con la respuesta {cor[k]:.2f});"
              f" respuesta final brusca {p[0, -1]:.3f}, suave {p[1, -1]:.3f}; tramos de prueba nº {ib} y {isv}")

# ---------- idea 1: una neurona recurrente hecha a mano, h_t = tanh(a·x_t + w·h_{t−1}) ----------
A1 = -0.5


def neurona(x, w):
    h, out = 0.0, []
    for v in x:
        h = math.tanh(A1 * v + w * h)
        out.append(h)
    return out


N1 = {"brusca": TR[50]["brusca"][:L0], "suave": TR[50]["suave"][:L0], "hueco": TR[50]["brusca"][L0:]}
for modo in ("lisa", "real"):
    hueco = [0.0] * 50 if modo == "lisa" else N1["hueco"]
    for w in (0.5, 0.9, 1.0, 1.5):
        s = [neurona(N1[k] + hueco, w) for k in ("brusca", "suave")]
        print(f"neurona a mano, hueco {modo}, w = {w}: h al acabar el arranque {s[0][L0 - 1]:.2f} (brusca) y {s[1][L0 - 1]:.2f} (suave);"
              f" 1 s después {s[0][L0 + 9]:.3f} y {s[1][L0 + 9]:.3f}; al final {s[0][-1]:.3f} y {s[1][-1]:.3f}")

# ---------- idea 3: la LSTM de una sola unidad (hueco de 25), con sus pesos redondeados a 2 decimales ----------
m1 = cargar("LSTM", 1, 25)
P = {k: v.detach().numpy().ravel() for k, v in m1.state_dict().items()}
UNO = dict(W=r2(P["rec.weight_ih_l0"]), U=r2(P["rec.weight_hh_l0"]), b=r2(P["rec.bias_ih_l0"] + P["rec.bias_hh_l0"]),
           wo=round(float(P["out.weight"][0]), 2), bo=round(float(P["out.bias"][0]), 2))
print("LSTM de una unidad, pesos redondeados (bloques i, f, g, o):", UNO)
_, _, (Xt, Yt, LIM) = datos(25)


def celda1(x, W, U, b, f_fija=None):
    """La LSTM de una unidad, paso a paso, en el mismo orden de operaciones que el JS."""
    sg = lambda z: 1 / (1 + math.exp(-z))
    h = c = 0.0; out = []
    for v in x:
        xs = v / ESCALA
        i = sg(W[0] * xs + U[0] * h + b[0]); f = sg(W[1] * xs + U[1] * h + b[1]) if f_fija is None else f_fija
        g = math.tanh(W[2] * xs + U[2] * h + b[2]); o = sg(W[3] * xs + U[3] * h + b[3])
        c = f * c + i * g; h = o * math.tanh(c)
        out.append((i, f, g, o, c, h))
    return out


resp = lambda h: 1 / (1 + math.exp(-(UNO["wo"] * h + UNO["bo"])))
ok = sum((resp(celda1(r2(x), UNO["W"], UNO["U"], UNO["b"])[-1][5]) > 0.5) == bool(y) for x, y in zip(Xt, Yt))
print(f"LSTM de una unidad con los pesos redondeados: acierta {ok} de 1 000 tramos de prueba (hueco 25)")
bache = lambda i: np.abs(np.diff(LIM[i][L0:])).max() > 5
I3 = {"brusca": int(np.argmax(Yt == 1)), "suave": int(np.argmax(Yt == 0)),
      "bache": next(i for i in range(len(Yt)) if Yt[i] == 0 and bache(i))}
TR3 = {k: r2(Xt[i]) for k, i in I3.items()}
for k, x in TR3.items():
    s = celda1(x, UNO["W"], UNO["U"], UNO["b"]); s9 = celda1(x, UNO["W"], UNO["U"], UNO["b"], 0.9)
    print(f"una unidad, tramo «{k}» (prueba nº {I3[k]}): c al final {s[-1][4]:.2f}, h {s[-1][5]:.2f}, respuesta {resp(s[-1][5]):.3f};"
          f" f mínima {min(r[1] for r in s):.4f}; con f fija en 0,9: c {s9[-1][4]:.2f}, respuesta {resp(s9[-1][5]):.3f}")
for f9 in (None, 0.9):
    ok9 = sum((resp(celda1(r2(x), UNO["W"], UNO["U"], UNO["b"], f9)[-1][5]) > 0.5) == bool(y) for x, y in zip(Xt, Yt))
    print(f"   puerta de olvido {'aprendida' if f9 is None else 'fija en 0,9'}: acierta {ok9} de 1 000")

# ---------- idea 4: prever la velocidad 3 s por delante ----------
PV = {}
for tipo in ("RNN", "LSTM"):
    mp = prever.Prevision(tipo); mp.load_state_dict(torch.load(f"prever_{tipo}.pt")); mp.eval()
    with torch.no_grad():
        PV[tipo] = mp(torch.tensor(prever.X_TE[:3])).numpy()
PREV = {"v": [r2(v) for v in prever.X_TE[:3]], "y": [r2(v) for v in prever.Y_TE[:3]],
        "rnn": [r2(v) for v in PV["RNN"]], "lstm": [r2(v) for v in PV["LSTM"]]}
for j in range(3):
    e = lambda P: float(np.mean(np.abs(np.array(P[j][10:]) - np.array(PREV["y"][j][10:]))))
    print(f"trayecto de prueba {j}: error medio RNN {e(PREV['rnn']):.2f}, LSTM {e(PREV['lstm']):.2f}, «seguirá igual» {e(PREV['v']):.2f} km/h")

ECOS = {k: v for k, v in ECO.items()}
E = dict(huecos=list(HUECOS), l0=L0, acc=ACC, trazas=TR, run=RUN, a1=A1, n1=N1, uno=UNO, tr3=TR3, eco=ECOS, prev=PREV)
json.dump(E, open("../embed.json", "w", encoding="utf-8"), separators=(",", ":"))
T = dict(huecos=list(HUECOS), l0=L0, eco=ECOS, acc={t: [[r["prueba"] for r in R if r["tipo"] == t and r["H"] == 16 and r["G"] == G and not r["extra"]]
                                                        for G in HUECOS] for t in ("RNN", "LSTM", "GRU")})
json.dump(T, open("../tembed.json", "w", encoding="utf-8"), separators=(",", ":"))
print("embed.json", os.path.getsize("../embed.json"), "bytes · tembed.json", os.path.getsize("../tembed.json"), "bytes")
