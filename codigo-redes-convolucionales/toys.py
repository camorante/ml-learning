"""Datos de los dibujos: ../embed.json (sencilla) y ../tembed.json (técnica). Imprime las cifras que citan las
páginas. Antes hay que ejecutar comparar.py (guarda las redes entrenadas)."""
import json, os
import numpy as np
import torch
from datos import X_VA, Y_VA, LIMPIA_VA, INI_VA, L, PRINCIPIO, media5
from modelos import CNN, Densa, UnFiltro, prob

os.chdir(os.path.dirname(os.path.abspath(__file__)))


def cargar(M, k):
    m = M(); m.load_state_dict(torch.load(f"{M.__name__}_{k}.pt")); m.eval(); return m


r2 = lambda a: [round(float(v), 2) for v in np.ravel(a)]


def suma(w, x, i):
    """Una casilla del mapa: suma de productos sin sesgo, acumulada en el mismo orden que el JS
    (el sum() de Python 3.12 compensa el redondeo y daría otro último decimal)."""
    a = 0.0
    for k in range(len(w)):
        a += w[k] * x[i + k]
    return a


def respuestas(w, X):
    """El máximo del mapa de rasgos del juguete en cada ventana."""
    return [max(suma(w, x, i) for i in range(len(x) - len(w) + 1)) for x in X]


def mejor_umbral(r, y):
    """El umbral que más ventanas acierta con «alerta si el máximo pasa del umbral» (igual que el JS)."""
    v = sorted(set(r)); cands = [v[0] - 1] + [(a + b) / 2 for a, b in zip(v, v[1:])]
    best, bt = -1, None
    for t in cands:
        ok = sum((ri > t) == bool(yi) for ri, yi in zip(r, y))
        if ok > best:
            best, bt = ok, t
    return bt, best


# ---------- el juguete: el filtro de 5 pesos que aprendió la red de un filtro ----------
uf = cargar(UnFiltro, "A")
W_APR = r2(uf.c.weight.data.numpy())
VAL = [r2(x) for x in X_VA[:200]]; YV = Y_VA[:200].tolist()
PRESETS = {"aprendido": W_APR, "media": [-0.2] * 5, "instante": [0, 0, -1, 0, 0], "flanco": [0.5, 0.5, 0, -0.5, -0.5],
           "alreves": [0.4] * 5}
print(f"filtro aprendido (semilla 1): {np.round(uf.c.weight.data.numpy().ravel(), 4)}, sesgo {uf.c.bias.item():.4f},"
      f" cabeza a = {uf.out.weight.item():.4f}, c = {uf.out.bias.item():.4f}")
print(f"umbral de la red: máximo + sesgo > −c/a  →  máximo > {-uf.out.bias.item() / uf.out.weight.item() - uf.c.bias.item():.3f}")
print(f"200 ventanas de validación para el juguete: {sum(YV)} bruscas")
UMB = {}
for k, w in PRESETS.items():
    t, ok = mejor_umbral(respuestas(w, VAL), YV); UMB[k] = t
    print(f"   filtro {k:9s} {w}: mejor umbral {t:.4f}, acierta {ok} de 200 ({ok / 200:.1%})")


def tipo(i):
    lim = LIMPIA_VA[i]; bache = np.abs(np.diff(lim)).max(); m5 = media5(lim).min()
    return bache, m5, lim.max()


def primero(cond):
    return next(i for i in range(200, len(Y_VA)) if cond(i))


EJ = {"brusca": primero(lambda i: Y_VA[i] and tipo(i)[1] < -4 and tipo(i)[0] < 2 and 18 <= INI_VA[i] <= 34 and tipo(i)[2] < 0.6),
      "bache": primero(lambda i: not Y_VA[i] and tipo(i)[0] > 6 and X_VA[i].min() < -3 and tipo(i)[1] > -2.2),
      "suave": primero(lambda i: not Y_VA[i] and tipo(i)[0] < 2 and -2.6 < tipo(i)[1] < -2.0 and tipo(i)[2] < 0.6),
      "acelera": primero(lambda i: Y_VA[i] and tipo(i)[2] > 1.3 and tipo(i)[0] < 2 and -3.8 < tipo(i)[1])}
TR = {k: r2(X_VA[i]) for k, i in EJ.items()}
w = W_APR
for k, x in TR.items():
    s = [suma(w, x, i) for i in range(L - 4)]
    print(f"ventana «{k}» (validación nº {EJ[k]}, brusca = {Y_VA[EJ[k]]}): máximo del mapa {max(s):.2f} en la posición {int(np.argmax(s))}"
          f" (= {np.argmax(s) / 10:.1f} s), mínima lectura {min(x):.2f}; filtro de un instante: {max(-v for v in x):.2f}")

# ---------- idea 1: la misma frenada, movida a cada posición ----------
rng = np.random.default_rng(7)
fondo = 0.25 * np.sin(2 * np.pi * (np.arange(L) / 80 + 0.3)) + rng.normal(0, 0.35, L)
FRENADAS = {"brusca": np.r_[-4.0 * np.arange(1, 4) / 3, -4.0 * np.ones(6), -4.0 * np.arange(3, -1, -1) / 4],
            "suave": np.r_[-2.0 * np.arange(1, 4) / 3, -2.0 * np.ones(6), -2.0 * np.arange(3, -1, -1) / 4]}
MODS = {f"{n}_{k}": cargar(M, k) for n, M in (("cnn", CNN), ("densa", Densa)) for k in "AB"}
MOVER = {"fondo": r2(fondo), "frenadas": {k: r2(f) for k, f in FRENADAS.items()}, "p": {}}
for fk, f in FRENADAS.items():
    Xs = np.array([np.r_[fondo[:s], fondo[s:s + len(f)] + f, fondo[s + len(f):]] for s in range(L - len(f) + 1)])
    for mk, m in MODS.items():
        MOVER["p"][f"{fk}|{mk}"] = [round(float(v), 4) for v in prob(m, Xs)]
pb = MOVER["p"]
for mk in MODS:
    p = np.array(pb[f"brusca|{mk}"]); q = np.array(pb[f"suave|{mk}"])
    zona = np.arange(len(p)); dentro = (zona >= PRINCIPIO[0]) & (zona <= PRINCIPIO[1])
    print(f"mover la frenada · {mk:8s}: brusca p mín {p.min():.3f} máx {p.max():.3f} (en la zona del principio {p[dentro].min():.3f}–{p[dentro].max():.3f},"
          f" fuera {p[~dentro].min():.3f}–{p[~dentro].max():.3f});  suave p máx {q.max():.3f}")
print(f"posiciones de inicio: 0–{L - len(FRENADAS['brusca'])} (la frenada dura {len(FRENADAS['brusca'])} lecturas)")

# ---------- idea 3: los 4 filtros de la primera capa de la red convolucional ----------
cnn = MODS["cnn_A"]
F1 = dict(w=[r2(v) for v in cnn.c1.weight.data.numpy()[:, 0]], b=r2(cnn.c1.bias.data.numpy()))
with torch.no_grad():
    for k, i in EJ.items():
        mp = cnn.mapas(torch.tensor(X_VA[i:i + 1]))[0].numpy()
        print(f"capa 1 sobre «{k}»: máximo de cada mapa {np.round(mp.max(1), 2)}")
print("filtros de la capa 1:", F1)

E = dict(val=VAL, yv=YV, presets=PRESETS, umbral={k: round(v, 4) for k, v in UMB.items()}, trazas=TR,
         ytrazas={k: int(Y_VA[i]) for k, i in EJ.items()}, mover=MOVER, f1=F1,
         principio=list(PRINCIPIO))
json.dump(E, open("../embed.json", "w", encoding="utf-8"), separators=(",", ":"))
T = dict(mover=MOVER, principio=list(PRINCIPIO))
json.dump(T, open("../tembed.json", "w", encoding="utf-8"), separators=(",", ":"))
print("embed.json", os.path.getsize("../embed.json"), "bytes · tembed.json", os.path.getsize("../tembed.json"), "bytes")
