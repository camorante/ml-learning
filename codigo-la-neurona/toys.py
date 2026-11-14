"""Números de los juguetes de la versión sencilla, calculados con la misma aritmética que el
JavaScript de la página (floats de doble precisión, mismo orden de operaciones), y embed.json."""
import json, math
import numpy as np
from sklearn.svm import SVC
from datos import JORNADAS, X, y, W_IDEA, B_IDEA, ETA, nuevas, carga_consumo


def errores(w1, w2, b, pts=JORNADAS):
    return sum(1 for a, c, t in pts if ((w1 * a + w2 * c + b) > 0) != (t == 1))


def perceptron(w1, w2, b, orden=range(20), eta=ETA, traza=False):
    """Igual que la página: recorre las jornadas en orden; fallo si s·z <= 0."""
    pasos, pasadas = [], 0
    while pasadas < 1000:
        pasadas += 1; fallos = 0
        for i in orden:
            a, c, t = JORNADAS[i]; s = 1 if t == 1 else -1
            z = w1 * a + w2 * c + b
            if s * z <= 0:
                w1 += eta * s * a; w2 += eta * s * c; b += eta * s; fallos += 1
                pasos.append((pasadas, i + 1, round(z, 4), round(w1, 4), round(w2, 4), round(b, 4)))
        if fallos == 0:
            break
    return (w1, w2, b), pasos, pasadas


print("idea del jefe (1, 0,2, −6): errores", errores(*W_IDEA, B_IDEA), "de 20")
mal = [i + 1 for i, (a, c, t) in enumerate(JORNADAS) if ((W_IDEA[0] * a + W_IDEA[1] * c + B_IDEA) > 0) != (t == 1)]
print("   jornadas mal clasificadas:", mal)
(w1, w2, b), pasos, pasadas = perceptron(*W_IDEA, B_IDEA)
print(f"perceptrón desde la idea: {len(pasos)} correcciones, {pasadas} pasadas (la última sin fallos)")
print(f"   pesos finales w1 = {w1:.2f}  w2 = {w2:.2f}  b = {b:.2f}   frontera: frenadas + {w2 / w1:.2f}·horas = {-b / w1:.2f}")
print("   fallos por pasada:", [sum(1 for p in pasos if p[0] == k) for k in range(1, pasadas + 1)])
for p in pasos[:4]:
    print("   pasada %d, jornada %d: z = %.2f → w = (%.2f, %.2f), b = %.1f" % p)
WT = (round(w1, 2), round(w2, 2), round(b, 2))

Xn, yn = nuevas()
acc = lambda w1, w2, b: float(np.mean(((w1 * Xn[:, 0] + w2 * Xn[:, 1] + b) > 0) == yn))
print(f"1 000 jornadas nuevas ({yn.sum()} a revisar): idea del jefe {acc(*W_IDEA, B_IDEA):.3f}  entrenada {acc(*WT):.3f}")
svm = SVC(kernel="linear", C=1e6).fit(X, y)
sw, sb = svm.coef_[0], svm.intercept_[0]
print(f"recta del medio (SVM de margen máximo): w = ({sw[0]:.3f}, {sw[1]:.3f}), b = {sb:.3f}"
      f"  → frenadas + {sw[1] / sw[0]:.2f}·horas = {-sb / sw[0]:.2f}  acierto nuevas {acc(*sw, sb):.3f}  errores en las 20: {errores(*sw, sb)}")
acs = []
for s in range(200):
    o = np.random.default_rng(s).permutation(20)
    q, _, _ = perceptron(*W_IDEA, B_IDEA, orden=o)
    acs.append(acc(*q)); assert errores(*q) == 0
print(f"200 órdenes barajados: acierto en nuevas entre {min(acs):.3f} y {max(acs):.3f} (media {np.mean(acs):.3f})")

# sigmoide con los pesos entrenados
z = WT[0] * X[:, 0] + WT[1] * X[:, 1] + WT[2]
p = 1 / (1 + np.exp(-z))
print("puntuaciones z de las 20:", np.round(z, 2).tolist())
print("probabilidades σ(z):", np.round(p, 3).tolist())

# carga y consumo: la mejor recta posible
Cq, cy = carga_consumo()
def mejor_general(P, t, n=7200):
    sc = np.array([22.0, 26.0]); Q = (P - P.mean(0)) / sc; best = (99, None)
    for k in range(n):
        th = math.pi * k / n; u = Q @ np.array([math.cos(th), math.sin(th)])
        o = np.argsort(u); us, ts = u[o], t[o]
        for j in range(len(u) + 1):                      # umbral entre us[j-1] y us[j]
            e1 = ts[:j].sum() + (1 - ts[j:]).sum()          # «no cuadra» a la izquierda… no: a la derecha
            e = min(e1, len(u) - e1)
            if e < best[0]: best = (e, th)
    return best
print("carga-consumo: viajes", len(cy), "no cuadran", cy.sum(), " mejor recta posible: errores", mejor_general(Cq, cy)[0])
def mejor_familia(P, t):
    best = (99,)
    for m in np.arange(0, 1.2001, 0.01):
        for a in np.arange(10, 40.0001, 0.1):
            r = P[:, 1] - (a + m * P[:, 0])
            for lado in (1, -1):
                e = int(np.sum((lado * r > 0) != (t == 1)))
                if e < best[0]: best = (e, round(m, 2), round(a, 1), lado)
    return best
print("   mejor recta con pendiente 0–1,2 y altura 10–40:", mejor_familia(Cq, cy))
fr = (Cq[:, 1] - (24 + 0.55 * Cq[:, 0]))
print("   franja 24 + 0,55·carga ± 2,5: errores", int(np.sum((np.abs(fr) > 2.5) != (cy == 1))))

emb = {"J": [list(j) for j in JORNADAS], "WT": WT, "svm": [round(float(sw[0]), 4), round(float(sw[1]), 4), round(float(sb), 4)],
       "nf": Xn[:, 0].round(1).tolist(), "nr": Xn[:, 1].round(1).tolist(), "ny": yn.tolist(),
       "cc": Cq[:, 0].tolist(), "cq": Cq[:, 1].tolist(), "cy": cy.tolist()}
json.dump(emb, open("../embed.json", "w"), separators=(",", ":"))
print("embed.json:", len(json.dumps(emb, separators=(",", ":"))), "bytes")
