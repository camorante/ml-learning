"""Números de los juguetes de la versión sencilla (misma aritmética que el JavaScript), embed.json y tembed.json."""
import json
import numpy as np
from datos import carga_consumo, red_inicial, ETA, W1_0, B1_0, W2_0, B2_0
from retro import diferencias_centrales, gradiente_vector

C, y = carga_consumo()
r1 = json.load(open("../../r1/embed.json", encoding="utf-8"))
assert r1["cc"] == C[:, 0].tolist() and r1["cq"] == C[:, 1].tolist() and r1["cy"] == y.tolist()
print("viajes idénticos a los de los temas 1 y 3:", len(y), "viajes,", y.sum(), "no cuadran")

red = red_inicial()
z = red.adelante(C)[:, 0]; p = 1 / (1 + np.exp(-z))
print(f"red de partida: pérdida media {red.perdida_media(C, y):.4f}, errores {int(np.sum((p > .5) != y))};"
      f" viajes que se escapan: {[i + 1 for i in range(30) if y[i] and p[i] <= .5]}")

# idea 1: empujar cada peso a mano (diferencia hacia delante) frente a la retropropagación
g = gradiente_vector(red, C, y)
nom = ["wA·carga", "wA·consumo", "wB·carga", "wB·consumo", "bA", "bB", "vA", "vB", "c"]   # orden de MLP.vector()
L0 = red.perdida_media(C, y)
v0 = red.vector()
for i, h in [(3, 0.01), (3, 0.1), (5, 0.1), (1, 0.1)]:
    e = np.zeros(9); e[i] = h; red.cargar(v0 + e); L1 = red.perdida_media(C, y); red.cargar(v0)
    print(f"  empujar {nom[i]} en {h}: pérdida {L0:.4f} → {L1:.4f} (cambio {L1 - L0:+.4f}); gradiente × empujón = {g[i] * h:+.4f}")
print("  gradientes (orden W1 por filas, b1, w2, b2):", np.round(g, 5).tolist())
print(f"  diferencias centrales frente a retropropagación: máx. {np.max(np.abs(g - diferencias_centrales(red, C, y))):.1e}")

# idea 3: sensibilidad de la pérdida al consumo medido, por los dos caminos (viaje 3, consumo variable)
for qq in (20.0, 22.6, 27.0, 33.0):
    W, b = np.array(W1_0), np.array(B1_0); x = np.array([C[2, 0], qq])
    h = 1 / (1 + np.exp(-(W @ x + b))); zz = 10 * h.sum() - 5; pp = 1 / (1 + np.exp(-zz))
    d = (pp - 1) * 10 * h * (1 - h)
    print(f"  viaje 3 con consumo {qq}: camino A {d[0] * W[0, 1]:+.5f}, camino B {d[1] * W[1, 1]:+.5f}, suma {d @ W[:, 1]:+.5f}")

# idea 6: varios pasos con los 30 viajes y tres tasas
pasos = {}
for eta in (0.01, 0.03, 0.1):
    r = red_inicial(); fila = []
    for s in range(13):
        zz = r.adelante(C)[:, 0]
        fila.append((round(float(r.perdida_media(C, y)), 4), int(np.sum((zz > 0) != y))))
        r.paso(C, y, eta)
    pasos[eta] = fila
    print(f"  η = {eta}: (pérdida, errores) en los pasos 0–6: {fila[:7]}")

coste = json.load(open("coste.json", encoding="utf-8"))
capas = json.load(open("capas.json", encoding="utf-8"))
emb = {"cc": C[:, 0].tolist(), "cq": C[:, 1].tolist(), "cy": y.tolist(),
       "W1": W1_0, "b1": B1_0, "w2": W2_0[0], "b2": B2_0[0], "eta": ETA,
       "coste": [{"P": f["P"], "tp": round(f["tp"], 7), "tb": round(f["tb"], 7), "tf": round(f["tf"], 4), "est": f["est"]} for f in coste],
       "capas": {k: [round(x, 9) for x in v["g"]] for k, v in capas.items()},
       "pend": {k: round(float(np.mean(v["pend"])), 3) for k, v in capas.items()}}
json.dump(emb, open("../embed.json", "w", encoding="utf-8"), separators=(",", ":"))
json.dump({"cc": emb["cc"], "cq": emb["cq"], "cy": emb["cy"]}, open("../tembed.json", "w", encoding="utf-8"), separators=(",", ":"))
print("embed.json:", len(json.dumps(emb, separators=(",", ":"))), "bytes")
