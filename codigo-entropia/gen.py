"""Genera ../embed.json (datos de la versión sencilla) e imprime las cifras que se citan en el texto."""
import json, math
import numpy as np
from sim import *
from entropia import *

out = {}
print("héroe P_ANUAL", P_ANUAL, "H =", round(entropia(P_ANUAL), 4), "sorpresas", np.round(sorpresa(P_ANUAL), 3),
      "Huffman", huffman(P_ANUAL), longitud_media(P_ANUAL, huffman(P_ANUAL)))
print("   uniforme 4: H =", entropia(np.ones(4)))
for p0 in (0.25, 0.5, 0.75, 0.9, 0.99):
    t = min(1, (p0 - 0.25) / 0.65); w = (1 - t) * np.ones(3) + t * np.array([6, 3.5, 0.5]); p = np.r_[p0, (1 - p0) * w / w.sum()]
    print(f"   p0 = {p0}: p = {np.round(p, 4)}  H = {entropia(p):.4f}  Huffman L = {longitud_media(p, huffman(p)):.3f}")

print("idea 1: sorpresa de 1 entre N:", {N: math.log2(N) for N in (2, 4, 16, 64, 1024)})
print("idea 2: moneda: H(0.5) =", entropia([.5, .5]), " H(0.1) =", round(entropia([.1, .9]), 4), " H(0.01) =", round(entropia([.01, .99]), 4))

msgs = 2000 * 2 * 60 * 24 * 30
print("idea 3: mensajes al mes", msgs)
for nom, p in (("normal", P_ESTADOS), ("iguales", np.ones(6) / 6), ("ruta", np.array([0.9, 0.04, 0.03, 0.015, 0.01, 0.005]))):
    c = huffman(p); L = longitud_media(p, c); H = entropia(p)
    print(f"   {nom}: H = {H:.4f}  L = {L:.4f}  códigos {c}  MB/mes fijo {msgs * 3 / 8e6:.1f}  Huffman {msgs * L / 8e6:.1f}  entropía {msgs * H / 8e6:.1f}")

for nom, q in (("invierno", P_INVIERNO), ("anual", P_ANUAL), ("uniforme", np.ones(4) / 4), ("nunca", np.array([0.99, 0.006, 0.003, 0.001]))):
    print(f"idea 4: q = {nom}: H(p) = {entropia(P_INVIERNO):.4f}  H(p,q) = {entropia_cruzada(P_INVIERNO, q):.4f}  KL = {kl(P_INVIERNO, q):.4f}"
          f"  coste por tipo −log2 q = {np.round(sorpresa(q), 3)}")
print("idea 5: KL(inv‖anual) =", round(kl(P_INVIERNO, P_ANUAL), 4), " KL(anual‖inv) =", round(kl(P_ANUAL, P_INVIERNO), 4))
for a in (0.0001, 0.001, 0.005, 0.02, 0.05, 0.1):
    q = np.r_[(1 - a) * P_INVIERNO[:3] / P_INVIERNO[:3].sum(), a]
    print(f"   q(accidente) = {a}: KL(p‖q) = {kl(P_INVIERNO, q):.4f}  KL(q‖p) = {kl(q, P_INVIERNO):.4f}")

print("idea 6: −ln q:", {q: round(-math.log(q), 3) for q in (0.001, 0.01, 0.1, 0.5, 0.9, 0.99)}, " (1−q)²:", {q: round((1 - q) ** 2, 4) for q in (0.01, 0.5)})
print("   en bits −log2 0,01 =", round(-math.log2(0.01), 3), " 1 bit =", round(math.log(2), 4), "nats")
for p in (0.02, 0.1, 0.4):
    for q in (p, 0.5, 0.01):
        print(f"idea 7: p = {p} q = {q}: pérdida esperada {-(p * math.log(q) + (1 - p) * math.log(1 - q)):.4f}  H(p) = {entropia([p, 1 - p], math.e):.4f}  Brier esperado {p * (1 - q) ** 2 + (1 - p) * q**2:.4f}")

z = np.array([2.0, 0.5, -0.5, 0.0]); pr = softmax(z)
print("idea 8: softmax", np.round(pr, 4), " pérdida si es frenos", round(-math.log(pr[1]), 4), " p − y", np.round(pr - np.eye(4)[1], 4))

d = camiones(); y = d["averia"]; Q = {}
qs = [("km", "¿Más de 400 000 km?", d["km"] > 400_000), ("urbana", "¿Hace ruta urbana?", d["urbana"]),
      ("par", "¿Matrícula par?", d["par"]), ("t95", "¿Motor a más de 95 °C?", d["temp"] > 95),
      ("fuera", "¿Motor fuera de 80–100 °C?", (d["temp"] < 80) | (d["temp"] > 100))]
print("idea 9: averías", y.sum(), "de", len(y), " H =", round(entropia_etiquetas(y), 4))
out["arbol"] = []
for k, txt, m in qs:
    g = ganancia_informacion(y, m)
    ent = [entropia_etiquetas(y[m]), entropia_etiquetas(y[~m])]
    out["arbol"].append(dict(k=k, q=txt, si=[int(m.sum()), int(y[m].sum())], no=[int((~m).sum()), int(y[~m].sum())]))
    print(f"   {txt}: sí {m.sum()} ({y[m].sum()} averías, H = {ent[0]:.4f})  no {(~m).sum()} ({y[~m].sum()} averías, H = {ent[1]:.4f})  ganancia {g:.4f}")
bins = np.arange(62, 122, 4); idx = np.digitize(d["temp"], bins)
tb = []
for i in range(1, len(bins)):
    m = idx == i
    if m.sum(): tb.append([float(bins[i - 1] + 2), int(m.sum()), int(y[m].sum())])
out["temp"] = tb
print("   temperatura por tramos:", tb)
print("   correlación temp–avería:", round(float(np.corrcoef(d["temp"], y)[0, 1]), 4),
      " IM (tramos de 4 °C):", round(informacion_mutua(idx, y, 2), 4), "bits")
from sklearn.feature_selection import mutual_info_classif
print("   mutual_info_classif (nats):", np.round(mutual_info_classif(np.c_[d["temp"], d["km"], d["urbana"], d["par"]], y, discrete_features=[False, False, True, True], random_state=0), 4))
json.dump(out, open("../embed.json", "w"))
