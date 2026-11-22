"""Datos de los dibujos: ../embed.json (sencilla) y ../tembed.json (técnica). Antes hay que ejecutar mapa.py,
dimensiones.py y tramos.py. Imprime las cifras que citan las páginas."""
import json, os
import numpy as np
from datos import V, NOMBRES, DESC, SISTEMAS, SISTEMA, PERT, VIS_TR, P_TR, P_TE
from modelos import coseno, vecinos, pureza

os.chdir(os.path.dirname(os.path.abspath(__file__)))
mp = json.load(open("mapa.json", encoding="utf-8"))
dm = json.load(open("dimensiones.json", encoding="utf-8"))
tr = json.load(open("tramos.json", encoding="utf-8"))

fotos = mp["fotos"]
hist = {h[0]: h for h in mp["hist"]}
E = np.array(mp["E"]["60"])
S = coseno(E)
print("juguete (d = 2): pérdida en prueba y aciertos top-3 por foto")
for ep in fotos:
    print(f"  época {ep:2d}: pérdida {hist[ep][3]:.3f}, top-3 {hist[ep][4]:.1%}, vecinos del mismo sistema {pureza(np.array(mp['E'][str(ep)]), SISTEMA)[0]} de 60")
for c in ("P0217", "P0401", "U0100", "U0121", "C0110"):
    i = NOMBRES.index(c)
    print(f"  vecinos de {c}: " + ", ".join(f"{NOMBRES[j]} {S[i, j]:.3f}" for j in vecinos(S, i)))
i, j = NOMBRES.index("P0217"), NOMBRES.index("C0035")
print(f"  cos(P0217, C0035) = {S[i, j]:.3f} · distancia one-hot entre dos códigos cualesquiera = √2 = {np.sqrt(2):.3f}, coseno 0")
print(f"  visitas de ejemplo: {[[NOMBRES[c] for c in v] for v in VIS_TR[:6]]}")
n_vist = np.bincount(P_TR[:, 0], minlength=V)

Et = np.array(tr["E"])
print("tramos (embedding con 4 decimales, como en el dibujo): flecha de día a noche")
for k, z in enumerate(tr["zonas"]):
    v = Et[k + 4] - Et[k]
    print(f"  {z:9s} largo {np.hypot(*v):.2f}  ángulo {np.degrees(np.arctan2(v[1], v[0])):.1f}°"
          f"  ·  predice de día {tr['pred'][k][0]:.1f} L/100 km y {tr['pred'][k][1]:.1f} min/h; de noche {tr['pred'][k + 4][0]:.1f} y {tr['pred'][k + 4][1]:.1f}")
q = Et[3] + Et[4] - Et[0]
print("  autopista día + (centro noche − centro día): distancias " + ", ".join(f"{d:.2f}" for d in np.linalg.norm(Et - q, axis=1)))
res = dm["res"]
print("tamaño de la tabla y de la primera capa con 64 neuronas:")
for n in (20, 500, 5000, 50000):
    print(f"  {n} categorías: one-hot + capa de 64 → {n * 64 + 64} pesos · embedding de 8 + capa de 64 → {n * 8 + 8 * 64 + 64}")

D = {"cod": NOMBRES, "desc": DESC, "sis": SISTEMAS, "s": SISTEMA.tolist(), "pert": PERT.tolist(),
     "fotos": fotos, "E": mp["E"], "hist": {str(ep): [hist[ep][3], hist[ep][4]] for ep in fotos},
     "vis": VIS_TR[:6], "nvis": len(VIS_TR), "npar": len(P_TR), "npte": len(P_TE), "nv": n_vist.tolist(),
     "tramos": tr, "dims": {k: {"param": v["param"], "te": v["te"][0],
                                 "pureza": v["pureza"][0]} for k, v in res.items()},
     "ref": dm["ref"], "azar": round(float(np.log(19)), 4), "E16": dm["E"]["16"], "pca16": dm["pca16"], "var16": dm["var16"], "km16": dm["km16"]}
json.dump(D, open("../embed.json", "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))

# técnica: la proyección PCA del embedding de cada dimensión (semilla 0)
proj = {}
for k, Ek in dm["E"].items():
    Z = np.array(Ek); Z = Z - Z.mean(0)
    if Z.shape[1] == 1:
        proj[k] = {"xy": np.c_[Z, np.zeros(V)].round(4).tolist(), "var": [1.0, 0.0]}
        continue
    U_, s, Vt = np.linalg.svd(Z, full_matrices=False)
    P = Z @ Vt[:2].T
    proj[k] = {"xy": P.round(4).tolist(), "var": [round(float(v), 4) for v in (s ** 2 / (s ** 2).sum())[:2]]}
    print(f"  d = {k}: PCA explica {proj[k]['var'][0] + proj[k]['var'][1]:.1%}")
T = {"cod": NOMBRES, "sis": SISTEMAS, "s": SISTEMA.tolist(), "proj": proj,
     "dims": {k: {"param": v["param"], "te": v["te"], "top3": v["top3"], "pureza": v["pureza"]} for k, v in res.items()},
     "ref": dm["ref"]}
json.dump(T, open("../tembed.json", "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
print(f"embed.json {os.path.getsize('../embed.json')} bytes · tembed.json {os.path.getsize('../tembed.json')} bytes")
