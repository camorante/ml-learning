"""Datos de los dibujos: ../embed.json (sencilla) y ../tembed.json (técnica). Antes hay que ejecutar detector.py,
tamano.py, curva.py, mapa.py y denoising.py. Imprime las cifras que citan las páginas, calculadas con los mismos
números redondeados que reciben los dibujos (así página y Python cuentan igual)."""
import json, os
import numpy as np
import torch
from datos import X_TE, Y_TE, TIPOS, FALSA, PERDIDA
from modelos import AE, reconstruir

os.chdir(os.path.dirname(os.path.abspath(__file__)))
ld = lambda f: json.load(open(f, encoding="utf-8"))
det, tam, cur, mp, dn = ld("detector.json"), ld("tamano.json"), ld("curva.json"), ld("mapa.json"), ld("denoising.json")
e_va = np.round(det["e_va"], 6)
e_te = np.round(det["e_te"], 6)
net = AE(4); net.load_state_dict(torch.load("detector.pt"))


def cuantil(a, q):
    """Como np.quantile (interpolación lineal), escrito con un bucle para copiarlo tal cual en JS."""
    s = np.sort(a); pos = (len(s) - 1) * q; i = int(np.floor(pos)); fr = pos - i
    return float(s[i] + (s[min(i + 1, len(s) - 1)] - s[i]) * fr)


def cuenta(u):
    al = e_te > u
    fp, vp = int((al & (Y_TE == 0)).sum()), int((al & (Y_TE > 0)).sum())
    return fp, vp, 200 - vp, fp * FALSA + (200 - vp) * PERDIDA


# el juguete: 4 ventanas de cada tipo, elegidas por su error dentro del tipo (la más baja, a veces, se escapa)
ej = {}
for k in range(5):
    idx = np.flatnonzero(Y_TE == k)
    orden = idx[np.argsort(e_te[idx])]
    qs = [0.2, 0.5, 0.8, 1.0] if k == 0 else [0.0, 0.3, 0.6, 0.9]
    ej[k] = [int(orden[min(int(q * (len(orden) - 1)), len(orden) - 1)]) for q in qs]
print("ventanas del juguete (índice de prueba: error):")
for k in range(5):
    print(f"  {TIPOS[k]:10s} " + " · ".join(f"{i}: {e_te[i]:.4f}" for i in ej[k]))
todas = sorted(set(sum(ej.values(), [])))
R = reconstruir(net, X_TE[todas])
ventanas = {str(i): {"x": X_TE[i].round(3).tolist(), "r": R[j].round(3).tolist(), "e": float(e_te[i]), "y": int(Y_TE[i])}
            for j, i in enumerate(todas)}

u99 = cuantil(e_va, 0.99)
print(f"\numbral percentil 99 (de los errores redondeados a 6 decimales): {u99:.6f}")
GRID = [round(0.024 + 0.001 * i, 3) for i in range(57)]           # umbrales del deslizador: 0,024 … 0,080
print("umbral · % de normales de validación por debajo · falsas alarmas · cazadas · se escapan · coste €")
costes = []
for u in GRID:
    fp, vp, fn, c = cuenta(u)
    costes.append(c)
    if u in (0.030, 0.032, 0.034, 0.037, 0.044, 0.050, 0.060, 0.080):
        print(f"  {u:.3f} · {np.mean(e_va <= u):6.1%} · {fp:4d} · {vp} · {fn:3d} · {c} €")
b = int(np.argmin(costes))
print(f"  el más barato de la rejilla: {GRID[b]:.3f} → {cuenta(GRID[b])}")
for q in (0.95, 0.99, 0.999):
    u = cuantil(e_va, q); fp, vp, fn, c = cuenta(u)
    prec, rec = vp / (vp + fp), vp / 200
    print(f"percentil {q * 100:g}: umbral {u:.4f} · falsas {fp} · cazadas {vp} · escapan {fn} · precisión {prec:.3f}"
          f" · recall {rec:.3f} · F1 {2 * prec * rec / (prec + rec):.3f} · {c} €")
print("por tipo con el percentil 99:", {TIPOS[k]: int((e_te[Y_TE == k] > u99).sum()) for k in range(5)})

# el mapa: 100 normales y las 200 averías llevan su ventana para dibujarla al tocar el punto
idx = np.array(mp["idx"]); y = np.array(mp["y"])
con = set(np.r_[idx[y == 0][:100], idx[y > 0]].tolist())
mapa = {"z": mp["z"], "y": mp["y"], "e": mp["e"], "u": round(mp["u_e"], 6), "dec": mp["dec"],
        "x": {str(i): X_TE[i].round(2).tolist() for i in idx if int(i) in con}, "idx": mp["idx"]}

D = {"tipos": TIPOS, "falsa": FALSA, "perdida": PERDIDA, "ej": {str(k): v for k, v in ej.items()}, "win": ventanas,
     "e_va": e_va.tolist(), "e_te": e_te.tolist(), "y_te": Y_TE.tolist(), "grid": GRID,
     "tam": {"res": tam["res"], "ej": tam["ej"], "rec": tam["rec"], "x": X_TE[tam["ej"]].round(3).tolist()},
     "curva": cur, "mapa": mapa, "dn": dn}
json.dump(D, open("../embed.json", "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
T = {"tipos": TIPOS, "falsa": FALSA, "perdida": PERDIDA, "e_va": e_va.tolist(), "e_te": e_te.tolist(), "y_te": Y_TE.tolist()}
json.dump(T, open("../tembed.json", "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
print(f"\nembed.json {os.path.getsize('../embed.json')} bytes · tembed.json {os.path.getsize('../tembed.json')} bytes")
