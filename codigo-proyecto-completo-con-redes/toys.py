"""Datos de los dibujos (embed.json para la sencilla, tembed.json para la técnica), todos de las ejecuciones reales.
Comprueba que las probabilidades redondeadas a 6 decimales dan los mismos recuentos que las originales en todos
los umbrales del deslizador, e imprime las cifras que citan los textos."""
import json, os
import numpy as np
from comun import trozos
from metricas import pr_auc, coste, matriz, bootstrap, UMBRAL
from datos import MODELOS, DEPOS

os.chdir(os.path.dirname(os.path.abspath(__file__)))
D, F, E, _ = trozos()
P = np.load("probs.npz")
C = json.load(open("comparar.json", encoding="utf-8"))
EV = json.load(open("evaluar.json", encoding="utf-8"))
FU = json.load(open("fuga.json", encoding="utf-8"))
EJ = json.load(open("ejemplo.json", encoding="utf-8"))
te = F["te"]; y = te["y"].astype(int)
MOD = {"red": P["red completa|0"], "boosting": P["boosting"], "logistica": P["logistica"]}
R5 = {k: np.round(v, 6) for k, v in MOD.items()}
U = np.round(np.arange(0.01, 0.5001, 0.005), 3)
dif = sum(matriz(y, MOD[k], u) != matriz(y, R5[k], u) for k in MOD for u in U)
print("umbrales donde el redondeo cambia la matriz de confusión:", dif)

# intervalos bootstrap (camiones enteros, 1 000 réplicas) de cada modelo
ci = {}
for k, p in list(R5.items()) + [("nunca", 0 * y + 0.0)]:
    a = bootstrap(te["cam"], y, p, lambda yy, pp: coste(yy, pp), B=1000, seed=2)
    b = bootstrap(te["cam"], y, p, lambda yy, pp: pr_auc(yy, pp), B=1000, seed=2) if k != "nunca" else a * 0
    ci[k] = [round(float(np.percentile(a, 2.5)), 1), round(float(np.percentile(a, 97.5)), 1),
             round(float(np.percentile(b, 2.5)), 3), round(float(np.percentile(b, 97.5)), 3)]
    print(f"{k:10s} € {coste(y, p):.1f} IC [{ci[k][0]}, {ci[k][1]}]   PR-AUC {pr_auc(y, p) if k != 'nunca' else 0:.3f} IC [{ci[k][2]}, {ci[k][3]}]")

# histograma del bootstrap de la diferencia de coste boosting − red
hists = {}
for k, nm, ancho in [("eur", "€ boosting − red", 2.0), ("ap", "PR-AUC red − boosting", 0.005)]:
    d = np.array(EV["bt"][nm])
    h0 = np.floor(d.min() / ancho) * ancho
    hists[k] = dict(h0=float(h0), w=ancho, n=np.bincount(((d - h0) // ancho).astype(int)).tolist(),
                    lo=float(np.percentile(d, 2.5)), hi=float(np.percentile(d, 97.5)), pos=float(np.mean(d > 0)), med=float(np.median(d)))
    print(f"{nm}: IC [{np.percentile(d, 2.5):.3f}, {np.percentile(d, 97.5):.3f}], mediana {np.median(d):.3f}, réplicas > 0 {np.mean(d > 0):.1%}")

# ejemplos de camiones para la idea 3
pr = MOD["red"]; mx = te["X"][:, :, 0].max(1)
subeT = te["X"][:, 21:, 1].mean(1) - te["X"][:, :21, 1].mean(1)
def primero(mask):
    return int(np.flatnonzero(mask)[0])
sel = [("El camión del ejemplo resuelto", EJ["i"]),
       ("Baches, pero sano", primero((y == 0) & (pr > UMBRAL) & (mx > np.percentile(mx, 98)))),
       ("Se calienta poco a poco", primero((y == 1) & (te["tipo"] == 1) & (te["vis"] == 1) & (pr > UMBRAL) & (subeT > 1.9) & (te["tab"][:, 4] <= 1) & (np.arange(len(y)) != EJ["i"]))),
       ("Avería sin aviso", primero((y == 1) & (te["vis"] == 0) & (pr < 0.06))),
       ("Un mes tranquilo", primero((y == 0) & (pr < 0.04) & (te["tab"][:, 4] == 0)))]
ej = []
for nm, i in sel:
    ej.append(dict(nm=nm, X=np.round(te["X"][i], 2).tolist(), tab=np.round(te["tab"][i], 2).tolist(), mod=MODELOS[te["mod"][i]],
                   dep=DEPOS[te["dep"][i]], y=int(y[i]), vis=int(te["vis"][i]), tipo=int(te["tipo"][i]),
                   pr=round(float(R5["red"][i]), 6), pb=round(float(R5["boosting"][i]), 6), cam=int(te["cam"][i])))
    print(f"{nm:32s} fila {i:4d} camión {te['cam'][i]} y={y[i]} vis={te['vis'][i]} tipo={te['tipo'][i]} p red {pr[i]:.3f} boosting {MOD['boosting'][i]:.3f}"
          f" vib max {mx[i]:.2f} sube temp {subeT[i]:.2f} averías previas {te['tab'][i, 4]:.0f}")

curvas = {k: C["curvas"][k] for k in ("red completa|0", "red completa|1", "red completa|2", "sin pesos de clase|0")}
for k, v in curvas.items():
    tr = [c[0] for c in v["curva"]]; va = [c[1] for c in v["curva"]]
    print(f"{k:22s} épocas {len(va)}, mejor {v['mejor']} (val {va[v['mejor'] - 1]:.4f}, entr. {tr[v['mejor'] - 1]:.4f}); última val {va[-1]:.4f} entr. {tr[-1]:.4f}")
tabla = [dict(n=n, m=ms, v=vs) for n, ms, vs in C["filas"]]
emb = dict(y=y.tolist(), p={k: v.tolist() for k, v in R5.items()}, vis=te["vis"].tolist(), tipo=te["tipo"].tolist(),
           fuga=FU, tabla=tabla, ci=ci, hist=hists, curvas=curvas, fiab=EV["fiab"], ej=ej, w=EV["w"],
           dep=te["dep"].tolist(), mod=te["mod"].tolist())
json.dump(emb, open("../embed.json", "w", encoding="utf-8"), separators=(",", ":"), ensure_ascii=False)
json.dump(dict(y=y.tolist(), p={k: v.tolist() for k, v in R5.items()}, ci=ci), open("../tembed.json", "w", encoding="utf-8"), separators=(",", ":"))
print("embed", os.path.getsize("../embed.json") // 1024, "KB")
