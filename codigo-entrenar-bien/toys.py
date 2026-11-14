"""Junta los entrenamientos reales en ../embed.json (sencilla) y ../tembed.json (técnica) e imprime las cifras
que citan las páginas. Antes hay que ejecutar comparar, escala, programas, sobreajuste, lotes y bucle."""
import json, os
import numpy as np
from datos import X_TR, Y_TR, X_VA, Y_VA, MEDIA, DESV, RPM0, RPM_S, TMP0, TMP_S, BORROSO
from entrenar import epoca_meta

os.chdir(os.path.dirname(os.path.abspath(__file__)))
ld = lambda f: json.load(open(f, encoding="utf-8"))
R, ESC, PRO, SOB, LOT, BUC = (ld(f) for f in ("rejilla.json", "escala.json", "programas.json", "sobreajuste.json",
                                                 "lotes.json", "bucle.json"))
r3 = lambda v: [round(x, 3) for x in v]
corrida = lambda o, lr, b: next(r for r in R if r["opt"] == o and r["lr"] == lr and r["lote"] == b)

# juguete: 3 optimizadores × 8 tasas × 3 lotes
toy = {f"{r['opt']}|{r['lr']:g}|{r['lote']}": dict(tr=r3(r["tr"]), va=r3(r["va"]), acc=r["acc"], P=r["P"], pasos=r["pasos"])
       for r in R if r["opt"] != "rmsprop"}
d = corrida("adam", 0.1, 32)
print(f"juguete por defecto (Adam, 0.1, lotes de 32): validación {d['va'][-1]:.3f}, acierto {d['acc']:.1%},"
      f" baja de 0.25 en la época {epoca_meta(d['va'], 0.25)}, {d['pasos']} pasos")
for o, lr, b in [("sgd", 0.1, 32), ("sgd", 1.0, 32), ("adam", 0.001, 32), ("adam", 3.0, 32), ("adam", 0.1, 600),
                 ("adam", 0.1, 1), ("sgd", 1.0, 600), ("momento", 1.0, 1), ("adam", 0.01, 1), ("sgd", 0.03, 1)]:
    r = corrida(o, lr, b)
    print(f"   {o:8s} {lr:<5g} lote {b:3d}: " + (f"validación {r['va'][-1]:.3f}  acierto {r['acc']:.1%}  meta en {epoca_meta(r['va'], 0.25)}"
                                               if len(r["va"]) == 61 else f"se dispara en la época {len(r['va']) - 1}"))

tasa = {"0.001": r3(corrida("momento", 0.001, 32)["va"]), "0.01": r3(corrida("momento", 0.01, 32)["va"]),
        "0.1": r3(PRO["constante 0.1"]["va"]), "1": r3(PRO["constante 1"]["va"]), "10": r3(PRO["constante 10"]["va"]),
        "coseno": r3(PRO["coseno desde 1"]["va"]), "calent": r3(PRO["calentamiento + coseno desde 1"]["va"])}
curvas = {"buena": corrida("adam", 0.1, 32), "lenta": corrida("adam", 0.001, 32), "brusca": corrida("adam", 3.0, 32)}
curvas = {k: dict(tr=r3(v["tr"]), va=r3(v["va"])) for k, v in curvas.items()}
curvas["sobre"] = dict(tr=r3(SOB["tr"][::5]), va=r3(SOB["va"][::5]), mejor=SOB["mejor"])
escala = {k: dict(tr=r3(v["tr"]), va=r3(v["va"]), lr=v["lr"], acc=v["acc"]) for k, v in ESC.items()}
print(f"sin escalar, la red dice «normal» a todo: {int((1 - Y_VA).sum())} de {len(Y_VA)} aciertos en validación")
p0 = Y_TR.mean(); base = float(-np.mean(Y_VA * np.log(p0) + (1 - Y_VA) * np.log(1 - p0)))
print(f"decir siempre p = {p0:.3f} (la proporción de anomalías del entrenamiento): pérdida de validación {base:.4f}")
print(f"rangos: rpm {X_TR[:, 0].min():.0f}–{X_TR[:, 0].max():.0f}, °C {X_TR[:, 1].min():.1f}–{X_TR[:, 1].max():.1f};"
      f" validación al empezar {d['va'][0]:.4f}")

E = dict(xr=X_TR[:, 0].tolist(), xt=X_TR[:, 1].tolist(), y=Y_TR.tolist(), media=MEDIA.round(6).tolist(),
         desv=DESV.round(6).tolist(), elipse=[RPM0, RPM_S, TMP0, TMP_S, BORROSO], toy=toy, bucle=BUC,
         lotes=LOT, tasa=tasa, base=round(base, 4), curvas=curvas, escala=escala)
json.dump(E, open("../embed.json", "w", encoding="utf-8"), separators=(",", ":"))
T = dict(runs={f"{r['opt']}|{r['lr']:g}": dict(tr=r3(r["tr"]), va=r3(r["va"])) for r in R if r["lote"] == 32})
json.dump(T, open("../tembed.json", "w", encoding="utf-8"), separators=(",", ":"))
print("embed.json", os.path.getsize("../embed.json"), "bytes · tembed.json", os.path.getsize("../tembed.json"), "bytes")
