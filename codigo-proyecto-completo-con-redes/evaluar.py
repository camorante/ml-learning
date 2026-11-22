"""El examen, una sola vez: la red elegida (semilla 0) frente a las referencias en los 2 400 fotos del examen.
Umbral por costes, matriz de confusión, intervalos bootstrap por camión, calibración y análisis de errores."""
import json, os
import numpy as np
from comun import trozos
from metricas import pr_auc, roc_auc, coste, matriz, bootstrap, curva_pr, UMBRAL, C_REV, C_AVE

os.chdir(os.path.dirname(os.path.abspath(__file__)))
D, F, E, _ = trozos()
P = np.load("probs.npz")
y, cam = F["te"]["y"], F["te"]["cam"]
red, gb, lr = P["red completa|0"], P["boosting"], P["logistica"]
print(f"examen: {len(y)} fotos de {len(np.unique(cam))} camiones, {int(y.sum())} con avería ({y.mean():.1%})")
print(f"umbral por costes: {C_REV:.0f} / {C_AVE:.0f} = {UMBRAL:.2f}")

# 1. métricas y matriz de confusión en el umbral por costes
for nm, p in [("red", red), ("boosting", gb), ("logística", lr)]:
    m = matriz(y, p)
    print(f"{nm:9s} PR-AUC {pr_auc(y, p):.3f}  ROC {roc_auc(y, p):.3f}  {coste(y, p):6.1f} €/camión·mes  avisos {m['vp'] + m['fp']:4d}"
          f"  VP {m['vp']:3d} FP {m['fp']:4d} FN {m['fn']:3d} VN {m['vn']:4d}  exhaustividad {m['vp'] / (m['vp'] + m['fn']):.1%}"
          f"  precisión {m['vp'] / max(1, m['vp'] + m['fp']):.1%}")
nunca = coste(y, 0 * y)
print(f"nunca avisar: {nunca:.1f} €; la red ahorra {nunca - coste(y, red):.1f} € por camión y mes "
      f"({(nunca - coste(y, red)) / nunca:.1%}); en la flota de 3 000 camiones, {12 * 3000 * (nunca - coste(y, red)):,.0f} € al año")

# 2. ¿y si eligiéramos el umbral mirando el examen? (trampa) frente al elegido con la validación
U = np.round(np.arange(0.01, 0.501, 0.005), 3)
cv = [coste(F["va"]["y"], P["va|0"], u) for u in U]
ct = [coste(y, red, u) for u in U]
print(f"umbral que minimiza el coste en validación: {U[int(np.argmin(cv))]:.3f}; en el examen: {U[int(np.argmin(ct))]:.3f}"
      f" ({min(ct):.1f} €, frente a {coste(y, red):.1f} € con 0,10)")

# 3. intervalos bootstrap del 95 %, remuestreando camiones enteros
B = 2000
ci = lambda a: (np.percentile(a, 2.5), np.percentile(a, 97.5))
bt = {}
for nm, f in [("PR-AUC red", lambda yy, pp: pr_auc(yy, pp[:, 0])),
              ("PR-AUC red − boosting", lambda yy, pp: pr_auc(yy, pp[:, 0]) - pr_auc(yy, pp[:, 1])),
              ("€ red", lambda yy, pp: coste(yy, pp[:, 0])),
              ("€ boosting − red", lambda yy, pp: coste(yy, pp[:, 1]) - coste(yy, pp[:, 0]))]:
    a = bootstrap(cam, y, np.c_[red, gb], f, B=B, seed=1)
    bt[nm] = a
    print(f"{nm:22s} IC 95 %: [{ci(a)[0]:.3f}, {ci(a)[1]:.3f}]   (réplicas > 0: {np.mean(a > 0):.1%})")

# 4. calibración: sin corregir el peso de clase, la red exagera; corregida, cuadra
w = json.load(open("comparar.json", encoding="utf-8"))["curvas"]["red completa|0"]["w"]
cruda = 1 / (1 + (1 - red) / (red * w))                        # deshace la corrección: probabilidad con el peso
bins = np.array([0, .05, .1, .15, .2, .3, .5, 1.0])


def fiab(p):
    k = np.clip(np.digitize(p, bins) - 1, 0, len(bins) - 2)
    return [(float(p[k == j].mean()), float(y[k == j].mean()), int((k == j).sum())) for j in range(len(bins) - 1) if (k == j).any()]


def ece(p):
    return sum(n * abs(a - b) for a, b, n in fiab(p)) / len(p)


print(f"peso de clase w = {w:.3f}; probabilidad media: con el peso {cruda.mean():.3f}, corregida {red.mean():.3f}, real {y.mean():.3f}")
print(f"error de calibración (ECE): con el peso {ece(cruda):.3f}, corregida {ece(red):.3f};  Brier: {np.mean((cruda - y) ** 2):.4f} → {np.mean((red - y) ** 2):.4f}")
print(f"con el peso y umbral 0,10: {coste(y, cruda):.1f} € ({int((cruda > UMBRAL).sum())} avisos)")
for a, b, n in fiab(red):
    print(f"   predicho {a:.3f}  real {b:.3f}  ({n} fotos)")

# 5. análisis de errores de la red en el umbral por costes
av = red > UMBRAL
vis, tipo = F["te"]["vis"], F["te"]["tipo"]
pos = y == 1
print(f"averías con aviso ya visible en los 28 días: {int((vis == 1).sum())}, detectadas {np.mean(av[vis == 1]):.1%};"
      f" sin aviso visible: {int((vis == 0).sum())}, detectadas {np.mean(av[vis == 0]):.1%}")
print(f"de las {int((pos & ~av).sum())} averías que se escapan, {np.mean(vis[pos & ~av] == 0):.1%} no daban ninguna señal")
for t, nm in enumerate(["rodamiento", "refrigeración", "aceite"]):
    s = pos & (tipo == t)
    print(f"   {nm:13s} {int(s.sum()):3d} averías, detectadas {np.mean(av[s]):.1%} (boosting {np.mean(gb[s] > UMBRAL):.1%})")
fm = (F["te"]["mod"] == 5) & np.isin(F["te"]["dep"], [6, 7])
print(f"modelo F en Asturias y León: {int(fm.sum())} fotos, {int(y[fm].sum())} averías, tasa {y[fm].mean():.1%};"
      f" p media red {red[fm].mean():.3f}, boosting {gb[fm].mean():.3f}, logística {lr[fm].mean():.3f}")
fpos = av & ~pos
print(f"falsas alarmas: {int(fpos.sum())}; su fragilidad oculta media {D['frag'][cam[fpos]].mean():+.2f} (sanas sin aviso {D['frag'][cam[~av & ~pos]].mean():+.2f})")

# datos para las páginas
cur = {}
for nm, p in [("red", red), ("boosting", gb), ("logistica", lr)]:
    r, pr, u = curva_pr(y, p)
    k = np.unique(np.r_[np.linspace(0, len(r) - 1, 160).astype(int)])
    cur[nm] = dict(r=np.round(r[k], 4).tolist(), p=np.round(pr[k], 4).tolist(), u=np.round(u[k], 4).tolist())
json.dump(dict(cur=cur, U=U.tolist(), cv=np.round(cv, 2).tolist(), ct=np.round(ct, 2).tolist(),
               bt={k: np.round(v, 4).tolist() for k, v in bt.items()}, fiab=dict(cruda=fiab(cruda), red=fiab(red)), w=w),
          open("evaluar.json", "w", encoding="utf-8"), separators=(",", ":"))
