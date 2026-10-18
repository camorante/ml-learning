import numpy as np
from sim import flota, confusion, atipico, NOMBRES
from cor import matriz_corr, matriz_cov, pearson, spearman, correlacion_parcial, vif

X = flota(); R = matriz_corr(X)
iu = np.triu_indices(9, 1); o = np.argsort(-np.abs(R[iu]))[:4]
print("pares más correlacionados:", [(NOMBRES[iu[0][k]], NOMBRES[iu[1][k]], round(float(R[iu][k]), 3)) for k in o])
print("VIF km, litros, €, carga, ralentí, edad:", vif(X[:, :6]).round(1))
print("VIF sin los €:                       ", vif(X[:, [0, 1, 3, 4, 5]]).round(1))
w, V = np.linalg.eigh(R[:7, :7]); print("autovalores de R (7 columnas reales):", w[::-1].round(2), " varianza explicada por el 1.º:", f"{w[-1] / 7:.0%}")
S = matriz_cov(X[:, :3]); print("covarianzas km/litros/€ (escalas distintas): diagonal", np.diag(S).round(0))

edad, ciudad, cons = confusion()
print(f"edad y consumo: r {pearson(edad, cons):+.3f};  parcial dado ruta {correlacion_parcial(edad, cons, ciudad):+.3f};"
      f"  r(edad, ciudad) {pearson(edad, ciudad):+.3f}  r(ciudad, consumo) {pearson(ciudad, cons):+.3f}")
(a, b), (x, y) = atipico()
for nombre, p, q, (ox, oy) in [("sin relación", a, b, (72, 52)), ("con relación", x, y, (27, 21))]:
    P, Q = np.r_[p, ox], np.r_[q, oy]
    print(f"atípico, {nombre}: r {pearson(p, q):+.3f} → {pearson(P, Q):+.3f};  Spearman {spearman(p, q):+.3f} → {spearman(P, Q):+.3f}")
R15 = matriz_corr(X[:15]); ruido = [(i, j) for i, j in zip(*iu) if j >= 7]
k = max(ruido, key=lambda t: abs(R15[t])); print(f"15 camiones: {NOMBRES[k[0]]} / {NOMBRES[k[1]]} r = {R15[k]:+.2f} (500 camiones: {R[k]:+.2f})")
