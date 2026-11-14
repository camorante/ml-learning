"""Pesos de clase en la entropía cruzada binaria: pesar cada avería w veces equivale a bajar el
umbral de aviso a 1/(1 + w). Con w = coste de una avería / coste de una revisión, se minimiza el coste."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from datos import camiones, COSTE_AVERIA, COSTE_REVISION

xe, ye = camiones(2000, seed=100)                  # entrenamiento
xp, yp = camiones(2000, seed=101)                  # prueba: otros 2 000 camiones
U = lambda x: ((x - 30) / 30)[:, None]
w = COSTE_AVERIA / COSTE_REVISION
print(f"averías: {ye.mean():.1%} en entrenamiento, {yp.mean():.1%} en prueba;  w = {COSTE_AVERIA}/{COSTE_REVISION} = {w:.2f}")
m1 = LogisticRegression(C=1e6).fit(U(xe), ye)
mw = LogisticRegression(C=1e6, class_weight={0: 1, 1: w}).fit(U(xe), ye)
print(f"sin pesos: b = {m1.intercept_[0]:.3f}, w_km = {m1.coef_[0, 0]:.3f}   con pesos: b = {mw.intercept_[0]:.3f},"
      f" w_km = {mw.coef_[0, 0]:.3f}   diferencia de b {mw.intercept_[0] - m1.intercept_[0]:.3f} (log w = {np.log(w):.3f})")
eur = lambda c: f"{c:,.0f}".replace(",", " ")      # miles con espacio
def coste(aviso):
    escapan, de_mas = np.sum((~aviso) & (yp == 1)), np.sum(aviso & (yp == 0))
    return escapan, de_mas, (escapan * COSTE_AVERIA + de_mas * COSTE_REVISION)       # € en los 2 000 camiones de prueba
p1, pw = m1.predict_proba(U(xp))[:, 1], mw.predict_proba(U(xp))[:, 1]
for nombre, aviso in [("sin pesos, umbral 0,5", p1 >= 0.5), (f"sin pesos, umbral 1/(1+w) = {1 / (1 + w):.4f}", p1 >= 1 / (1 + w)),
                      ("con pesos, umbral 0,5", pw >= 0.5), ("avisar a todos", np.ones_like(p1, bool)),
                      ("no avisar a nadie", np.zeros_like(p1, bool))]:
    e, d, c = coste(aviso)
    print(f"  {nombre:38s} averías que se escapan {e:4d}   revisiones de más {d:5d}   coste {eur(c):>9s} € en 2 000 camiones")
ws = np.arange(1, 61); cs = [coste(p1 >= 1 / (1 + k))[2] for k in ws]
print(f"barrido de w = 1…60 con umbral 1/(1+w): el coste más bajo es {eur(min(cs))} € con w = {ws[np.argmin(cs)]}")
print(f"km desde la revisión a partir de los que se avisa: umbral 0,5 → {30 + 30 * (-m1.intercept_[0] / m1.coef_[0, 0]):.1f} mil;"
      f"  umbral {1 / (1 + w):.4f} → {30 + 30 * ((np.log(1 / w) - m1.intercept_[0]) / m1.coef_[0, 0]):.1f} mil")
