"""Datos de los juguetes de la versión sencilla (embed.json) y sus números clave."""
import json, numpy as np
from producto import *

# Juguete: coste del viaje
km = np.array([40, 120, 200]); precio = np.array([0.60, 0.40, 0.50])
print("viaje:", km * precio, punto(km, precio))

# Idea 3: perfiles (horas/semana en ciudad, en autopista)
T = {"A": (12, 28), "B": (24, 56), "C": (22, 18), "D": (40, 10), "E": (6, 44), "F": (30, 34)}
names = list(T)
for r in names:
    d = {o: np.hypot(*(np.subtract(T[r], T[o]))) for o in names if o != r}
    c = {o: coseno(T[r], T[o]) for o in names if o != r}
    print(r, "dist->", min(d, key=d.get), round(min(d.values()), 1), "cos->", max(c, key=c.get), round(max(c.values()), 3),
          {o: (round(d[o], 1), round(c[o], 3), round(angulo_grados(T[r], T[o]), 1)) for o in d})

# Idea 4: camiones con/sin avería (vibración 0-10, horas desde revisión en cientos 0-10)
rng = np.random.default_rng(4)
n = 22
av = np.c_[rng.normal(6.6, 1.3, n), rng.normal(6.4, 1.5, n)]
sa = np.c_[rng.normal(3.6, 1.3, n), rng.normal(3.6, 1.5, n)]
X = np.clip(np.r_[av, sa], 0.3, 9.7).round(1); y = np.r_[np.ones(n), np.zeros(n)].astype(int)
def errs(w, b):
    s = X @ np.array(w) + b; return int(((s > 0) != y).sum())
print("errores w=(1,0) b=-5:", errs((1, 0), -5))
best = min(((errs((w1, w2), b), w1, w2, b) for w1 in np.arange(0, 2.01, .1) for w2 in np.arange(0, 2.01, .1) for b in np.arange(-15, 0.01, .1)), key=lambda t: t[0])
print("mejor", best)
print("errores w=(1,0.8) b=-9:", errs((1, .8), -9), " w=(1,1) b=-10:", errs((1, 1), -10))
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=100).fit(X, y); print("logística", lr.coef_, lr.intercept_, errs(lr.coef_[0], lr.intercept_[0]))
# camión del ejemplo
x0 = np.array([6.2, 5.5]); w = np.array([1, .8]); b = -9
print("ejemplo:", puntuacion(w, x0, b), norma(w), distancia_frontera(w, x0, b))

# Idea 5: incidencias con 3 números (frenos, electricidad, motor) puestos a mano
past = [("Chirrido al frenar en bajada", [0.9, 0.0, 0.1]),
        ("Pastillas de freno gastadas", [1.0, 0.0, 0.0]),
        ("No arranca: batería descargada", [0.0, 0.9, 0.4]),
        ("Testigo de batería encendido", [0.1, 1.0, 0.0]),
        ("Humo negro al acelerar", [0.0, 0.1, 1.0]),
        ("Ruido metálico en el motor", [0.1, 0.0, 0.9])]
new = [("Ruido al pisar el freno", [0.9, 0.0, 0.3]),
       ("Fallo del alternador", [0.0, 0.9, 0.3]),
       ("El motor pierde potencia", [0.0, 0.15, 1.0])]
for t, v in new:
    print(t, [round(coseno(v, p), 3) for _, p in past])
json.dump(dict(T=T, X=X.tolist(), y=y.tolist(), past=past, new=new), open("../embed.json", "w"), ensure_ascii=False)
