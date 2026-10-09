"""Números de los juguetes de la versión sencilla y datos de la idea 5 (embed.json)."""
import json, numpy as np

np.set_printoptions(suppress=True)
# Juguete: ejemplos
pre = {"identidad": [[1, 0], [0, 1]], "estirar": [[2, 0], [0, 1]], "girar": [[0.8, -0.6], [0.6, 0.8]],
       "reflejar": [[-1, 0], [0, 1]], "inclinar": [[1, 1], [0, 1]], "aplastar": [[1, 2], [0.5, 1]]}
for k, M in pre.items():
    M = np.array(M, float); print(f"{k:10s} det={np.linalg.det(M):+.3f}  ángulo giro={np.degrees(np.arctan2(M[1,0], M[0,0])):.2f}")

# Idea 1: A = [[2,-1],[1,1]] y p = (3, 2)
A = np.array([[2, -1], [1, 1]]); p = np.array([3, 2])
print("idea1:", A @ p, "=", 3 * A[:, 0], "+", 2 * A[:, 1], " det", np.linalg.det(A).round(6))
G = np.array([[14, 25], [0, 9]]); h = np.array([3, 2])
print("litros/peaje:", G @ h, " filas:", 14 * 3 + 25 * 2, 0 * 3 + 9 * 2)

# Idea 2: triángulo depósito A=(0,0), cliente B=(4,1), gasolinera C=(1,3) (km)
P = np.array([[0, 0], [4, 1], [1, 3]], float)
c, s = 4 / np.sqrt(17), 1 / np.sqrt(17)
T2 = {"original": np.eye(2), "girar a la carretera": np.array([[c, s], [-s, c]]), "a millas": 0.621371 * np.eye(2),
      "estirar el este x2": np.array([[2, 0], [0, 1]]), "inclinar": np.array([[1, 1], [0, 1]]), "cambiar este y norte": np.array([[0, 1], [1, 0]])}
def medir(Q):
    ab, ac = Q[1] - Q[0], Q[2] - Q[0]
    ang = np.degrees(np.arccos(ab @ ac / np.linalg.norm(ab) / np.linalg.norm(ac)))
    area = (ab[0] * ac[1] - ab[1] * ac[0]) / 2
    return np.linalg.norm(ab), ang, area
for k, M in T2.items():
    Q = P @ M.T; d, a, ar = medir(Q)
    print(f"idea2 {k:22s} B={Q[1].round(3)} C={Q[2].round(3)} AB={d:.3f} ángulo={a:.1f} área={ar:+.3f}")

# Idea 3: girar 90° y estirar x2 en horizontal
R = np.array([[0, -1], [1, 0]]); E = np.array([[2, 0], [0, 1]])
print("idea3: primero girar, luego estirar = E R =", (E @ R).tolist(), "  primero estirar, luego girar = R E =", (R @ E).tolist())

# Idea 4: columnas que se van pareciendo
c1 = np.array([1.5, 0.5]); a2, z2 = np.array([-0.5, 1.2]), np.array([0.75, 0.25])
for t in [0, 0.5, 0.9, 0.98, 1]:
    M = np.c_[c1, (1 - t) * a2 + t * z2]; dt = np.linalg.det(M)
    print(f"idea4 t={t:.2f} det={dt:.4f} cond={np.linalg.cond(M):.1f}")
M = np.c_[c1, z2]; print("idea4 t=1:", M @ [1, 1], M @ [2, -1])

# Idea 5: temperatura del motor y del aceite (desviaciones estandarizadas) en 160 camiones
rng = np.random.default_rng(11)
n_ok, n_bad = 110, 50
s_ok = rng.uniform(-2.2, 2.2, n_ok); d_ok = rng.uniform(-0.55, 0.55, n_ok)
s_bad = rng.uniform(-2.0, 2.0, n_bad); d_bad = rng.choice([-1, 1], n_bad) * rng.uniform(1.1, 2.2, n_bad)
s_all = np.r_[s_ok, s_bad]; d_all = np.r_[d_ok, d_bad]
X = np.c_[s_all + d_all / 2, s_all - d_all / 2]           # x1 - x2 = d (diferencia), (x1 + x2)/2 = s
y = np.r_[np.zeros(n_ok), np.ones(n_bad)].astype(int)
X = X.round(2)
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
best = (1.0,)
for th in np.linspace(0, 2 * np.pi, 1441)[:-1]:             # mejor recta posible, búsqueda exhaustiva
    u = X @ [np.cos(th), np.sin(th)]
    us = np.unique(u); mids = np.r_[us[0] - 0.01, (us[1:] + us[:-1]) / 2, us[-1] + 0.01]
    for t in mids:
        e = np.mean((u > t) != y)
        if e < best[0]: best = (e, th, t)
print("idea5: mejor recta en los datos originales, error", round(best[0], 4), "=", round(best[0] * len(y)), "de", len(y), "th", round(best[1], 4), "t", round(best[2], 4))
c = 0.8
H = np.maximum(0, X @ np.array([[1, -1], [-1, 1]]).T - c)
pred = (H.sum(1) > 0).astype(int)
print("idea5: tras matriz + desplazamiento (c=0.8) + ReLU, regla h1+h2>0: errores", int((pred != y).sum()),
      " normales en el origen:", int(((H == 0).all(1) & (y == 0)).sum()))
for cc in [0.3, 0.55, 0.8, 1.1, 1.4]:
    H = np.maximum(0, X @ np.array([[1, -1], [-1, 1]]).T - cc); print("   c=", cc, "errores", int(((H.sum(1) > 0).astype(int) != y).sum()))
json.dump(dict(X=X.tolist(), y=y.tolist(), line=[round(float(best[1]), 5), round(float(best[2]), 5)]), open("../embed.json", "w"))
