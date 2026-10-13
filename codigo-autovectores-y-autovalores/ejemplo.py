import numpy as np
from autovectores import eig2, potencia, estacionaria

A = np.array([[2.0, 1.0], [1.0, 2.0]])
t, d = np.trace(A), np.linalg.det(A)
print("traza", t, " det", round(d, 10), " λ =", eig2(A))                         # λ² − 4λ + 3 = 0
v1, v2 = np.array([1, 1]) / np.sqrt(2), np.array([1, -1]) / np.sqrt(2)
print("A v1 =", (A @ v1 * np.sqrt(2)).round(12), "= 3·(1, 1)   A v2 =", (A @ v2 * np.sqrt(2)).round(12), "= 1·(1, −1)   v1·v2 =", round(v1 @ v2, 12) + 0.0)
Q = np.c_[v1, v2]; L = np.diag([3.0, 1.0])
print("Q Λ Qᵀ =", (Q @ L @ Q.T).round(12).tolist(), "  A⁵ =", np.linalg.matrix_power(A, 5).tolist(), "  (3⁵ ± 1)/2 =", (3**5 + 1) / 2, (3**5 - 1) / 2)
x = np.array([1.0, 0.0])
for k in range(5):
    print(" paso", k, x, " ángulo", round(np.degrees(np.arctan2(x[1], x[0])), 2), "°  ‖Ax‖/‖x‖ =", round(np.linalg.norm(A @ x) / np.linalg.norm(x), 4),
          " Rayleigh xᵀAx/xᵀx =", round(x @ A @ x / (x @ x), 6))
    x = A @ x
lam, v, k = potencia(A, [1, 0]); print(f"potencia(): λ = {lam:.12f}, v = {v.round(6)}, {k} iteraciones")
P = np.array([[0.9, 0.1], [0.6, 0.4]])
print("estacionaria", estacionaria(P, "lineal"), " autovalores de Pᵀ", np.linalg.eigvals(P.T).real, " traza − 1 =", round(np.trace(P) - 1, 10))
p = np.array([0.0, 1.0])
for k in range(5):
    print(" día", k, p.round(4), " fórmula 6/7 − (6/7)·0.3^k =", round(6 / 7 - 6 / 7 * 0.3**k, 4)); p = p @ P
