import numpy as np
from svd import svd_jacobi, rango_k

A = np.array([[2.0, 2.0], [-1.0, 2.0]])
G = A.T @ A; print(G, np.trace(G), round(np.linalg.det(G), 9))           # AᵀA, traza y determinante
lam = np.roots([1, -np.trace(G), np.linalg.det(G)]); print(lam, np.sqrt(lam))
v1 = np.array([1, 2]) / np.sqrt(5); v2 = np.array([-2, 1]) / np.sqrt(5)
print(G @ v1 / v1, G @ v2 / v2)                                          # autovectores: G v = λ v
u1 = A @ v1 / 3; u2 = A @ v2 / 2; print(u1 * np.sqrt(5), u2 * np.sqrt(5), round(u1 @ u2, 12))
U = np.c_[u1, u2]; V = np.c_[v1, v2]; print((U * [3, 2]) @ V.T)          # U Σ Vᵀ = A
print(np.degrees(np.arctan2(v1[1], v1[0])).round(2), np.degrees(np.arctan2(u1[1], u1[0])).round(2), np.linalg.det(A).round(9))
print(np.linalg.svd(A)[1], np.abs(svd_jacobi(A)[0].T @ U).round(12))     # mismos σ; mismas columnas salvo signo

B = np.array([[8.0, 6.0], [6.0, 3.0], [6.0, 4.0]])
H = B.T @ B; print(H, np.trace(H), round(np.linalg.det(H), 9), np.roots([1, -np.trace(H), np.linalg.det(H)]))
w1 = np.array([3, 2]) / np.sqrt(13); print(B @ w1 * np.sqrt(13), np.linalg.norm(B @ w1))
B1 = np.outer(B @ w1, w1); print(B1.round(4), (B1 * 13).round(9))
E = B - B1; print((E * 13).round(9) + 0.0, np.linalg.norm(E, "fro").round(12), np.linalg.norm(E, 2).round(12))
print(np.abs(rango_k(B, 1) - B1).max() < 1e-12, round(196 / 197 * 100, 2))
