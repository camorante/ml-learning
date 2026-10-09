import numpy as np
from matrices import matvec, matvec_columnas, matmul, det, inversa, solve
A, p = [[2, -1], [1, 1]], [3, 2]
print("1.", matvec(A, p), matvec_columnas(A, p))
E, R = [[2, 0], [0, 1]], [[0, -1], [1, 0]]
print("2.", matmul(E, R), matmul(R, E), matvec(matmul(E, R), p), matvec(matmul(R, E), p))
print("3.", det(A), det(E), det(R), det(matmul(E, R)))
Ai = inversa(A); print("4.", np.round(Ai, 4).tolist(), matvec(Ai, [4, 5]), np.round(matmul(A, Ai), 12).tolist())
G, b = [[14, 25], [0, 9]], [92, 18]
print("5.", det(G), solve(G, b), np.round(inversa(G), 5).tolist())
G2, b2 = [[14, 25], [37, 66.5]], [92, 244]
print("6.", det(G2), np.round(solve(G2, b2), 6), np.round(solve(G2, [93, 244]), 3), f"cond = {np.linalg.cond(G2):.0f}", f"cond(G) = {np.linalg.cond(G):.2f}")
