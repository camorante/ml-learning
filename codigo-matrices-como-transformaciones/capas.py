import json, numpy as np
rng = np.random.default_rng(5)
# tres capas sin activación = una sola capa afín
W = [rng.normal(size=(8, 4)), rng.normal(size=(6, 8)), rng.normal(size=(2, 6))]
b = [rng.normal(size=8), rng.normal(size=6), rng.normal(size=2)]
X = rng.normal(size=(1000, 4))
H = X
for Wi, bi in zip(W, b):
    H = H @ Wi.T + bi
Wt = W[2] @ W[1] @ W[0]; bt = W[2] @ (W[1] @ b[0] + b[1]) + b[2]
print("sin activación: |red - (W x + b)| máx =", f"{np.abs(H - (X @ Wt.T + bt)).max():.1e}", " W total", Wt.shape)
H = X
for Wi, bi in zip(W, b):
    H = np.maximum(0, H @ Wi.T + bi)
print("con ReLU:       |red - (W x + b)| máx =", f"{np.abs(H - (X @ Wt.T + bt)).max():.2f}")
# la capa de la versión sencilla (temperaturas de motor y aceite)
d = json.load(open("../embed.json")); X = np.array(d["X"]); y = np.array(d["y"])
W1 = np.array([[1.0, -1.0], [-1.0, 1.0]]); b1 = np.array([-0.8, -0.8])
H = np.maximum(0, X @ W1.T + b1)
print("capa ReLU + regla h1 + h2 > 0: errores", int(((H.sum(1) > 0) != y).sum()), "de", len(y))
mejor = len(y)                                   # la mejor recta posible: búsqueda exhaustiva
for th in np.linspace(0, 2 * np.pi, 1441)[:-1]:
    u = np.sort(X @ [np.cos(th), np.sin(th)]); cortes = np.r_[u[0] - 1, (u[1:] + u[:-1]) / 2]
    mejor = min(mejor, min(int(((X @ [np.cos(th), np.sin(th)] > t) != y).sum()) for t in cortes))
print("mejor recta en los datos originales: errores", mejor, "de", len(y))
