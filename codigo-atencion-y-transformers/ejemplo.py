"""El ejemplo resuelto: la atención de «regreso» sobre un viaje de 3 fichas con d = 2, a mano, y comprobado con
F.scaled_dot_product_attention."""
import numpy as np
import torch
import torch.nn.functional as F

fichas = ["salida", "Madrid", "regreso"]
X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])            # sus embeddings (una fila por ficha)
Wq = np.array([[1.0, 1.0], [-2.0, 1.0]]); Wk = np.eye(2); Wv = np.eye(2)
Q, K, V = X @ Wq, X @ Wk, X @ Wv
print("q de regreso:", Q[2], "· claves k:", K.tolist())
s = K @ Q[2]                                                   # q·k con cada ficha
print("puntuaciones q·k:", s, "· ÷√2:", np.round(s / np.sqrt(2), 5))
e = np.exp(s / np.sqrt(2))
w = e / e.sum()
print("e^(q·k/√2):", np.round(e, 5), "· suma", round(e.sum(), 5))
print("pesos:", np.round(w, 5), "· suman", w.sum())
o = w @ V
print("salida = suma de pesos × v:", np.round(o, 5))
e1 = np.exp(s); print("sin dividir entre √2, pesos:", np.round(e1 / e1.sum(), 5))
m = np.exp(K[:2] @ Q[1] / np.sqrt(2)); print("fila de Madrid con máscara causal (solo salida y Madrid):",
                                            "q =", Q[1], "pesos", np.round(m / m.sum(), 5))
t = torch.tensor
ref = F.scaled_dot_product_attention(t(Q)[None], t(K)[None], t(V)[None], is_causal=True)[0]
print("PyTorch, fila de regreso:", np.round(ref[2].numpy(), 5), "· error", float(np.abs(ref[2].numpy() - o).max()))
