"""Widget de la §3: capa oculta fija de k neuronas (centros repartidos por el rango de velocidades)
y solo la capa de salida ajustada por mínimos cuadrados. Mismo cálculo que el JS del widget."""
import numpy as np
from datos import tramos

v, c = tramos(); vn, cn = tramos(200, seed=7)
PHI = {"relu": lambda z: np.maximum(z, 0), "tanh": np.tanh, "sigmoide": lambda z: 1 / (1 + np.exp(-z)),
       "ninguna": lambda z: z}

def capa_oculta(v, k, act):
    u = np.asarray(v, float) / 100
    cen = 0.12 + 0.98 * np.arange(k) / k                    # el primer centro, al principio de los datos
    s = 0.98 / k                                            # anchura: la separación entre centros
    return np.column_stack([np.ones_like(u)] + [PHI[act]((u - ck) / s) for ck in cen])

def ajuste(k, act):
    H = capa_oculta(v, k, act)
    a = np.linalg.solve(H.T @ H + 1e-9 * np.eye(H.shape[1]), H.T @ c)   # ecuaciones normales
    e = np.sqrt(np.mean((H @ a - c) ** 2)); en = np.sqrt(np.mean((capa_oculta(vn, k, act) @ a - cn) ** 2))
    return e, en

for act in PHI:
    print(f"{act:8s}", "  ".join(f"k={k}: {ajuste(k, act)[0]:.2f}/{ajuste(k, act)[1]:.2f}" for k in (1, 2, 3, 4, 6, 8, 12)))
