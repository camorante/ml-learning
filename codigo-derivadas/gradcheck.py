import numpy as np
from datos import CARGA, CONSUMO
from valor import grad_numerico, error_relativo

X = np.c_[np.ones(10), CARGA / 10]; y = CONSUMO
mse = lambda *b: np.mean((X @ np.array(b) - y) ** 2)
bien = lambda b: 2 / len(y) * X.T @ (X @ b - y)
mal1 = lambda b: 1 / len(y) * X.T @ (X @ b - y)             # olvidado el 2 de la derivada de r²
mal2 = lambda b: 2 / len(y) * X.T @ (y - X @ b)             # signo cambiado
mal3 = lambda b: 2 / len(y) * (X @ b - y) @ X[:, ::-1]       # columnas cruzadas
b = np.array([23.0, 6.0]); num = grad_numerico(mse, b)
print("numérico (dif. centrales):", np.round(num, 5))
for nombre, g in (("correcto", bien), ("sin el 2", mal1), ("signo cambiado", mal2),
                  ("columnas cruzadas", mal3)):
    print(f"{nombre:18s} analítico {np.round(g(b), 5)}   error relativo {error_relativo(g(b), num):.1e}")
# precisión simple: el mismo control con float32 da errores mucho mayores aunque el gradiente esté bien
X32, y32 = X.astype(np.float32), y.astype(np.float32)
f32 = lambda b: float(np.mean((X32 @ b.astype(np.float32) - y32) ** 2))
for h in (1e-2, 1e-3, 1e-4):
    n32 = np.array([(f32(b + h * e) - f32(b - h * e)) / (2 * h) for e in np.eye(2)])
    print(f"float32, h = {h:g}: error relativo del gradiente correcto {error_relativo(bien(b), n32):.1e}")
