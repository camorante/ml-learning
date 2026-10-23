"""Verificación de valor.py: modo inverso, modo directo y fórmulas cerradas contra derivadas
analíticas, diferencias finitas, SciPy y scikit-learn (torch y jax no están instalados)."""
import math, warnings
import numpy as np
from scipy.optimize import approx_fprime, check_grad
from sklearn.linear_model import LinearRegression, LogisticRegression
from valor import (Valor, grad_inverso, grad_directo, grad_numerico, error_relativo,
                   exp, log, sigmoide, tanh, sig, grad_mse, grad_logloss, hess_logloss)

rng = np.random.default_rng(0)
s_ = lambda t: 1 / (1 + math.exp(-t))

# 1) funciones de prueba con su gradiente escrito a mano
def g2(a, b, c):
    e, q = math.exp(a * b), 1 + c * c
    return [b * e / q, a * e / q, -2 * c * e / q ** 2]
def g3(a, b, c):
    L, t = math.log(1 + a * a), math.tanh(b - c)
    return [2 * a / (1 + a * a) * t, L * (1 - t * t), -L * (1 - t * t)]
def g4(a, b, c):
    s = s_(2 * a + b)
    return [2 * c * s * (1 - s) - 1 / (b * b + 1), c * s * (1 - s) + 2 * a * b / (b * b + 1) ** 2, s]
F = [(lambda a, b, c: a * b + c ** 3,                         lambda a, b, c: [b, a, 3 * c ** 2]),
     (lambda a, b, c: exp(a * b) / (1 + c ** 2),               g2),
     (lambda a, b, c: log(1 + a ** 2) * tanh(b - c),           g3),
     (lambda a, b, c: sigmoide(a * 2 + b) * c - a / (b ** 2 + 1), g4),
     (lambda a, b, c: (a - b) ** 2 + (b - c) ** 2 * a,
      lambda a, b, c: [2 * (a - b) + (b - c) ** 2, -2 * (a - b) + 2 * a * (b - c), -2 * a * (b - c)])]
e_inv = e_dir = e_num = 0.0
for f, g in F:
    for _ in range(40):
        x = rng.uniform(-1.5, 1.5, 3); ga = np.array(g(*x))
        e_inv = max(e_inv, error_relativo(grad_inverso(f, x)[1], ga))
        e_dir = max(e_dir, error_relativo(grad_directo(f, x), ga))
        e_num = max(e_num, error_relativo(grad_numerico(f, x), ga))
print(f"5 funciones × 40 puntos: modo inverso vs analítico {e_inv:.1e}   modo directo {e_dir:.1e}"
      f"   dif. centrales {e_num:.1e}")

# 2) red 3-4-1 (tanh) con pérdida logarítmica: Valor frente a retropropagación matricial en NumPy
def red_valor(W1, b1, W2, b2, X, y):
    perdida = 0
    for xi, yi in zip(X, y):
        h = [tanh(sum(W1[j][k] * xi[k] for k in range(3)) + b1[j]) for j in range(4)]
        p = sigmoide(sum(W2[j] * h[j] for j in range(4)) + b2)
        perdida = perdida - (log(p) if yi else log(1 - p))
    return perdida / len(y)

def red_numpy(W1, b1, W2, b2, X, y):
    """Hacia delante y hacia atrás con la regla de la cadena en forma matricial."""
    A = X @ W1.T + b1; H = np.tanh(A); z = H @ W2 + b2; p = sig(z)
    L = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    dz = (p - y) / len(y)                       # ∂L/∂z = (p − y)/n
    dW2 = H.T @ dz; db2 = dz.sum()
    dA = np.outer(dz, W2) * (1 - H ** 2)        # ∂L/∂A = ∂L/∂H ⊙ tanh'
    return L, dA.T @ X, dA.sum(0), dW2, db2

X = rng.normal(size=(8, 3)); y = (rng.random(8) < 0.5).astype(float)
W1, b1, W2, b2 = rng.normal(size=(4, 3)), rng.normal(size=4), rng.normal(size=4), rng.normal()
P = [[[Valor(v) for v in fila] for fila in W1], [Valor(v) for v in b1], [Valor(v) for v in W2], Valor(b2)]
L = red_valor(*P, X, y); L.atras()
gv = np.r_[[v.grad for fila in P[0] for v in fila], [v.grad for v in P[1]],
           [v.grad for v in P[2]], P[3].grad]
desplegar = lambda th: (th[:12].reshape(4, 3), th[12:16], th[16:20], th[20])
plano = np.r_[W1.ravel(), b1, W2, b2]
fn = lambda th: red_numpy(*desplegar(th), X, y)[0]
def gfn(th):
    _, a, b, c, d = red_numpy(*desplegar(th), X, y)
    return np.r_[a.ravel(), b, c, d]
print(f"red 3-4-1, 21 parámetros: pérdida Valor {L.dato:.12f}  NumPy {fn(plano):.12f}")
print(f"   Valor vs retropropagación NumPy {error_relativo(gv, gfn(plano)):.1e}"
      f"   vs scipy approx_fprime {error_relativo(gv, approx_fprime(plano, fn, 1e-7)):.1e}"
      f"   scipy check_grad {check_grad(fn, gfn, plano):.1e}")

# 3) regresión lineal: gradiente cerrado, Valor y diferencias; cero en la solución de scikit-learn
Xr = np.c_[np.ones(50), rng.normal(size=(50, 3))]; yr = Xr @ [1, 2, -1, 0.5] + rng.normal(0, 0.3, 50)
beta = rng.normal(size=4)
fila = lambda b, i: sum(Xr[i, j] * b[j] for j in range(4))          # x_iᵀβ con cualquier tipo
mse = lambda *b: sum((fila(b, i) - yr[i]) ** 2 for i in range(50)) / 50
ga = grad_mse(Xr, yr, beta)
print(f"lineal: (2/n)Xᵀ(Xβ − y) vs Valor {error_relativo(ga, grad_inverso(mse, beta)[1]):.1e}"
      f"   vs dif. centrales {error_relativo(ga, grad_numerico(mse, beta)):.1e}")
lr = LinearRegression(fit_intercept=False).fit(Xr, yr)
print(f"   ‖gradiente‖ en la solución de LinearRegression: {np.linalg.norm(grad_mse(Xr, yr, lr.coef_)):.1e}")

# 4) regresión logística: (1/n)Xᵀ(p − y), hessiana y solución de scikit-learn
yl = (rng.random(50) < sig(Xr @ [-0.5, 1.5, -1, 0.8])).astype(float)
ll = lambda *b: sum(-log(sigmoide(fila(b, i))) if yl[i] else -log(1 - sigmoide(fila(b, i)))
                    for i in range(50)) / 50
ga = grad_logloss(Xr, yl, beta)
print(f"logística: (1/n)Xᵀ(p − y) vs Valor {error_relativo(ga, grad_inverso(ll, beta)[1]):.1e}"
      f"   vs dif. centrales {error_relativo(ga, grad_numerico(ll, beta)):.1e}")
Hn = np.array([grad_numerico(lambda *b: grad_logloss(Xr, yl, np.array(b))[k], beta) for k in range(4)])
print(f"   hessiana (1/n)XᵀWX vs diferencias del gradiente {error_relativo(hess_logloss(Xr, yl, beta), Hn):.1e}")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    lg = LogisticRegression(penalty=None, fit_intercept=False, tol=1e-12, max_iter=10000).fit(Xr, yl)
print(f"   ‖gradiente‖ en la solución de LogisticRegression: {np.linalg.norm(grad_logloss(Xr, yl, lg.coef_[0])):.1e}")
