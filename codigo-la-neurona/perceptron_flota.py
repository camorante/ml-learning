import warnings
import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import Perceptron as SkPerceptron
from neurona import Perceptron
from datos import X, y, W_IDEA, B_IDEA

# 1) desde la primera idea del responsable, en el orden de la lista
p = Perceptron(eta=0.1, w0=W_IDEA, b0=B_IDEA).fit(X, y)
print(f"desde (1, 0.2, −6): {p.correcciones} correcciones, fallos por pasada {p.fallos_por_pasada}")
print(f"   w = {np.round(p.w, 4)}  b = {p.b:.4f}   errores {np.sum(p.predict(X) != y)} de {len(y)}")

# 2) desde cero, con las variables tal cual y estandarizadas
def margen(Xa, s):
    """γ = máx_{‖u‖=1} mín_i s_i u·x̃_i: se resuelve mín ‖u‖² con s_i u·x̃_i ≥ 1 y γ = 1/‖u‖."""
    r = minimize(lambda u: u @ u, np.ones(Xa.shape[1]), jac=lambda u: 2 * u, method="SLSQP",
                 constraints=[{"type": "ineq", "fun": lambda u: s * (Xa @ u) - 1, "jac": lambda u: s[:, None] * Xa}],
                 options={"ftol": 1e-14, "maxiter": 500})
    return 1 / np.linalg.norm(r.x)

s = np.where(y == 1, 1.0, -1.0)
Xs = (X - X.mean(0)) / X.std(0)
for nombre, Z in (("tal cual", X), ("estandarizadas", Xs)):
    q = Perceptron(eta=0.1).fit(Z, y)
    Za = np.c_[Z, np.ones(len(Z))]                       # x̃ = (x, 1): el sesgo es un peso más
    R, g = np.max(np.linalg.norm(Za, axis=1)), margen(Za, s)
    print(f"desde cero, {nombre:14}: {q.correcciones:4d} correcciones en {len(q.fallos_por_pasada):2d} pasadas"
          f"   R = {R:.3f}  γ = {g:.4f}  cota de Novikoff (R/γ)² = {(R / g) ** 2:,.0f}")

# 3) el mismo entrenamiento con scikit-learn (y = ±1, fallo si y·z ≤ 0, sin barajar)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    sk = SkPerceptron(eta0=0.1, shuffle=False, tol=None, max_iter=50, penalty=None).fit(X, y)
q = Perceptron(eta=0.1).fit(X, y)
print(f"scikit-learn Perceptron: w = {sk.coef_[0]}  b = {sk.intercept_[0]:.4f}"
      f"   diferencia máxima con el nuestro {max(np.max(np.abs(sk.coef_[0] - q.w)), abs(sk.intercept_[0] - q.b)):.1e}")
