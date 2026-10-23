import numpy as np
from scipy import stats, optimize
from sim import P_ANUAL, P_INVIERNO, velocidades
from entropia import entropia, entropia_cruzada, kl

print(f"H(inv) = {entropia(P_INVIERNO):.4f}  H(inv, anual) = {entropia_cruzada(P_INVIERNO, P_ANUAL):.4f}"
      f"  KL(inv‖anual) = {kl(P_INVIERNO, P_ANUAL):.4f}  KL(anual‖inv) = {kl(P_ANUAL, P_INVIERNO):.4f}")
q0 = np.array([0.8, 0.12, 0.08, 0.0])                                   # un modelo que da 0 al accidente
print("modelo con q(accidente) = 0:  KL(inv‖q) =", kl(P_INVIERNO, q0), "  KL(q‖inv) =", round(kl(q0, P_INVIERNO), 4))

# ajustar una normal q a la mezcla de velocidades p (ciudad 45 % N(32, 8), autovía 55 % N(84, 6))
x = np.linspace(-40, 160, 20001); dx = x[1] - x[0]
p = 0.45 * stats.norm.pdf(x, 32, 8) + 0.55 * stats.norm.pdf(x, 84, 6)
def KL(a, b):                                                           # ∫ a log(a/b), en nats, por la regla del rectángulo
    m = a > 1e-300
    return float(np.sum(a[m] * (np.log(a[m]) - np.log(np.maximum(b[m], 1e-300)))) * dx)
q = lambda t: stats.norm.pdf(x, t[0], np.exp(t[1]))
fw = min((optimize.minimize(lambda t: KL(p, q(t)), [m0, np.log(20)], method="Nelder-Mead") for m0 in (30, 60, 85)), key=lambda r: r.fun)
rv = min((optimize.minimize(lambda t: KL(q(t), p), [m0, np.log(10)], method="Nelder-Mead") for m0 in (30, 60, 85)), key=lambda r: r.fun)
mu = np.sum(x * p) * dx; sd = np.sqrt(np.sum((x - mu) ** 2 * p) * dx)
print(f"KL(p‖q) mínima: μ = {fw.x[0]:.2f}  σ = {np.exp(fw.x[1]):.2f}  (media y desviación de p: {mu:.2f} {sd:.2f})  KL = {fw.fun:.4f}")
print(f"KL(q‖p) mínima: μ = {rv.x[0]:.2f}  σ = {np.exp(rv.x[1]):.2f}  KL = {rv.fun:.4f}")
r2 = optimize.minimize(lambda t: KL(q(t), p), [30, np.log(10)], method="Nelder-Mead")
print(f"   (desde μ = 30 se queda en el otro modo: μ = {r2.x[0]:.2f}  σ = {np.exp(r2.x[1]):.2f}  KL = {r2.fun:.4f})")
v = velocidades()
print(f"máxima verosimilitud con 5 000 velocidades: μ̂ = {v.mean():.2f}  σ̂ = {v.std():.2f}  ← minimizar KL(p‖q) es ajustar por MV")
