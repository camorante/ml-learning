import numpy as np
from scipy import stats
from sim import velocidades
from entropia import entropia, entropia_normal

v = velocidades()                                                    # km/h, mezcla ciudad/autovía
s = v.std()
print(f"normal con σ = {s:.2f} km/h: h = {entropia_normal(s, 2):.4f} bits  (scipy: {stats.norm(0, s).entropy() / np.log(2):.4f})")
print(f"   la misma en m/s (σ = {s / 3.6:.2f}): h = {entropia_normal(s / 3.6, 2):.4f} bits = antes − log2 3,6 = {entropia_normal(s, 2) - np.log2(3.6):.4f}")
print(f"   σ = 0,1 km/h: h = {entropia_normal(0.1, 2):.4f} bits  ← negativa")
for delta in (8, 2, 0.5):
    c = np.bincount(np.floor((v - v.min()) / delta).astype(int))
    print(f"tramos de {delta:>3} km/h: H discreta = {entropia(c):.4f} bits   H + log2 Δ = {entropia(c) + np.log2(delta):.4f}")
# h de la mezcla (por integración) frente a la normal con la misma varianza: la normal tiene más
x = np.linspace(-40, 160, 40001); dx = x[1] - x[0]
p = 0.45 * stats.norm.pdf(x, 32, 8) + 0.55 * stats.norm.pdf(x, 84, 6)
hm = -np.sum(p * np.log2(p + 1e-300)) * dx
sd = np.sqrt(np.sum(x**2 * p) * dx - (np.sum(x * p) * dx) ** 2)
print(f"mezcla: h = {hm:.4f} bits  normal con su misma σ = {sd:.2f}: h = {entropia_normal(sd, 2):.4f} bits (máximo para esa varianza)")
