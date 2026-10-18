import numpy as np
r = np.random.default_rng(4)
for n in (1_000, 10_000, 100_000, 1_000_000):
    averia = r.random(n) < 0.02
    alerta = r.random(n) < np.where(averia, 0.90, 0.05)
    print(f"{n:>9d} semanas  alertas {alerta.sum():>6d}  P(alerta|avería) {alerta[averia].mean():.3f}  P(avería|alerta) {averia[alerta].mean():.3f}")
