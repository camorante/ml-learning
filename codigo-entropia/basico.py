import numpy as np
from scipy import stats
from sim import TIPOS, P_ANUAL, P_INVIERNO, partes
from entropia import sorpresa, entropia

for t, p, s in zip(TIPOS, P_ANUAL, sorpresa(P_ANUAL)):
    print(f"{t:16s} p = {p:5.3f}  sorpresa = {s:.3f} bits = {s * np.log(2):.3f} nats")
print("H(anual)   =", round(entropia(P_ANUAL), 4), "bits  =", round(entropia(P_ANUAL, np.e), 4), "nats",
      " scipy:", round(stats.entropy(P_ANUAL, base=2), 4))
print("H(invierno)=", round(entropia(P_INVIERNO), 4), " máximo con 4 tipos: log2 4 =", np.log2(4))

# estimar H a partir de partes observados: el estimador «de frecuencias» se queda corto con pocos datos
x = partes(10_000_000, P_INVIERNO, seed=7)
for n in (20, 100, 1000, 10_000):
    cuentas = [np.bincount(m, minlength=4) for m in x[: 1000 * n].reshape(1000, n)]     # 1000 muestras de n partes
    est = [entropia(c) for c in cuentas]
    mm = [h + (np.count_nonzero(c) - 1) / (2 * n * np.log(2)) for h, c in zip(est, cuentas)]   # Miller-Madow
    print(f"n = {n:6d}: media del estimador {np.mean(est):.4f}  con Miller-Madow {np.mean(mm):.4f}  (verdad {entropia(P_INVIERNO):.4f})")
