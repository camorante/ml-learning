import numpy as np
from resumen import varianza, varianza_ingenua, Welford

base = np.array([4.0, 7.0, 13.0, 16.0])           # varianza exacta (n − 1): 30
print("desplazamiento   float64: ingenua   Welford   dos pasadas | float32: ingenua   Welford")
for k in (0, 4, 6, 8, 9):
    x = base + 10.0 ** k
    w = Welford()
    for v in x:
        w.add(v)
    x32 = x.astype(np.float32); w32 = Welford(); w32.m = np.float32(0); w32.M2 = np.float32(0)
    for v in x32:
        w32.add(v)
    print(f"10^{k}  {varianza_ingenua(x):18.6g} {w.varianza():9.6g} {varianza(x):12.6g} |"
          f" {float(varianza_ingenua(x32)):16.6g} {float(w32.varianza()):9.6g}")
