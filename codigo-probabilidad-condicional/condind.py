import numpy as np
from cond import chi2, condicional

# conductor-días; filas: frenada brusca sí / no; columnas: exceso de velocidad sí / no
nov = np.array([[36, 84], [54, 126]])     # 300 días de novatos: P(F) = 0.4, P(V) = 0.3, independientes
vet = np.array([[7, 63], [63, 567]])      # 700 días de veteranos: P(F) = 0.1, P(V) = 0.1, independientes
tot = nov + vet
for nombre, T in [("novatos", nov), ("veteranos", vet), ("todos", tot)]:
    n = T.sum(); pv = T[:, 0].sum() / n; pvf = condicional(T)[0, 0]
    e, g, p = chi2(T)
    print(f"{nombre:9s} n={n:4d}  P(V)={pv:.3f}  P(V|F)={pvf:.3f}  P(V|no F)={condicional(T)[1, 0]:.3f}  chi2={e:6.2f}  p={p:.2g}")
