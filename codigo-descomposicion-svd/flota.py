import numpy as np
from sim import flota
from svd import svd_jacobi

X, verdad, anom, tipo = flota()
U, s, Vt = svd_jacobi(X)
e = s**2 / np.sum(s**2)
print("σ₁…σ₈:", s[:8].round(0))
print("% de ‖X‖²_F por capa:", " ".join(f"{x:.2f}" for x in 100 * e[:6]), "  acumulado con 3:", round(100 * e[:3].sum(), 2))
ok = ~anom
print("RMSE frente a la verdad (casillas normales); datos tal cual:", round(np.sqrt(np.mean((X - verdad)[ok] ** 2)), 1))
for k in range(1, 7):
    Xk = (U[:, :k] * s[:k]) @ Vt[:k]
    print(f"  k={k}: RMSE {np.sqrt(np.mean((Xk - verdad)[ok] ** 2)):5.1f}   ‖X − X_k‖₂ = {np.linalg.norm(X - Xk, 2):7.1f} = σ_{k + 1} {s[k]:7.1f}")

def top(score, n=20):
    o = np.argsort(-score.ravel())[:n]; return anom.ravel()[o].sum()

k = 3; Xk = (U[:, :k] * s[:k]) @ Vt[:k]; R = X - Xk
print("anomalías entre las 20 casillas de mayor |residuo| (k=3):", top(np.abs(R)), "de 20")
# referencia ingenua: distancia a la mediana del mismo camión en el mismo día de la semana
dow = np.arange(28) % 7
med = np.stack([np.median(X[:, dow == w], axis=1) for w in range(7)], 1)[:, dow]
print("referencia: |km − mediana del camión ese día de la semana|:", top(np.abs(X - med)), "de 20")
# ¿qué recogen las capas? intensidad media de cada capa por tipo de camión
for j in range(3):
    sg = np.sign(Vt[j].sum()) or 1.0
    print(f"capa {j + 1}: patrón (lun 1.ª sem … dom 2.ª sem)", (sg * Vt[j, :14] / np.abs(Vt[j]).max()).round(2),
          " intensidad media por tipo:", [round(float(sg * (U[tipo == t, j] * s[j]).mean() * np.abs(Vt[j]).max())) for t in range(3)])
