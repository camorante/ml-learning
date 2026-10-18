import numpy as np
from scipy import stats

rng = np.random.default_rng(17); R, K = 50_000, 30
# prueba A/A: cada día 20 camiones por grupo; se mira el test z acumulado al final de cada día
dia = rng.normal(0, 1, (R, K)) * np.sqrt(2 / 20)          # diferencia de medias del día (σ = 1)
acum = np.cumsum(dia, 1) / np.arange(1, K + 1)              # diferencia acumulada
z = acum / np.sqrt(2 / (20 * np.arange(1, K + 1)))
p = 2 * stats.norm.sf(np.abs(z))
print(f"mirando solo el día 30: falsos positivos = {np.mean(p[:, -1] < 0.05):.4f}")
for k in (2, 5, 10, 30):
    idx = np.unique(np.round(np.linspace(K / k, K, k)).astype(int) - 1)
    print(f"mirando {k:2d} veces y parando al primer p < 0,05: {np.mean((p[:, idx] < 0.05).any(1)):.4f}")
dias = np.argmax(p < 0.05, 1)[(p < 0.05).any(1)] + 1
print("día de la primera «significación» (cuartiles):", np.quantile(dias, [0.25, 0.5, 0.75]).tolist())
mx = np.abs(z).max(1); c = np.quantile(mx, 0.95)
print(f"umbral constante (Pocock) para 30 miradas: |z| > {c:.3f}  ↔  p < {2 * stats.norm.sf(c):.4f} cada día")
# O'Brien-Fleming: |z_k| > c·√(K/k), muy exigente al principio
cs = np.linspace(1.8, 2.4, 601); tasa = [np.mean((np.abs(z) > cc * np.sqrt(K / np.arange(1, K + 1))).any(1)) for cc in cs]
cof = cs[np.argmin(np.abs(np.array(tasa) - 0.05))]
print(f"O'Brien-Fleming: c = {cof:.3f} → día 1 exige |z| > {cof * np.sqrt(K):.2f}, el día 30 |z| > {cof:.2f} (p < {2 * stats.norm.sf(cof):.4f})")
