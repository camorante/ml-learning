import numpy as np
from scipy import stats
from sim import poblacion
from muestreo import media_se, n_efectivo_ar1, asignar

d, ruta, camion = poblacion()
N = len(d)
rng = np.random.default_rng(12)
i = rng.choice(N, 16, replace=False)           # las 16 paradas comprobadas
x = d[i]
print("muestra:", x.astype(int).tolist(), " rutas:", np.bincount(ruta[i]).tolist())
n = len(x); m = x.mean(); dev = x - m
print(f"1) suma = {x.sum():.0f}  x̄ = {m:.2f}")
print(f"2) Σ(x − x̄)² = {np.sum(dev**2):.0f}  s² = {np.sum(dev**2)/(n-1):.2f}  s = {x.std(ddof=1):.3f}")
se = x.std(ddof=1) / np.sqrt(n)
print(f"3) error estándar s/√n = {se:.3f}")
f = np.sqrt(1 - n / N)
print(f"4) fpc √(1 − n/N) = {f:.5f}  →  {se*f:.3f};   con n = 4000: √(1 − 1/3) = {np.sqrt(1-4000/N):.3f}")
s = x.std(ddof=1)
print(f"5) para error 2 min: n = (s/2)² = {(s/2)**2:.0f}  coste {4*(s/2)**2:,.0f} €;  para la mitad (4,75): n = {4*n}")
mu, sg = d.mean(), d.std()
z40 = (40 - mu) / (sg / 4); z32 = (32 - mu) / (sg / 4)
sim = d[rng.integers(0, N, (400_000, 16))].mean(1)
print(f"6) μ = {mu:.2f}  σ = {sg:.2f}  σ/4 = {sg/4:.3f}")
print(f"   P(x̄ ≥ 32): z = {z32:.3f}  normal {stats.norm.sf(z32):.3f}  simulación {np.mean(sim >= 32):.3f}")
print(f"   P(x̄ ≥ 40): z = {z40:.3f}  normal {stats.norm.sf(z40):.4f}  simulación {np.mean(sim >= 40):.4f}")
print(f"   P(x̄ ≤ 20): z = {(20-mu)/(sg/4):.3f}  normal {stats.norm.cdf((20-mu)/(sg/4)):.4f}  simulación {np.mean(sim <= 20):.4f}")
ne = n_efectivo_ar1(16, 0.6)
print(f"7) 16 seguidas con φ = 0,6: n_ef = {ne:.2f} (aprox. {16*0.4/1.6:.1f});  error ×{np.sqrt(16/ne):.2f} = {se*np.sqrt(16/ne):.2f}")
Nh = np.array([6000, 4000, 2000]); Sh = np.array([9.41, 19.57, 47.36]); W = Nh / N
print(f"8) ΣW·S² = {np.sum(W*Sh**2):.1f}   ΣW·S = {np.sum(W*Sh):.3f}   σ² = {sg**2:.1f}")
for nn in [120]:
    print(f"   n = {nn}: simple {sg/np.sqrt(nn):.2f}  proporcional {np.sqrt(np.sum(W*Sh**2)/nn):.2f}"
          f"  Neyman {np.sum(W*Sh)/np.sqrt(nn):.2f}  reparto Neyman {asignar(nn, Nh, Sh).tolist()}"
          f"  pesos N_h·S_h = {(Nh*Sh).round(0).tolist()}")
print(f"9) coste anual: 12 000 × 0,70 × {m:.1f} = {N*0.7*m:,.0f} € ± {N*0.7*se:,.0f} €   (verdad: {N*0.7*mu:,.0f} €)")
print(f"   con 361 paradas: ± {N*0.7*s/np.sqrt(361):,.0f} €")
