import numpy as np
from scipy import stats
from sim import ALERTAS_A, paradas
from mv import mle_poisson, mle_normal, mle_expon, mle_lognormal

rng = np.random.default_rng(0)
consumo = rng.normal(32, 4.5, 30)                    # L/100 km de 30 trayectos
espera = rng.exponential(9.6, 50)                     # horas entre dos alertas de un camión
par = paradas()                                       # 60 paradas en muelle (min), lognormal

print("Poisson  λ̂ =", mle_poisson(ALERTAS_A),
      "| stats.fit:", round(stats.fit(stats.poisson, ALERTAS_A, bounds={"mu": (0, 20)}).params.mu, 4))
mu, s2 = mle_normal(consumo); loc, esc = stats.norm.fit(consumo)
print(f"normal   μ̂ = {mu:.4f}  σ̂ = {np.sqrt(s2):.4f} | norm.fit: {loc:.4f} {esc:.4f} | std(ddof=1) = {consumo.std(ddof=1):.4f}")
print(f"expon    escala = 1/tasa = {1 / mle_expon(espera):.4f} h | expon.fit(floc=0): {stats.expon.fit(espera, floc=0)[1]:.4f}")
m, s = mle_lognormal(par); sh, _, sc = stats.lognorm.fit(par, floc=0)
print(f"lognormal μ̂ = {m:.4f} (mediana e^μ̂ = {np.exp(m):.2f} min)  σ̂ = {s:.4f} | lognorm.fit: s = {sh:.4f}, scale = {sc:.2f}")

# el σ² de máxima verosimilitud divide entre n: sesgado hacia abajo un factor (n − 1)/n
for n in (2, 5, 10, 30):
    X = rng.normal(32, 4.5, (200_000, n))
    print(f"n = {n:2d}: media de σ̂²_MV = {X.var(axis=1, ddof=0).mean():6.2f}  "
          f"(n−1)/n·σ² = {(n - 1) / n * 4.5**2:6.2f}   media de s² (n − 1) = {X.var(axis=1, ddof=1).mean():6.2f}  σ² = {4.5**2}")
