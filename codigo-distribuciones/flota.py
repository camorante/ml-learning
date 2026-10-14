import numpy as np
from scipy import stats
from sim import flota

consumo, tipo, alertas, tasa, paradas, tarde = flota()
pc = lambda v: f"{100 * v:.2f} %"

# 1. consumo: una campana por tipo, dos campanas juntas
for nombre, x in [("larga distancia", consumo[tipo == 0]), ("toda la flota", consumo)]:
    m, s = x.mean(), x.std(ddof=1)
    r = stats.probplot(x, dist="norm")[1][2]
    print(f"consumo {nombre:16s} media {m:.2f}  σ {s:.2f}  dentro de ±2σ: {pc(np.mean(abs(x - m) < 2 * s))}"
          f"  (normal: 95.45 %)  r del QQ {r:.4f}")
x = consumo; m, s = x.mean(), x.std(ddof=1)
print(f"  flota: trayectos por debajo de 26 L/100 km  reales {pc(np.mean(x < 26))}  normal ajustada {pc(stats.norm.cdf(26, m, s))}")

# 2. alertas: Poisson por camión, sobredispersión en la flota
a = alertas.ravel()
print(f"alertas: media {a.mean():.3f}  varianza {a.var(ddof=1):.3f}  índice var/media {a.var(ddof=1) / a.mean():.2f}")
idx = alertas.var(axis=1, ddof=1) / alertas.mean(axis=1)
print(f"  índice var/media de cada camión (60 días): mediana {np.median(idx):.2f}, percentiles 5–95 {np.percentile(idx, 5):.2f}–{np.percentile(idx, 95):.2f}")
lam = a.mean()
k = a.mean() ** 2 / (a.var(ddof=1) - a.mean())               # binomial negativa por momentos: Var = μ + μ²/k
nb = stats.nbinom(k, k / (k + lam))
for umbral in (7, 10):
    print(f"  días con ≥ {umbral:2d} alertas  reales {pc(np.mean(a >= umbral))}  Poisson({lam:.2f}) {pc(stats.poisson.sf(umbral - 1, lam))}"
          f"  binomial negativa (k = {k:.2f}) {pc(nb.sf(umbral - 1))}")
tot = alertas.sum(axis=1)
p_cam = stats.poisson.sf(tot - 1, 60 * lam)                  # P(≥ total del camión) si todos tuvieran la tasa media
print(f"  camiones con P(≥ su total | tasa media) < 0.001: {np.sum(p_cam < 0.001)} de 300; su tasa real media {tasa[p_cam < 0.001].mean():.2f}")

# 3. paradas: normal frente a lognormal
m, s = paradas.mean(), paradas.std(ddof=1)
sl, _, esc = stats.lognorm.fit(paradas, floc=0)
print(f"paradas: media {m:.1f} min  mediana {np.median(paradas):.1f}  σ {s:.1f}  asimetría {stats.skew(paradas):.2f}")
print(f"  lognormal MV: mediana e^μ = {esc:.2f} min, s = {sl:.3f}")
for nombre, F in [("normal (momentos)", stats.norm(m, s)), ("lognormal (MV)", stats.lognorm(sl, scale=esc))]:
    print(f"  {nombre:18s} P(< 0) {pc(F.cdf(0)):>8s}  P(> 60 min) {pc(F.sf(60)):>7s}  cuantil 95 {F.ppf(0.95):5.1f}  KS {stats.kstest(paradas, F.cdf).statistic:.3f}")
print(f"  datos              P(< 0) {pc(0.0):>8s}  P(> 60 min) {pc(np.mean(paradas > 60)):>7s}  cuantil 95 {np.percentile(paradas, 95):5.1f}")

# 4. entregas tarde: frecuencias observadas frente a binomial(20, p̂)
p_hat = tarde.mean() / 20
obs = np.append(np.bincount(tarde, minlength=6)[:6], np.sum(tarde >= 6))
esp = 250 * np.append(stats.binom.pmf(np.arange(6), 20, p_hat), stats.binom.sf(5, 20, p_hat))
chi = stats.chisquare(obs, esp, ddof=1)
print(f"entregas tarde: p̂ = {p_hat:.4f}   observado {obs.tolist()}   esperado {esp.round(1).tolist()}")
print(f"  χ² = {chi.statistic:.2f}, p = {chi.pvalue:.2f} (5 grados de libertad: 7 casillas − 1 − p estimado)")
