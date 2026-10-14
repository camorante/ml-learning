import numpy as np
from scipy import stats

consumo = stats.norm(loc=32, scale=2.5)          # L/100 km por trayecto: μ = 32, σ = 2.5
tarde = stats.binom(n=20, p=0.10)                # entregas tarde de 20
alertas = stats.poisson(mu=2)                    # alertas de frenada por camión y día

print("consumo  P(X > 37)", round(consumo.sf(37), 4), " P(30 < X < 35)", round(consumo.cdf(35) - consumo.cdf(30), 4),
      " 95 % central", consumo.ppf([0.025, 0.975]).round(2))
print("tarde    P(X = 2)", round(tarde.pmf(2), 4), " P(X ≤ 2)", round(tarde.cdf(2), 4), " P(X ≥ 6)", round(tarde.sf(5), 4),
      " media y varianza", [float(v) for v in tarde.stats(moments="mv")])
print("alertas  P(X = 0)", round(alertas.pmf(0), 4), " P(X ≥ 7)", round(alertas.sf(6), 4),
      " intervalo 95 %", [float(v) for v in alertas.interval(0.95)], " cuantil 99,9 %", float(alertas.ppf(0.999)))
print("muestras", alertas.rvs(size=10, random_state=1), consumo.rvs(size=3, random_state=1).round(1))
