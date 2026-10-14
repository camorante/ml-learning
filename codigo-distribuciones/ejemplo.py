import math
import numpy as np
from scipy import stats

# 1–2. normal: consumo μ = 32, σ = 2.5 L/100 km
z = (37 - 32) / 2.5; print("z =", z, " P(X > 37) =", round(stats.norm.sf(z), 4), " 30 L × 1,65 € =", 30 * 1.65)
za, zb = (30 - 32) / 2.5, (35 - 32) / 2.5
print("z:", za, zb, " Φ:", round(stats.norm.cdf(zb), 4), round(stats.norm.cdf(za), 4), " P(30 < X < 35) =", round(stats.norm.cdf(zb) - stats.norm.cdf(za), 4))
# 3–5. binomial: 20 entregas, p = 0.1
p = [math.comb(20, k) * 0.1**k * 0.9**(20 - k) for k in range(8)]
print("C(20,k):", [math.comb(20, k) for k in range(6)], " 0.9^20, 0.9^19, 0.9^18:", round(0.9**20, 4), round(0.9**19, 4), round(0.9**18, 4))
print("P(k), k = 0..5:", [round(v, 4) for v in p[:6]], " P(≤ 2) =", round(sum(p[:3]), 4), " P(≤ 5) =", round(sum(p[:6]), 4), " P(≥ 6) =", round(1 - sum(p[:6]), 4))
print("media", 20 * 0.1, " varianza", round(20 * 0.1 * 0.9, 2), " σ", round(math.sqrt(1.8), 3), " coste/día", 2 * 15, " 250 días", 250 * 2 * 15, " días con ≥ 6:", round(250 * (1 - sum(p[:6])), 1))
# 6–7. Poisson λ = 2 y la binomial que se le parece
q = [math.exp(-2) * 2**k / math.factorial(k) for k in range(8)]
print("Poisson P(k), k = 0..6:", [round(v, 4) for v in q[:7]], " P(≥ 7) =", round(1 - sum(q[:7]), 4), " e^-2 =", round(math.exp(-2), 4))
print("binomial(20, 0.1) frente a Poisson(2), k = 0..4:", [(round(a, 3), round(b, 3)) for a, b in zip(p[:5], q[:5])])
print("binomial(200, 0.01): P(0) =", round(0.99**200, 4))
# 8. estimar λ con 10 días de un camión
x = np.array([2, 1, 0, 1, 2, 2, 0, 3, 3, 3])
print("suma", x.sum(), " λ̂ =", x.mean(), " cuadrados de las desviaciones", ((x - x.mean())**2).round(2).tolist(), " suma", round(((x - x.mean())**2).sum(), 2),
      " s² =", round(x.var(ddof=1), 3), " s²/λ̂ =", round(x.var(ddof=1) / x.mean(), 2), " ±", round(math.sqrt(2 / 9), 2))
print("P(≥ 7 | λ̂ = 1.7) =", round(stats.poisson.sf(6, 1.7), 4))
