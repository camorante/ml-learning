import math
import numpy as np
from scipy import stats

x = np.array([30.9, 31.1, 37.0, 40.7, 30.5, 30.6, 25.2, 37.2, 30.4, 30.4]); n = 10
m = x.sum() / n; d = x - m
print("1. suma", round(x.sum(), 1), " media", round(m, 2))
print("2. d", np.round(d, 1).tolist(), "\n   d²", np.round(d**2, 2).tolist(), " Σd²", round(np.sum(d**2), 2))
s2 = np.sum(d**2) / (n - 1); s = math.sqrt(s2); print("   s² = Σd²/9 =", round(s2, 4), " s =", round(s, 4))
se = s / math.sqrt(n); print("3. √10 =", round(math.sqrt(10), 4), " error estándar", round(se, 4))
print("4. 1,96·se =", round(1.96 * se, 3), " [", round(m - 1.96 * se, 2), ",", round(m + 1.96 * se, 2), "]")
t = stats.t.ppf(0.975, 9); print("5. t(0,975; 9) =", round(t, 4), " t·se =", round(t * se, 3), " [", round(m - t * se, 2), ",", round(m + t * se, 2), "]")
print("6. n = (1,96·4,5/0,5)² =", round((1.96 * 4.5 / 0.5)**2, 2), " (1,96·4,5/0,25)² =", round((1.96 * 4.5 / 0.25)**2, 1))
hp = t * s * math.sqrt(1 + 1 / n); print("7. √(1 + 1/10) =", round(math.sqrt(1.1), 4), " margen", round(hp, 3), " [", round(m - hp, 2), ",", round(m + hp, 2), "]")
k, N, z = 1, 40, 1.96; p = k / N
hw = z * math.sqrt(p * (1 - p) / N); print("8. Wald: p", p, " √(p(1−p)/n) =", round(math.sqrt(p * (1 - p) / N), 5), " ±", round(hw, 4), " [", round(p - hw, 4), ",", round(p + hw, 4), "]")
z2 = z * z; c = (p + z2 / (2 * N)) / (1 + z2 / N); h = z / (1 + z2 / N) * math.sqrt(p * (1 - p) / N + z2 / (4 * N * N))
print("   Wilson: z²/n =", round(z2 / N, 4), " centro", round(c, 5), " p(1−p)/n + z²/4n² =", round(p * (1 - p) / N + z2 / (4 * N * N), 6), " ±", round(h, 5), " [", round(c - h, 4), ",", round(c + h, 4), "]")
r = np.random.default_rng(2024)
for j in range(3):
    i = r.integers(0, n, n); print(f"9. remuestreo {j + 1}: posiciones {(i + 1).tolist()}  media {x[i].mean():.2f}")
res = stats.bootstrap((x,), np.mean, n_resamples=10000, method="percentile", rng=np.random.default_rng(7))
print("   10 000 remuestreos: percentil [", round(res.confidence_interval.low, 2), ",", round(res.confidence_interval.high, 2), "]  error estándar bootstrap", round(res.standard_error, 3),
      " (s/√n·√((n−1)/n) =", round(se * math.sqrt(0.9), 3), ")")
