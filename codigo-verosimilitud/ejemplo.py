import math
from scipy import stats
from sim import ALERTAS_A as x
from mv import ll_poisson, intervalo_perfil, razon_verosimilitud, map_binom

S, n = int(x.sum()), len(x); C = sum(math.lgamma(k + 1) for k in x)       # Σ log k!
print("1) Σk =", S, " n =", n, " Σ log k! =", round(C, 4))
ll = lambda l: S * math.log(l) - n * l - C
print("2) λ̂ = S/n =", S / n, " ℓ(λ̂) =", round(ll(S / n), 4), " (scipy:", round(ll_poisson(2.5, x), 4), ")")
print("3) ℓ(1) =", round(ll(1), 4), " cociente L(2,5)/L(1) = e^%.4f = %.1f" % (ll(2.5) - ll(1), math.exp(ll(2.5) - ll(1))))
I = S / (S / n) ** 2
print("4) −ℓ''(λ̂) = S/λ̂² =", I, " ee =", round(1 / math.sqrt(I), 4),
      " Wald:", [round(2.5 + s * 1.96 / math.sqrt(I), 3) for s in (-1, 1)])
c = stats.chi2.ppf(0.95, 1) / 2
lo, hi = intervalo_perfil(ll, 2.5, 1e-6, 20)
print("5) umbral ℓ(λ̂) − %.4f = %.4f  perfil [%.3f, %.3f]  ℓ en los extremos: %.4f %.4f" % (c, ll(2.5) - c, lo, hi, ll(lo), ll(hi)))
d, p = razon_verosimilitud(ll(2.0), ll(2.5))
print("6) ℓ(2) =", round(ll(2.0), 4), " 2Δℓ =", round(d, 3), " p =", round(p, 3))
print("7) 0 de 5: MV 0;  MAP con m = 20 al 8 %%: (0 + 1,6)/(5 + 20) = %.4f" % map_binom(0, 5, 20, 0.08))
z = 2.11799 * (94 - 163.804) / 100; pz = 1 / (1 + math.exp(-z))
print("8) entrega de 94 km, tarde: z = %.4f  p = %.4f  castigo −log p = %.4f" % (z, pz, -math.log(pz)))
print("   castigo si hubiera llegado a tiempo: −log(1 − p) = %.4f" % -math.log(1 - pz))
