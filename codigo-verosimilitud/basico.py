import numpy as np
from scipy import stats, optimize
from sim import ALERTAS_A as x                      # [3, 1, 4, 0, 4, 2, 3, 1, 3, 2, 1, 6, 3, 2]

ll = lambda lam: stats.poisson.logpmf(x, lam).sum()  # ℓ(λ) = Σ log p(xᵢ | λ)
rejilla = np.linspace(0.1, 6, 5901)
print("cima en la rejilla:", rejilla[np.argmax([ll(l) for l in rejilla])])
r = optimize.minimize_scalar(lambda l: -ll(l), bounds=(0.01, 20), method="bounded")
print("minimize_scalar   :", round(r.x, 6), " fórmula cerrada x̄ =", x.mean())
print("ℓ(λ̂) =", round(ll(x.mean()), 4), " producto =", f"{np.prod(stats.poisson.pmf(x, x.mean())):.4e}")
for lam in (1, 2, 3):
    print(f"λ = {lam}: ℓ = {ll(lam):.3f}   L(λ̂)/L(λ) = {np.exp(ll(x.mean()) - ll(lam)):.1f}")
