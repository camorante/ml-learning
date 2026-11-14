"""Cada pérdida es una log-verosimilitud negativa media: gaussiana → MSE, Laplace → MAE,
Bernoulli → BCE, categórica → CE. Comprobado con scipy.stats y torch.distributions."""
import numpy as np
import torch
from scipy import stats
import perdidas as P

rng = np.random.default_rng(14)
n = 1000
yhat = rng.uniform(25, 45, n)                       # consumo que predice la red (L/100 km)
y = yhat + rng.normal(0, 1.5, n)                    # lo medido: predicción + ruido gaussiano de σ = 1,5
s = 1.5
nll = -np.mean(stats.norm.logpdf(y, yhat, s))
print(f"gaussiana σ = {s}: −log-verosimilitud media {nll:.6f}")
print(f"   MSE/(2σ²) + ½·log(2πσ²)              {P.mse(yhat, y) / (2 * s**2) + 0.5 * np.log(2 * np.pi * s**2):.6f}")
print(f"   torch GaussianNLLLoss(full=True)       "
      f"{torch.nn.GaussianNLLLoss(full=True)(torch.tensor(yhat), torch.tensor(y), torch.full((n,), s**2)).item():.6f}")
print(f"   σ² que maximiza la verosimilitud = MSE = {P.mse(yhat, y):.4f}  (σ = {np.sqrt(P.mse(yhat, y)):.3f})")
bL = 1.2
print(f"Laplace b = {bL}: −log-verosimilitud {-np.mean(stats.laplace.logpdf(y, yhat, bL)):.6f}"
      f" = MAE/b + log(2b) = {P.mae(yhat, y) / bL + np.log(2 * bL):.6f}")
z = rng.normal(-1.5, 2, n); yb = (rng.random(n) < P.sigmoide(z)).astype(int)
print(f"Bernoulli: −log-verosimilitud {-np.mean(stats.bernoulli.logpmf(yb, P.sigmoide(z))):.6f}"
      f" = BCE desde logits {P.bce_logits(z, yb):.6f}")
Z = rng.normal(0, 2, (n, 3)); c = rng.integers(0, 3, n)
lp = torch.distributions.Categorical(logits=torch.tensor(Z)).log_prob(torch.tensor(c)).numpy()
print(f"categórica: −log-verosimilitud {-lp.mean():.6f} = CE desde logits {P.ce_logits(Z, c):.6f}")
