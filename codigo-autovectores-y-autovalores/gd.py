import numpy as np
# f(w) = ½ wᵀHw con H = diag(λmax, λmin): descenso de gradiente w ← w − η H w con el mejor paso fijo η = 2/(λmax+λmin)
print(" κ = λmax/λmin   factor por paso (κ−1)/(κ+1)   pasos hasta ‖w‖ < 1e-6‖w₀‖   con momento (heavy ball)")
for kappa in [1, 10, 100, 1000]:
    H = np.diag([float(kappa), 1.0]); eta = 2 / (kappa + 1)
    w = np.array([1.0, 1.0]); k = 0
    while np.linalg.norm(w) > 1e-6 * np.sqrt(2): w = w - eta * H @ w; k += 1
    # Polyak: η = 4/(√L+√μ)², β = ((√κ−1)/(√κ+1))²
    eta2 = 4 / (np.sqrt(kappa) + 1) ** 2; beta = ((np.sqrt(kappa) - 1) / (np.sqrt(kappa) + 1)) ** 2
    w = np.array([1.0, 1.0]); wp = w.copy(); j = 0
    while np.linalg.norm(w) > 1e-6 * np.sqrt(2) and j < 10**6: w, wp = w - eta2 * H @ w + beta * (w - wp), w; j += 1
    print(f"{kappa:8d}          {(kappa - 1) / (kappa + 1):.4f}                     {k:6d}                        {j:5d}")
