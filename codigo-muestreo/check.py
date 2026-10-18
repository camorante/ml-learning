import itertools
import numpy as np
from scipy import stats
from muestreo import media_se, autocorr, n_efectivo, estratificado, asignar

rng = np.random.default_rng(4)
err = {}

# 1) error estándar frente a scipy.stats.sem, 200 muestras de tamaños y escalas variadas
e = 0.0
for _ in range(200):
    x = rng.lognormal(3, 1, rng.integers(2, 500)) * 10 ** rng.uniform(-3, 3)
    m, se = media_se(x)
    e = max(e, abs(se - stats.sem(x)) / stats.sem(x), abs(m - x.mean()) / x.mean())
err["s/√n frente a scipy.stats.sem (relativo)"] = e

# 2) población finita: TODAS las muestras sin reemplazo de 4 entre 10
pop = rng.lognormal(3, 1, 10); N, n = 10, 4
xb = np.array([pop[list(c)].mean() for c in itertools.combinations(range(N), n)])
formula = pop.var() / n * (N - n) / (N - 1)
err["Var(x̄) con fpc frente a las 210 muestras posibles"] = abs(xb.var() - formula) / formula
# y la media de s²/n·(1 − n/N) sobre las 210 muestras es exactamente esa varianza (insesgado)
v = np.mean([media_se(pop[list(c)], N)[1] ** 2 for c in itertools.combinations(range(N), n)])
err["E[se² con fpc] = Var(x̄) exacta"] = abs(v - formula) / formula

# 3) estratificado: todas las combinaciones de dos estratos (N_h = 6 y 5, n_h = 3 y 2)
A, B = rng.lognormal(2.6, .5, 6), rng.lognormal(4, .6, 5)
est, var_est = [], []
for ca in itertools.combinations(range(6), 3):
    for cb in itertools.combinations(range(5), 2):
        y = np.r_[A[list(ca)], B[list(cb)]]; h = np.r_[[0] * 3, [1] * 2]
        m, se = estratificado(y, h, [6, 5]); est.append(m); var_est.append(se**2)
est = np.array(est); mu = np.r_[A, B].mean()
err["estratificado: sesgo exacto"] = abs(est.mean() - mu) / mu
err["estratificado: E[se²] = Var exacta"] = abs(np.mean(var_est) - est.var()) / est.var()

# 4) tamaño efectivo frente a 1ᵀΣ1/n² con una autocorrelación AR(2) válida
a1, a2, n = 0.5, 0.3, 50
rho = [1.0, a1 / (1 - a2)]
for k in range(2, n):
    rho.append(a1 * rho[-1] + a2 * rho[-2])
rho = np.array(rho)
S = rho[np.abs(np.subtract.outer(np.arange(n), np.arange(n)))]
err["n_ef frente a la varianza exacta con Σ"] = abs(n_efectivo(n, rho[1:]) - n**2 / S.sum()) / (n**2 / S.sum())

# 5) autocorrelación frente a np.correlate
x = rng.normal(size=300).cumsum(); dx = x - x.mean()
full = np.correlate(dx, dx, "full")[len(x) - 1:] / (dx @ dx)
err["autocorr frente a np.correlate (absoluto)"] = max(abs(autocorr(x, k) - full[k]) for k in range(1, 40))

# 6) el reparto siempre suma n y respeta las proporciones
ok = all(asignar(n, Nh, Sh).sum() == n for n, Nh, Sh in
         [(rng.integers(5, 500), rng.integers(10, 9999, 4), rng.uniform(1, 50, 4)) for _ in range(500)])

for k, v in err.items():
    print(f"{k:52s} {v:.1e}")
print("asignar() suma n en 500 casos:", ok)
