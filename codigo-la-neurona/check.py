"""Verificación de neurona.py contra scikit-learn, torch (autograd en float64) y diferencias finitas."""
import warnings
import numpy as np
import torch
from sklearn.linear_model import Perceptron as SkPerceptron, LogisticRegression
from neurona import Perceptron, NeuronaSigmoide, sigmoide

rng = np.random.default_rng(0)
warnings.simplefilter("ignore")

# 1) perceptrón: mismos pesos que sklearn.linear_model.Perceptron (sin barajar, η constante)
e_p, n_sep = 0.0, 0
for k in range(200):
    n, d = rng.integers(20, 80), rng.integers(2, 6)
    Xr = rng.normal(size=(n, d)) * rng.uniform(0.5, 5, d)
    yr = (Xr @ rng.normal(size=d) + rng.normal(0, 0.5 if k % 2 else 0.0, n) > 0).astype(int)
    eta, E = rng.uniform(0.05, 1), 30
    a = Perceptron(eta=eta, epocas=E).fit(Xr, yr)
    b = SkPerceptron(eta0=eta, shuffle=False, tol=None, max_iter=E, penalty=None).fit(Xr, yr)
    e_p = max(e_p, np.max(np.abs(np.r_[a.w - b.coef_[0], a.b - b.intercept_[0]])))
    n_sep += a.fallos_por_pasada[-1] == 0
print(f"perceptrón vs sklearn Perceptron, 200 problemas ({n_sep} acaban sin fallos): diferencia máxima {e_p:.1e}")

# 2) gradiente de la entropía cruzada: fórmula (p − y)x frente a torch y diferencias centrales
ns = NeuronaSigmoide(alfa=0.1)
e_t = e_f = 0.0
for _ in range(100):
    n, d = 30, 3
    Xr, yr = rng.normal(size=(n, d)), (rng.random(n) < 0.5).astype(float)
    w, b = rng.normal(size=d), rng.normal()
    gw, gb = ns.gradiente(Xr, yr, w, b); g = np.r_[gw, gb]
    th = torch.tensor(np.r_[w, b], requires_grad=True)
    z = torch.tensor(Xr) @ th[:d] + th[d]
    L = torch.nn.functional.binary_cross_entropy_with_logits(z, torch.tensor(yr)) + 0.05 * th[:d] @ th[:d]
    L.backward()
    e_t = max(e_t, np.max(np.abs(g - th.grad.numpy())))
    h, th0, fd = 1e-6, np.r_[w, b], np.zeros(d + 1)
    for j in range(d + 1):
        u = np.zeros(d + 1); u[j] = h
        fd[j] = (ns.perdida(Xr, yr, (th0 + u)[:d], (th0 + u)[d]) - ns.perdida(Xr, yr, (th0 - u)[:d], (th0 - u)[d])) / (2 * h)
    e_f = max(e_f, np.max(np.abs(g - fd)))
print(f"gradiente (1/n)Xᵀ(p − y) + αw, 100 casos: vs torch autograd {e_t:.1e}   vs diferencias centrales {e_f:.1e}")

# 3) neurona sigmoide entrenada = regresión logística de sklearn (sin y con regularización L2)
e0 = e1 = 0.0
for _ in range(50):
    n, d = 200, 3
    Xr = rng.normal(size=(n, d))
    yr = (rng.random(n) < sigmoide(Xr @ rng.normal(size=d) + rng.normal())).astype(int)
    a = NeuronaSigmoide().fit(Xr, yr)
    b = LogisticRegression(penalty=None, tol=1e-12, max_iter=10000).fit(Xr, yr)
    e0 = max(e0, np.max(np.abs(np.r_[a.w - b.coef_[0], a.b - b.intercept_[0]])))
    C = 0.5                                   # sklearn: C·Σ pérdida + ½‖w‖²  ⇔  media + ‖w‖²/(2Cn)
    a = NeuronaSigmoide(alfa=1 / (C * n)).fit(Xr, yr)
    b = LogisticRegression(C=C, tol=1e-12, max_iter=10000).fit(Xr, yr)
    e1 = max(e1, np.max(np.abs(np.r_[a.w - b.coef_[0], a.b - b.intercept_[0]])))
print(f"neurona sigmoide vs LogisticRegression, 50 problemas: sin penalización {e0:.1e}   con L2 (C = 0.5) {e1:.1e}")
