import warnings
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from neurona import NeuronaSigmoide, Perceptron
from datos import X, y, nuevas

Xn, yn = nuevas()                                   # 1 000 jornadas con etiquetas ruidosas
# 1) neurona sigmoide + entropía cruzada (Newton) frente a scikit-learn y torch
ns = NeuronaSigmoide().fit(Xn, yn)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    lr = LogisticRegression(penalty=None, tol=1e-12, max_iter=10000).fit(Xn, yn)
t = torch.zeros(3, dtype=torch.float64, requires_grad=True)
Xt, yt = torch.tensor(Xn), torch.tensor(yn, dtype=torch.float64)
opt = torch.optim.LBFGS([t], lr=1, max_iter=500, tolerance_grad=1e-14, tolerance_change=1e-16,
                        line_search_fn="strong_wolfe")
def cierre():
    opt.zero_grad()
    L = torch.nn.functional.binary_cross_entropy_with_logits(Xt @ t[:2] + t[2], yt)
    L.backward(); return L
for _ in range(5):
    opt.step(cierre)
nuestro, sk, to = np.r_[ns.w, ns.b], np.r_[lr.coef_[0], lr.intercept_[0]], t.detach().numpy()
print(f"neurona sigmoide (Newton): w = {np.round(ns.w, 6)}  b = {ns.b:.6f}"
      f"   frontera: frenadas + {ns.w[1] / ns.w[0]:.3f}·horas = {-ns.b / ns.w[0]:.3f}")
print(f"   vs LogisticRegression {np.max(np.abs(nuestro - sk)):.1e}   vs torch LBFGS {np.max(np.abs(nuestro - to)):.1e}")
print(f"   pérdida media {ns.perdida(Xn, yn, ns.w, ns.b):.6f}   acierto {np.mean((ns.predict_proba(Xn) > 0.5) == yn):.3f}")
u = torch.tensor(nuestro, requires_grad=True)
torch.nn.functional.binary_cross_entropy_with_logits(Xt @ u[:2] + u[2], yt).backward()
print(f"   gradiente de torch (autograd) en nuestra solución: ‖∇L‖ = {u.grad.norm():.1e}")

# 2) con datos separables (las 20 jornadas) no hay mínimo finito: los pesos crecen sin parar
for it in (100, 1000, 10000, 100000):
    m = NeuronaSigmoide().fit(X, y, metodo="gd", eta=0.5, iters=it, tol=0)
    print(f"descenso de gradiente, {it:>6} pasos: ‖w‖ = {np.linalg.norm(m.w):6.2f}"
          f"   pérdida {m.perdida(X, y, m.w, m.b):.5f}   frontera: frenadas + {m.w[1] / m.w[0]:.3f}·horas = {-m.b / m.w[0]:.3f}")
m = NeuronaSigmoide(alfa=0.01).fit(X, y)
print(f"con freno L2 (α = 0.01): ‖w‖ = {np.linalg.norm(m.w):.2f}"
      f"   frontera: frenadas + {m.w[1] / m.w[0]:.3f}·horas = {-m.b / m.w[0]:.3f}")
