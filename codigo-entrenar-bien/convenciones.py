"""Dónde se separan las convenciones: momento «de libro» frente al de PyTorch, ε dentro o fuera de la raíz,
y el ε de Adam antes o después de corregir el sesgo. Un solo peso y gradientes fijos para verlo claro."""
import numpy as np
import torch

torch.set_default_dtype(torch.float64)
g = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])            # gradiente constante, 6 pasos
lrs = [0.1, 0.1, 0.1, 0.01, 0.01, 0.01]                  # la tasa se divide por 10 en el paso 4

# 1) momento: Polyak/Sutskever (v ← μv − ηg; θ ← θ + v) frente a PyTorch (v ← μv + g; θ ← θ − ηv)
th_a = th_b = 0.0; va = vb = 0.0
for lr, gi in zip(lrs, g):
    va = 0.9 * va - lr * gi; th_a += va
    vb = 0.9 * vb + gi; th_b -= lr * vb
print(f"1) momento 0.9, tasa 0.1 → 0.01 en el paso 4:  de libro θ = {th_a:.4f}   PyTorch θ = {th_b:.4f}")
th_a = th_b = 0.0; va = vb = 0.0
for gi in g:
    va = 0.9 * va - 0.1 * gi; th_a += va
    vb = 0.9 * vb + gi; th_b -= 0.1 * vb
print(f"   con la tasa fija en 0.1 son idénticos:      de libro θ = {th_a:.4f}   PyTorch θ = {th_b:.4f}")

# 2) dampening: PyTorch NO amortigua el primer paso (v₁ = g)
p = torch.zeros(1, requires_grad=True); o = torch.optim.SGD([p], lr=1.0, momentum=0.9, dampening=0.9)
pasos = []
for _ in range(3):
    p.grad = torch.ones(1); o.step(); pasos.append(-p.item() - sum(pasos))
v, alt = 0.0, []
for _ in range(3):
    v = 0.9 * v + 0.1 * 1.0; alt.append(v)                # si v empezara en 0 y se amortiguara desde el principio
print(f"2) dampening 0.9, g = 1: pasos de PyTorch {np.round(pasos, 4)}   con v₀ = 0 serían {np.round(alt, 4)}")

# 3) RMSprop con gradientes diminutos: ε fuera de la raíz (PyTorch) o dentro (tf.train.RMSPropOptimizer de TensorFlow 1)
for gi in (1e-3, 1e-6, 1e-9):
    s = 0.01 * gi ** 2                                    # tras el primer paso, α = 0.99
    print(f"3) g = {gi:.0e}: paso/η con ε fuera = {gi / (np.sqrt(s) + 1e-8):.4f}   con ε = 1e-8 dentro = {gi / np.sqrt(s + 1e-8):.2e}")

# 4) Adam: ε tras la corrección (PyTorch, algoritmo 1 del artículo) frente a ε̂ sin corregir (versión «eficiente»)
gi = 1e-6
for t in (1, 10, 100):
    m = (1 - 0.9 ** t) * gi; v = (1 - 0.999 ** t) * gi ** 2   # medias móviles con g constante
    a = (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    b = (m / (1 - 0.9 ** t)) * np.sqrt(1 - 0.999 ** t) / (np.sqrt(v) + 1e-8)
    print(f"4) g = 1e-6, paso {t:3d}: paso/η PyTorch {a:.4f}   ε̂ sin corregir {b:.4f}")
