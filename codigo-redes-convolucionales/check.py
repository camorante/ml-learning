"""conv.py frente a torch.nn.functional (conv1d, conv2d, max_pool1d) y su autograd, en float64, y frente a
diferencias centrales. Error absoluto máximo de la salida y de cada gradiente."""
import numpy as np
import torch
import torch.nn.functional as Fn
from conv import conv1d, conv1d_atras, conv2d, conv2d_atras, maxpool1d, maxpool1d_atras

torch.set_default_dtype(torch.float64)
rng = np.random.default_rng(8)
err = lambda a, t: float(np.max(np.abs(a - t.detach().numpy())))
print(f"PyTorch {torch.__version__}, float64 · error absoluto máximo")


def prueba1d(n, C, L, F, k, s, p):
    x, W, b = rng.normal(size=(n, C, L)), rng.normal(size=(F, C, k)), rng.normal(size=F)
    y, cache = conv1d(x, W, b, s, p)
    dy = rng.normal(size=y.shape)                                     # un gradiente «de arriba» cualquiera
    dx, dW, db = conv1d_atras(dy, cache)
    xt, Wt, bt = (torch.tensor(a, requires_grad=True) for a in (x, W, b))
    yt = Fn.conv1d(xt, Wt, bt, stride=s, padding=p); (yt * torch.tensor(dy)).sum().backward()
    print(f"conv1d C={C} L={L} F={F} k={k} paso={s} relleno={p} → salida {y.shape[2]:2d}:"
          f"  y {err(y, yt):.1e}  ∂x {err(dx, xt.grad):.1e}  ∂W {err(dW, Wt.grad):.1e}  ∂b {err(db, bt.grad):.1e}")


prueba1d(4, 1, 64, 4, 5, 1, 0)
prueba1d(4, 4, 30, 8, 5, 1, 0)
prueba1d(3, 3, 17, 5, 3, 2, 1)
prueba1d(2, 2, 20, 3, 4, 3, 2)

x, W, b = rng.normal(size=(2, 3, 16, 16)), rng.normal(size=(6, 3, 3, 3)), rng.normal(size=6)
for s, p in [(1, 0), (2, 1)]:
    y, cache = conv2d(x, W, b, s, p); dy = rng.normal(size=y.shape); dx, dW, db = conv2d_atras(dy, cache)
    xt, Wt, bt = (torch.tensor(a, requires_grad=True) for a in (x, W, b))
    yt = Fn.conv2d(xt, Wt, bt, stride=s, padding=p); (yt * torch.tensor(dy)).sum().backward()
    print(f"conv2d 3×16×16 → 6 filtros 3×3 paso={s} relleno={p} → {y.shape[2]}×{y.shape[3]}:"
          f"  y {err(y, yt):.1e}  ∂x {err(dx, xt.grad):.1e}  ∂W {err(dW, Wt.grad):.1e}  ∂b {err(db, bt.grad):.1e}")

x = rng.normal(size=(4, 4, 60))
for k in (2, 3):
    y, cache = maxpool1d(x, k); dy = rng.normal(size=y.shape); dx = maxpool1d_atras(dy, cache)
    xt = torch.tensor(x, requires_grad=True); yt = Fn.max_pool1d(xt, k); (yt * torch.tensor(dy)).sum().backward()
    print(f"max_pool1d k={k}, 60 → {y.shape[2]}:  y {err(y, yt):.1e}  ∂x {err(dx, xt.grad):.1e}")

# diferencias centrales sobre una pérdida escalar: L = Σ c·maxpool(conv1d(x))
x, W, b = rng.normal(size=(2, 2, 12)), rng.normal(size=(3, 2, 3)), rng.normal(size=3)
c = rng.normal(size=(2, 3, 5))
def perdida(W):
    return float((c * maxpool1d(conv1d(x, W, b)[0], 2)[0]).sum())
y, cc = conv1d(x, W, b); z, cp = maxpool1d(y, 2); _, dW, _ = conv1d_atras(maxpool1d_atras(c, cp), cc)
h, num = 1e-6, np.zeros_like(W)
for i in np.ndindex(W.shape):
    Wp, Wm = W.copy(), W.copy(); Wp[i] += h; Wm[i] -= h
    num[i] = (perdida(Wp) - perdida(Wm)) / (2 * h)
print(f"conv1d + max-pool encadenados: ∂W frente a diferencias centrales {np.max(np.abs(num - dW)):.1e}")

# PyTorch hace correlación cruzada: el resultado es la convolución de NumPy con el filtro dado la vuelta
x1, w1 = rng.normal(size=10), np.array([1.0, 2.0, -1.0])
t = Fn.conv1d(torch.tensor(x1)[None, None], torch.tensor(w1)[None, None])[0, 0].numpy()
print(f"conv1d de PyTorch frente a np.correlate: {np.max(np.abs(t - np.correlate(x1, w1, 'valid'))):.1e};"
      f" frente a np.convolve(w): {np.max(np.abs(t - np.convolve(x1, w1, 'valid'))):.2f};"
      f" frente a np.convolve(w al revés): {np.max(np.abs(t - np.convolve(x1, w1[::-1], 'valid'))):.1e}")
# el gradiente de los pesos es la correlación de la entrada con dy
dy1 = rng.normal(size=8); _, cache = conv1d(x1[None, None], w1[None, None], np.zeros(1))
print(f"∂W = correlación(x, dy): {np.max(np.abs(conv1d_atras(dy1[None, None], cache)[1][0, 0] - np.correlate(x1, dy1, 'valid'))):.1e};"
      f"  ∂x = convolución completa(dy, W): {np.max(np.abs(conv1d_atras(dy1[None, None], cache)[0][0, 0] - np.convolve(dy1, w1, 'full'))):.1e}")

# la red convolucional entrenada (comparar.py), hacia delante solo con conv.py: los mismos logits que PyTorch
from datos import X_TE
from modelos import CNN
m = CNN(); m.load_state_dict(torch.load("CNN_A.pt"))
P = {k: v.detach().numpy() for k, v in m.state_dict().items()}
h = np.maximum(conv1d(X_TE[:, None, :], P["c1.weight"], P["c1.bias"])[0], 0)
h = np.maximum(conv1d(maxpool1d(h, 2)[0], P["c2.weight"], P["c2.bias"])[0], 0)
z = h.max(-1) @ P["out.weight"][0] + P["out.bias"][0]
with torch.no_grad():
    print(f"red convolucional entrenada, 1 000 ventanas de prueba: logits frente a PyTorch {np.max(np.abs(z - m(torch.tensor(X_TE)).numpy())):.1e}")
