"""celdas.py frente a torch.nn.RNN y torch.nn.LSTM (copiando los pesos) y su autograd, en float64, y frente a
diferencias centrales. Error absoluto máximo de los estados y de cada gradiente."""
import numpy as np
import torch
from torch import nn
from celdas import rnn_adelante, rnn_atras, lstm_adelante, lstm_atras

torch.set_default_dtype(torch.float64)
rng = np.random.default_rng(9)
torch.manual_seed(9)                                # los pesos al azar de nn.RNN y nn.LSTM, reproducibles
err = lambda a, t: float(np.max(np.abs(a - (t.detach().numpy() if torch.is_tensor(t) else t))))
print(f"PyTorch {torch.__version__}, float64 · error absoluto máximo")
n, T, D, H = 4, 30, 3, 5
X, h0, c0 = rng.normal(size=(n, T, D)), rng.normal(size=(n, H)), rng.normal(size=(n, H))
dH = rng.normal(size=(n, T, H))                     # un gradiente «de arriba» cualquiera sobre cada h_t

# ---------- RNN ----------
red = nn.RNN(D, H, batch_first=True)                 # pesos al azar de PyTorch
W, U = red.weight_ih_l0.detach().numpy(), red.weight_hh_l0.detach().numpy()
b = (red.bias_ih_l0 + red.bias_hh_l0).detach().numpy()   # PyTorch tiene dos sesgos; solo cuenta su suma
Hn, cache = rnn_adelante(X, W, U, b, h0)
dX, dW, dU, db, dh0 = rnn_atras(dH, cache)
Xt, h0t = torch.tensor(X, requires_grad=True), torch.tensor(h0[None], requires_grad=True)
Ht, _ = red(Xt, h0t); (Ht * torch.tensor(dH)).sum().backward()
print(f"RNN  T={T}:  h {err(Hn, Ht):.1e}  ∂x {err(dX, Xt.grad):.1e}  ∂W {err(dW, red.weight_ih_l0.grad):.1e}"
      f"  ∂U {err(dU, red.weight_hh_l0.grad):.1e}  ∂b_ih {err(db, red.bias_ih_l0.grad):.1e}"
      f"  ∂b_hh {err(db, red.bias_hh_l0.grad):.1e}  ∂h0 {err(dh0, h0t.grad[0]):.1e}")

# ---------- LSTM ----------
red = nn.LSTM(D, H, batch_first=True)
W, U = red.weight_ih_l0.detach().numpy(), red.weight_hh_l0.detach().numpy()   # (4H, D) y (4H, H), bloques i, f, g, o
b = (red.bias_ih_l0 + red.bias_hh_l0).detach().numpy()
Hn, Cn, cache = lstm_adelante(X, W, U, b, h0, c0)
dX, dW, dU, db, dh0, dc0 = lstm_atras(dH, cache)
Xt = torch.tensor(X, requires_grad=True)
h0t, c0t = torch.tensor(h0[None], requires_grad=True), torch.tensor(c0[None], requires_grad=True)
Ht, (hT, cT) = red(Xt, (h0t, c0t)); (Ht * torch.tensor(dH)).sum().backward()
print(f"LSTM T={T}:  h {err(Hn, Ht):.1e}  c_T {err(Cn[:, -1], cT[0]):.1e}  ∂x {err(dX, Xt.grad):.1e}"
      f"  ∂W {err(dW, red.weight_ih_l0.grad):.1e}  ∂U {err(dU, red.weight_hh_l0.grad):.1e}"
      f"  ∂b {err(db, red.bias_ih_l0.grad):.1e}  ∂h0 {err(dh0, h0t.grad[0]):.1e}  ∂c0 {err(dc0, c0t.grad[0]):.1e}")

# el orden de las puertas importa: leer los bloques en un orden equivocado (i, f, o, g) da otra red
i, f, g, o = np.split(W, 4); j, k, l, m = np.split(U, 4); bi, bf, bg, bo = np.split(b, 4)
H2, _, _ = lstm_adelante(X, np.r_[i, f, o, g], np.r_[j, k, m, l], np.r_[bi, bf, bo, bg], h0, c0)
print(f"LSTM con los bloques leídos en orden i, f, o, g: h difiere en {err(H2, Ht):.2f}")

# ---------- diferencias centrales sobre una pérdida escalar ----------
Wl, Ul, bl = rng.normal(size=(4 * H, D)) * 0.5, rng.normal(size=(4 * H, H)) * 0.5, rng.normal(size=4 * H) * 0.5
C = rng.normal(size=(n, T, H))
perdida = lambda U_: float((C * lstm_adelante(X, Wl, U_, bl, h0, c0)[0]).sum())
_, _, cache = lstm_adelante(X, Wl, Ul, bl, h0, c0); dUl = lstm_atras(C, cache)[2]
eps, num = 1e-6, np.zeros_like(Ul)
for idx in np.ndindex(Ul.shape):
    Up, Um = Ul.copy(), Ul.copy(); Up[idx] += eps; Um[idx] -= eps
    num[idx] = (perdida(Up) - perdida(Um)) / (2 * eps)
print(f"LSTM: ∂U frente a diferencias centrales (h = 1e-6): {np.max(np.abs(num - dUl)):.1e}")

# ---------- la LSTM entrenada (comparar.py), hacia delante solo con celdas.py ----------
from datos import datos
from modelos import Lectora, ESCALA
m = Lectora("LSTM", 16); m.load_state_dict(torch.load("red_LSTM16_G100.pt"))
P = {k: v.detach().numpy() for k, v in m.state_dict().items()}
_, _, (Xte, Yte, _) = datos(100)
Hn, _, _ = lstm_adelante(Xte[:, :, None] / ESCALA, P["rec.weight_ih_l0"], P["rec.weight_hh_l0"],
                         P["rec.bias_ih_l0"] + P["rec.bias_hh_l0"], np.zeros((len(Xte), 16)), np.zeros((len(Xte), 16)))
z = Hn[:, -1] @ P["out.weight"][0] + P["out.bias"][0]
with torch.no_grad():
    zt = m(torch.tensor(Xte)).numpy()
print(f"LSTM entrenada, 1 000 tramos de 116 lecturas: logits frente a PyTorch {np.max(np.abs(z - zt)):.1e},"
      f" acierto {np.mean((z > 0) == Yte):.1%}")
