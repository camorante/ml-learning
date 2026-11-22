"""autoencoder.py frente al mismo modelo de PyTorch con los pesos copiados (float64) y su autograd, y frente a
diferencias centrales. Errores absolutos máximos."""
import numpy as np
import torch
from autoencoder import Autoencoder
from modelos import AE, error
from datos import X_TR, X_TE

torch.set_default_dtype(torch.float64)
err = lambda a, t: float(np.max(np.abs(a - (t.detach().numpy() if torch.is_tensor(t) else t))))
print(f"PyTorch {torch.__version__}, float64 · error absoluto máximo")

torch.manual_seed(12); mt = AE(k=4)
capas = [mt.cod[0], mt.cod[2], mt.dec[0], mt.dec[2]]
pesos = [(c.weight.detach().numpy().copy(), c.bias.detach().numpy().copy()) for c in capas]
x = X_TR[:64]

# 1. hacia delante y hacia atrás en un lote de 64 ventanas
m = Autoencoder(pesos); L = m.perdida(x); grads, dx = m.atras()
xt = torch.tensor(x, requires_grad=True); Lt = ((mt(xt) - xt.detach()) ** 2).mean(); Lt.backward()
print(f"pérdida {err(L, Lt):.1e} · código {err(m.z, mt.cod(xt)):.1e} · reconstrucción {err(m.xr, mt(xt)):.1e}")
print("gradientes  " + " · ".join(f"capa {i + 1}: W {err(dW, c.weight.grad):.1e}, b {err(db, c.bias.grad):.1e}"
                                   for i, ((dW, db), c) in enumerate(zip(grads, capas))))
print(f"∂pérdida/∂x (con el objetivo fijo) {err(dx, xt.grad):.1e}")

# 2. diferencias centrales en 40 pesos al azar de cada capa
rng, h = np.random.default_rng(0), 1e-6
peor = 0.0
for i, (W, b) in enumerate(pesos):
    for _ in range(40):
        r, c = rng.integers(W.shape[0]), rng.integers(W.shape[1])
        P1 = [(w.copy(), bb.copy()) for w, bb in pesos]; P2 = [(w.copy(), bb.copy()) for w, bb in pesos]
        P1[i][0][r, c] += h; P2[i][0][r, c] -= h
        num = (Autoencoder(P1).perdida(x) - Autoencoder(P2).perdida(x)) / (2 * h)
        peor = max(peor, abs(num - grads[i][0][r, c]))
print(f"160 pesos frente a diferencias centrales (h = 1e-6): {peor:.1e}")

# 3. 30 pasos de descenso de gradiente, NumPy frente a torch.optim.SGD
m = Autoencoder(pesos); opt = torch.optim.SGD(mt.parameters(), lr=0.5)
for k in range(30):
    xb = X_TR[64 * k:64 * k + 64]
    m.paso(xb, 0.5)
    opt.zero_grad(); t = torch.tensor(xb); ((mt(t) - t) ** 2).mean().backward(); opt.step()
print("30 pasos de SGD (lr 0,5): " + " · ".join(f"W{i + 1} {err(c.W, cap.weight):.1e}" for i, (c, cap) in enumerate(zip(m.lineales, capas))))

# 4. el detector entrenado (detector.py), evaluado solo con autoencoder.py
sd = torch.load("detector.pt"); md = AE(k=4); md.load_state_dict(sd)
mn = Autoencoder([(sd[f"{p}.weight"].numpy(), sd[f"{p}.bias"].numpy()) for p in ("cod.0", "cod.2", "dec.0", "dec.2")])
e_np = ((mn.adelante(X_TE) - X_TE) ** 2).mean(1)
print(f"detector, {len(X_TE)} ventanas de prueba: error de reconstrucción NumPy frente a PyTorch {err(e_np, error(md, X_TE)):.1e}")
