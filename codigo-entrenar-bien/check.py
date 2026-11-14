"""Verificación de optim.py y red.py contra PyTorch (float64, CPU): mismos pesos iniciales, mismos gradientes,
N pasos, error máximo. Y el entrenamiento completo, paso a paso, frente al de torch.nn + torch.optim."""
import numpy as np
import torch
import red
from optim import SGD, RMSprop, Adam
from datos import X_TR, Y_TR, escalar

torch.set_default_dtype(torch.float64)
rng = np.random.default_rng(5)
N = 500
P0 = [rng.normal(0, 1, s) for s in [(8, 2), (8,), (1, 8), (1,)]]          # pesos iniciales comunes
GS = [[rng.normal(0, 1, p.shape) * 10 ** rng.uniform(-3, 1) for p in P0] for _ in range(N)]   # N gradientes


def mismos_gradientes(nuestro, suyo):
    """Aplica los mismos N gradientes a nuestro optimizador y al de PyTorch y compara los parámetros."""
    P = [p.copy() for p in P0]
    T = [torch.tensor(p, requires_grad=True) for p in P0]
    o, ot, err = nuestro(), suyo(T), 0.0
    for G in GS:
        o.step(P, G)
        for t, g in zip(T, G):
            t.grad = torch.tensor(g)
        ot.step()
        err = max(err, max(np.max(np.abs(p - t.detach().numpy())) for p, t in zip(P, T)))
    return err


casos = [("SGD lr=0.01", lambda: SGD(0.01), lambda T: torch.optim.SGD(T, lr=0.01)),
         ("SGD momentum=0.9", lambda: SGD(0.01, 0.9), lambda T: torch.optim.SGD(T, lr=0.01, momentum=0.9)),
         ("  + dampening=0.5", lambda: SGD(0.01, 0.9, 0.5), lambda T: torch.optim.SGD(T, lr=0.01, momentum=0.9, dampening=0.5)),
         ("  + nesterov", lambda: SGD(0.01, 0.9, nesterov=True), lambda T: torch.optim.SGD(T, lr=0.01, momentum=0.9, nesterov=True)),
         ("RMSprop lr=0.01", lambda: RMSprop(0.01), lambda T: torch.optim.RMSprop(T, lr=0.01)),
         ("Adam lr=0.001", lambda: Adam(0.001), lambda T: torch.optim.Adam(T, lr=0.001)),
         ("Adam lr=0.01 β=(0.8, 0.99)", lambda: Adam(0.01, (0.8, 0.99)), lambda T: torch.optim.Adam(T, lr=0.01, betas=(0.8, 0.99)))]
print(f"1) los mismos {N} gradientes en los dos optimizadores · error máximo en los parámetros (PyTorch {torch.__version__})")
for nombre, a, b in casos:
    print(f"   {nombre:28s} {mismos_gradientes(a, b):8.1e}")

# 2) el gradiente de red.py frente al autograd y a diferencias centrales
X, y = escalar(X_TR), Y_TR.astype(float)
P = red.iniciar(8, seed=0)
L, G = red.gradiente(P, X, y)
net = torch.nn.Sequential(torch.nn.Linear(2, 8), torch.nn.Tanh(), torch.nn.Linear(8, 1))
with torch.no_grad():
    for t, p in zip(net.parameters(), P):
        t.copy_(torch.tensor(p))
Lt = torch.nn.functional.binary_cross_entropy_with_logits(net(torch.tensor(X))[:, 0], torch.tensor(y)); Lt.backward()
eg = max(np.max(np.abs(g - t.grad.numpy())) for g, t in zip(G, net.parameters()))
h, ef = 1e-6, 0.0
for i, p in enumerate(P):
    for idx in np.ndindex(p.shape):
        Pm = [q.copy() for q in P]; Pp = [q.copy() for q in P]
        Pm[i][idx] -= h; Pp[i][idx] += h
        ef = max(ef, abs((red.perdida(Pp, X, y) - red.perdida(Pm, X, y)) / (2 * h) - G[i][idx]))
print(f"2) gradiente de la red (33 parámetros, 600 lecturas): pérdida {abs(L - Lt.item()):.1e},"
      f" frente al autograd {eg:.1e}, frente a diferencias centrales {ef:.1e}")

# 3) el entrenamiento completo: barajado, lotes de 32, 20 épocas, con cada optimizador
def comparar(nuestro, suyo, epocas=20, lote=32):
    P = red.iniciar(8, seed=0); o = nuestro()
    with torch.no_grad():
        for t, p in zip(net.parameters(), P):
            t.copy_(torch.tensor(p))
    ot = suyo(net.parameters()); r = np.random.default_rng(1); pasos = 0
    for _ in range(epocas):
        orden = r.permutation(len(y))
        for k in range(0, len(y), lote):
            j = orden[k:k + lote]
            _, G = red.gradiente(P, X[j], y[j]); o.step(P, G)
            ot.zero_grad()
            torch.nn.functional.binary_cross_entropy_with_logits(net(torch.tensor(X[j]))[:, 0], torch.tensor(y[j])).backward()
            ot.step(); pasos += 1
    return pasos, max(np.max(np.abs(p - t.detach().numpy())) for p, t in zip(P, net.parameters()))

print("3) entrenamiento completo, lotes de 32 barajados, 20 épocas · diferencia máxima en los pesos finales")
for nombre, a, b in [("SGD lr=0.3", lambda: SGD(0.3), lambda T: torch.optim.SGD(T, lr=0.3)),
                     ("momento 0.9, lr=0.1", lambda: SGD(0.1, 0.9), lambda T: torch.optim.SGD(T, lr=0.1, momentum=0.9)),
                     ("RMSprop lr=0.01", lambda: RMSprop(0.01), lambda T: torch.optim.RMSprop(T, lr=0.01)),
                     ("Adam lr=0.01", lambda: Adam(0.01), lambda T: torch.optim.Adam(T, lr=0.01))]:
    pasos, e = comparar(a, b)
    print(f"   {nombre:22s} {pasos} pasos   {e:8.1e}")
