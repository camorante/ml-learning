"""Verificación contra PyTorch (CPU, float64) de todo lo que el tema implementa desde cero: el gradiente con dropout,
Adam con L2 y AdamW, nn.Dropout, nn.BatchNorm1d (entrenamiento, evaluación y hacia atrás) y las inicializaciones."""
import numpy as np
import torch
import mlp
from bn import BatchNorm1d
from datos import V_TR, Y_TR, ex, ey

torch.set_default_dtype(torch.float64)
T = lambda a: torch.tensor(a)
X, y = ex(V_TR), ey(Y_TR)


def mascaras(seed, n, p, capas=(64, 64)):
    """Las mismas máscaras que mlp.adelante saca de default_rng(seed), en el mismo orden (capa 1, capa 2)."""
    r = np.random.default_rng(seed)
    return [(r.random((n, m)) >= p) / (1 - p) for m in capas]


def red_torch(Pt, X, M):
    h = X
    for i in range(0, len(Pt) - 2, 2):
        h = torch.tanh(h @ Pt[i].T + Pt[i + 1])
        if M is not None:
            h = h * T(M[i // 2])
    return (h @ Pt[-2].T + Pt[-1])[:, 0]


print(f"PyTorch {torch.__version__} · errores absolutos máximos")
# 1) el gradiente de mlp.py con dropout (las mismas máscaras) frente al autograd
P = mlp.iniciar([1, 64, 64, 1], 0)
L, G = mlp.gradiente(P, X, y, 0.3, np.random.default_rng(9))
Pt = [T(p).requires_grad_() for p in P]
Lt = torch.mean((red_torch(Pt, T(X), mascaras(9, 20, 0.3)) - T(y)) ** 2); Lt.backward()
eg = max(float(np.max(np.abs(g - t.grad.numpy()))) for g, t in zip(G, Pt))
print(f"1) gradiente con dropout p = 0.3 (4 353 parámetros): pérdida {abs(L - Lt.item()):.1e}, gradiente {eg:.1e}")
h, ef, L0, G0 = 1e-6, 0.0, *mlp.gradiente(P, X, y)
for i, idxs in {0: [(0, 0), (5, 0), (9, 0)], 2: [(0, 0), (5, 3), (40, 61)], 4: [(0, 0), (0, 7), (0, 50)]}.items():
    for idx in idxs:
        Pp = [q.copy() for q in P]; Pm = [q.copy() for q in P]; Pp[i][idx] += h; Pm[i][idx] -= h
        ef = max(ef, abs((mlp.gradiente(Pp, X, y)[0] - mlp.gradiente(Pm, X, y)[0]) / (2 * h) - G0[i][idx]))
print(f"   sin dropout, 9 pesos frente a diferencias centrales: {ef:.1e}")

# 2) Adam con L2 y AdamW frente a torch.optim con los mismos 500 gradientes
rng = np.random.default_rng(5)
Q0 = [rng.normal(0, 1, s) for s in [(8, 2), (8,), (1, 8), (1,)]]
GS = [[rng.normal(0, 1, q.shape) * 10 ** rng.uniform(-3, 1) for q in Q0] for _ in range(500)]


def mismos(nuestro, suyo):
    Q, Qt = [q.copy() for q in Q0], [T(q).requires_grad_() for q in Q0]
    o, ot, err = nuestro, suyo(Qt), 0.0
    for Gk in GS:
        o.step(Q, Gk)
        for t, g in zip(Qt, Gk):
            t.grad = T(g)
        ot.step()
        err = max(err, max(float(np.max(np.abs(q - t.detach().numpy()))) for q, t in zip(Q, Qt)))
    return err, Q


e1, Qa = mismos(mlp.Adam(0.01, 0.1, desacoplado=False), lambda q: torch.optim.Adam(q, lr=0.01, weight_decay=0.1))
e2, Qw = mismos(mlp.Adam(0.01, 0.1, desacoplado=True), lambda q: torch.optim.AdamW(q, lr=0.01, weight_decay=0.1))
dif = max(float(np.max(np.abs(a - b))) for a, b in zip(Qa, Qw))
print(f"2) 500 pasos, λ = 0.1: Adam(weight_decay) {e1:.1e} · AdamW {e2:.1e} · entre sí difieren {dif:.2f}")

# 3) dropout: nn.Dropout en modo entrenamiento y evaluación
torch.manual_seed(0)
x = torch.rand(1000, 1000) + 0.5
d = torch.nn.Dropout(0.3)
xr = x.clone().requires_grad_()
out = d(xr); out.sum().backward()
M = (out != 0).double()                                             # la máscara que sorteó PyTorch
nuestro = x.numpy() * M.numpy() / (1 - 0.3)                          # x · m / (1 − p)
z = (1 - M.mean().item() - 0.3) / np.sqrt(0.3 * 0.7 / 1e6)
print(f"3) nn.Dropout(0.3) sobre 10⁶ valores: apagados {1 - M.mean().item():.4f} (se esperan 0.3; z = {z:+.2f}),"
      f" x·m/(1 − p) {np.max(np.abs(nuestro - out.detach().numpy())):.1e}, gradiente m/(1 − p) {np.max(np.abs(xr.grad.numpy() - M.numpy() / 0.7)):.1e}")
print(f"   media de la salida / media de la entrada: {out.mean().item() / x.mean().item():.4f}; en modo eval, "
      f"{torch.max(torch.abs(d.eval()(x) - x)).item():.1e} (no toca nada)")

# 4) normalización por lotes: 5 pasos de entrenamiento, luego evaluación, y hacia atrás
r = np.random.default_rng(2)
nb, tb = BatchNorm1d(6), torch.nn.BatchNorm1d(6)
with torch.no_grad():
    g0, b0 = r.normal(1, 0.2, 6), r.normal(0, 0.2, 6)
    tb.weight.copy_(T(g0)); tb.bias.copy_(T(b0))
nb.gamma, nb.beta = g0.copy(), b0.copy()
ef = es = 0.0
for k in range(5):
    xb = r.normal(3, 2, (8, 6))
    ef = max(ef, float(np.max(np.abs(nb.forward(xb) - tb(T(xb)).detach().numpy()))))
    es = max(es, float(np.max(np.abs(nb.run_mean - tb.running_mean.numpy()))), float(np.max(np.abs(nb.run_var - tb.running_var.numpy()))))
xb = r.normal(3, 2, (8, 6)); dy = r.normal(0, 1, (8, 6))
xt = T(xb).requires_grad_(); yt = tb(xt); (yt * T(dy)).sum().backward()
yb = nb.forward(xb); dx = nb.backward(dy)
eb = max(float(np.max(np.abs(dx - xt.grad.numpy()))), float(np.max(np.abs(nb.dgamma - tb.weight.grad.numpy()))),
         float(np.max(np.abs(nb.dbeta - tb.bias.grad.numpy()))))
nb.train = False; tb.eval()
xe = r.normal(3, 2, (8, 6))
ee = float(np.max(np.abs(nb.forward(xe) - tb(T(xe)).detach().numpy())))
print(f"4) BatchNorm1d, 6 neuronas, lotes de 8: salida en entrenamiento {ef:.1e}, running_mean/var {es:.1e},"
      f" hacia atrás (x, γ, β) {eb:.1e}, salida en evaluación {ee:.1e}")
mu0 = np.zeros(6)
for k in range(6):
    mu0 = 0.9 * mu0 + 0.1 * 3.0
print(f"   tras 6 lotes de media 3, running_mean solo llega a {mu0[0]:.3f} = 3·(1 − 0.9⁶): arranca en 0")

# 5) inicializaciones: nn.init frente a la teoría y a las nuestras (matrices 1 024 × 1 024)
torch.manual_seed(0)
W = torch.empty(1024, 1024)
torch.nn.init.kaiming_normal_(W, nonlinearity="relu"); sk = W.std().item()
torch.nn.init.xavier_uniform_(W); sx, mx = W.std().item(), W.abs().max().item()
torch.nn.init.xavier_uniform_(W, gain=torch.nn.init.calculate_gain("tanh")); sxt = W.std().item()
rn = np.random.default_rng(0)
print(f"5) kaiming_normal_ (relu): desv. {sk:.5f}, teoría √(2/1024) = {np.sqrt(2 / 1024):.5f}, la nuestra {mlp.he_normal(1024, 1024, rn).std():.5f}")
print(f"   xavier_uniform_: desv. {sx:.5f}, teoría √(2/2048) = {np.sqrt(2 / 2048):.5f}, la nuestra {mlp.xavier_uniform(1024, 1024, rn).std():.5f};"
      f" |w| máx {mx:.5f} ≤ √(6/2048) = {np.sqrt(6 / 2048):.5f}")
print(f"   calculate_gain: tanh {torch.nn.init.calculate_gain('tanh'):.4f} (= 5/3), relu {torch.nn.init.calculate_gain('relu'):.4f} (= √2);"
      f" xavier_uniform_ con ganancia 5/3: desv. {sxt:.5f}")

# 6) el entrenamiento completo: AdamW λ = 1 y dropout 0.2, 300 épocas, frente a torch.nn + torch.optim.AdamW
P = mlp.iniciar([1, 64, 64, 1], 0); o = mlp.Adam(0.01, 1.0); rg = np.random.default_rng(1)
Pt = [T(p).requires_grad_() for p in P]; ot = torch.optim.AdamW(Pt, lr=0.01, weight_decay=1.0)
for ep in range(300):
    s = int(rg.integers(2 ** 31))                                   # una semilla por época para las máscaras
    _, G = mlp.gradiente(P, X, y, 0.2, np.random.default_rng(s)); o.step(P, G)
    ot.zero_grad(); torch.mean((red_torch(Pt, T(X), mascaras(s, 20, 0.2)) - T(y)) ** 2).backward(); ot.step()
print(f"6) 300 épocas de AdamW (λ = 1) con dropout 0.2, mismas máscaras: pesos finales {max(float(np.max(np.abs(p - t.detach().numpy()))) for p, t in zip(P, Pt)):.1e}")
