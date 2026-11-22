"""Verificación de atencion.py frente a PyTorch (float64): F.scaled_dot_product_attention, nn.MultiheadAttention,
nn.TransformerEncoderLayer, el autograd y diferencias centrales; y el modelo entrenado, evaluado solo con NumPy."""
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import atencion as A
from datos import datos
from modelo import Viajero, a_numpy, perdida

torch.set_default_dtype(torch.float64)
rng = np.random.default_rng(11)
n, d, h = 9, 16, 2
t = lambda x: torch.tensor(x, requires_grad=True)
err = lambda a, b: float(np.max(np.abs(np.asarray(a) - (b.detach().numpy() if torch.is_tensor(b) else b))))
print(f"PyTorch {torch.__version__} · secuencia de {n} fichas, d = {d}, {h} cabezas · error absoluto máximo")

# 1. atención de una cabeza
Q, K, Vm, dO = (rng.normal(size=(n, 8)) for _ in range(4))
for causal in (False, True):
    O, W = A.atencion(Q, K, Vm, causal)
    q, k, v = t(Q), t(K), t(Vm)
    o = F.scaled_dot_product_attention(q, k, v, is_causal=causal); o.backward(torch.tensor(dO))
    dq, dk, dv = A.atencion_atras(dO, Q, K, Vm, W)
    print(f"  scaled_dot_product_attention{' causal' if causal else '       '}  salida {err(O, o):.1e}"
          f"   dQ {err(dq, q.grad):.1e}  dK {err(dk, k.grad):.1e}  dV {err(dv, v.grad):.1e}")

# 2. multicabeza: nn.MultiheadAttention guarda Wq, Wk y Wv apilados en in_proj_weight (3d × d), cada uno traspuesto
P = {f"W{c}": rng.normal(0, 0.3, (d, d)) for c in "qkvo"} | {f"b{c}": rng.normal(0, 0.1, d) for c in "qkvo"}
X = rng.normal(size=(n, d))
mha = nn.MultiheadAttention(d, h, batch_first=True)
with torch.no_grad():
    mha.in_proj_weight.copy_(torch.tensor(np.concatenate([P["Wq"].T, P["Wk"].T, P["Wv"].T])))
    mha.in_proj_bias.copy_(torch.tensor(np.concatenate([P["bq"], P["bk"], P["bv"]])))
    mha.out_proj.weight.copy_(torch.tensor(P["Wo"].T)); mha.out_proj.bias.copy_(torch.tensor(P["bo"]))
Y, cache = A.mha(X, P, h, causal=True)
y, w = mha(*(torch.tensor(X[None]),) * 3, attn_mask=torch.tensor(A.mascara_causal(n)), average_attn_weights=False)
print(f"  nn.MultiheadAttention causal        salida {err(Y, y[0]):.1e}   pesos de las 2 cabezas {err(np.stack(cache[5]), w[0]):.1e}")

# 3. el bloque completo frente a nn.TransformerEncoderLayer (norma delante, sin dropout) y su autograd
P |= {"g1": 1 + rng.normal(0, 0.1, d), "be1": rng.normal(0, 0.1, d), "g2": 1 + rng.normal(0, 0.1, d),
      "be2": rng.normal(0, 0.1, d), "W1": rng.normal(0, 0.3, (d, 4 * d)), "b1": rng.normal(0, 0.1, 4 * d),
      "W2": rng.normal(0, 0.3, (4 * d, d)), "b2": rng.normal(0, 0.1, d)}
capa = nn.TransformerEncoderLayer(d, h, dim_feedforward=4 * d, dropout=0.0, batch_first=True, norm_first=True)
with torch.no_grad():
    capa.self_attn.in_proj_weight.copy_(mha.in_proj_weight); capa.self_attn.in_proj_bias.copy_(mha.in_proj_bias)
    capa.self_attn.out_proj.weight.copy_(mha.out_proj.weight); capa.self_attn.out_proj.bias.copy_(mha.out_proj.bias)
    for lin, W, b in ((capa.linear1, "W1", "b1"), (capa.linear2, "W2", "b2")):
        lin.weight.copy_(torch.tensor(P[W].T)); lin.bias.copy_(torch.tensor(P[b]))
    for ln, g, b in ((capa.norm1, "g1", "be1"), (capa.norm2, "g2", "be2")):
        ln.weight.copy_(torch.tensor(P[g])); ln.bias.copy_(torch.tensor(P[b]))
dY = rng.normal(size=(n, d))
Yb, cb = A.bloque(X, P, h, causal=True)
x = t(X[None])
yb = capa(x, src_mask=nn.Transformer.generate_square_subsequent_mask(n), is_causal=True); yb.backward(torch.tensor(dY[None]))
dX, g = A.bloque_atras(dY, cb, P)
ref = {"Wq": capa.self_attn.in_proj_weight.grad[:d].T, "Wk": capa.self_attn.in_proj_weight.grad[d:2 * d].T,
       "Wv": capa.self_attn.in_proj_weight.grad[2 * d:].T, "Wo": capa.self_attn.out_proj.weight.grad.T,
       "W1": capa.linear1.weight.grad.T, "W2": capa.linear2.weight.grad.T, "g1": capa.norm1.weight.grad, "g2": capa.norm2.weight.grad}
print(f"  nn.TransformerEncoderLayer causal   salida {err(Yb, yb[0]):.1e}   dX {err(dX, x.grad[0]):.1e}"
      f"   pesos {max(err(g[k], v) for k, v in ref.items()):.1e}")

# 4. diferencias centrales sobre el bloque: L = suma(Y · dY)
def L_(Pp, Xp):
    return float((A.bloque(Xp, Pp, h, True)[0] * dY).sum())
eps, peor = 1e-6, 0.0
for k in ("Wq", "Wk", "W1", "g2"):
    for idx in [tuple(rng.integers(0, s) for s in P[k].shape) for _ in range(5)]:
        Pm, Pp = {a: b.copy() for a, b in P.items()}, {a: b.copy() for a, b in P.items()}
        Pp[k][idx] += eps; Pm[k][idx] -= eps
        peor = max(peor, abs((L_(Pp, X) - L_(Pm, X)) / (2 * eps) - g[k][idx]))
print(f"  diferencias centrales (20 pesos)     gradiente {peor:.1e}")

# 5. el modelo entrenado (viajero.pt), evaluado con NumPy en los 1 000 viajes de prueba
m = Viajero(); m.load_state_dict(torch.load("viajero.pt")); Pm = a_numpy(m)
Xt, Lt = datos()[2]
tot, cnt, e_logit = 0.0, 0, 0.0
with torch.no_grad():
    Zt = m(torch.tensor(Xt)).numpy()
    lt = perdida(torch.tensor(Zt), torch.tensor(Xt), torch.tensor(Lt)).item()
for i in range(len(Xt)):
    ids = Xt[i, :Lt[i]]
    Z, _ = A.modelo(ids, Pm, 2)
    e_logit = max(e_logit, np.max(np.abs(Z - Zt[i, :Lt[i]])))
    l, _ = A.entropia_cruzada(Z[:-1], ids[1:]); tot += l * (len(ids) - 1); cnt += len(ids) - 1
print(f"  modelo entrenado, 1 000 viajes      logits {e_logit:.1e}   pérdida NumPy {tot / cnt:.6f} · PyTorch {lt:.6f}")
ids = Xt[0, :Lt[0]]
Z, cm = A.modelo(ids, Pm, 2)
_, dZ = A.entropia_cruzada(Z[:-1], ids[1:])
gm = A.modelo_atras(np.vstack([dZ, np.zeros((1, Z.shape[1]))]), cm, Pm)
m.zero_grad(); F.cross_entropy(m(torch.tensor(ids[None]))[0, :-1], torch.tensor(ids[1:])).backward()
tg = {"E": m.E.weight.grad, "Wq": m.q.weight.grad.T, "W1": m.f1.weight.grad.T, "Wout": m.sal.weight.grad.T, "gf": m.lnf.weight.grad}
print(f"  gradientes del modelo, un viaje      {max(err(gm[k], v) for k, v in tg.items()):.1e}"
      f"   (filas de E con gradiente: {int((np.abs(gm['E']).sum(1) > 0).sum())} de 20)")
