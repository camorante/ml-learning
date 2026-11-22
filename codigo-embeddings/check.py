"""embedding.py frente a torch.nn.Embedding, torch.nn.EmbeddingBag y el autograd de PyTorch (float64), y frente a
diferencias centrales. Errores absolutos máximos."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from embedding import Embedding, EmbeddingBolsa, Modelo
from modelos import Emb, evaluar
from datos import V, P_TR, P_TE

torch.set_default_dtype(torch.float64)
rng = np.random.default_rng(10)
err = lambda a, t: float(np.max(np.abs(a - (t.detach().numpy() if torch.is_tensor(t) else t))))
print(f"PyTorch {torch.__version__}, float64 · error absoluto máximo")

# 1. mirar una fila es multiplicar el one-hot por la tabla
E = rng.normal(size=(V, 3)); idx = np.array([4, 0, 4, 19, 7])
print(f"one-hot(idx) @ E frente a E[idx]: {err(np.eye(V)[idx] @ E, E[idx]):.1e}")

# 2. la capa, con índices repetidos
emb = Embedding(E.copy()); h = emb.adelante(idx); dh = rng.normal(size=h.shape); dE = emb.atras(dh)
te = nn.Embedding(V, 3); te.weight.data = torch.tensor(E); ht = te(torch.tensor(idx)); (ht * torch.tensor(dh)).sum().backward()
print(f"nn.Embedding: salida {err(h, ht):.1e} · ∂E {err(dE, te.weight.grad):.1e} ·"
      f" filas con gradiente: {int((np.abs(dE).sum(1) > 0).sum())} de {V} (índices {sorted(set(idx.tolist()))})")

# 3. la bolsa: la media de los embeddings de los códigos de cada visita
bolsas = [[1, 2], [5, 6, 7, 9], [12], [15, 17, 18]]
eb = EmbeddingBolsa(E.copy()); hb = eb.adelante(bolsas); dhb = rng.normal(size=hb.shape); dEb = eb.atras(dhb)
tb = nn.EmbeddingBag(V, 3, mode="mean"); tb.weight.data = torch.tensor(E)
plano, ini = torch.tensor(sum(bolsas, [])), torch.tensor(np.cumsum([0] + [len(b) for b in bolsas[:-1]]))
hbt = tb(plano, ini); (hbt * torch.tensor(dhb)).sum().backward()
print(f"nn.EmbeddingBag(mode='mean'): salida {err(hb, hbt):.1e} · ∂E {err(dEb, tb.weight.grad):.1e}")

# 4. el modelo entero, con los mismos pesos que un modelo de PyTorch, sobre un lote de 64 pares
torch.manual_seed(3); mt = Emb(V, 2)
Ew, Ww, bw = (p.detach().numpy().copy() for p in (mt.E.weight, mt.out.weight, mt.out.bias))
a, y = P_TR[:64, 0], P_TR[:64, 1]
m = Modelo(Ew, Ww, bw); L = m.perdida(a, y); dE, dW, db = m.atras()
Lt = Fn.cross_entropy(mt(torch.tensor(a)), torch.tensor(y)); Lt.backward()
print(f"modelo (lote de 64): pérdida {err(L, Lt):.1e} · ∂E {err(dE, mt.E.weight.grad):.1e} · ∂W {err(dW, mt.out.weight.grad):.1e}"
      f" · ∂b {err(db, mt.out.bias.grad):.1e}")
print(f"   filas de E con gradiente en ese lote: {int((np.abs(dE).sum(1) > 0).sum())} de {V}"
      f" (códigos vistos distintos: {len(set(a.tolist()))})")
ts = nn.Embedding(V, 2, sparse=True); ts.weight.data = torch.tensor(Ew)
Fn.cross_entropy(mt.out(ts(torch.tensor(a))).masked_fill(Fn.one_hot(torch.tensor(a), V).bool(), float("-inf")),
                 torch.tensor(y)).backward()
g = ts.weight.grad.coalesce()
print(f"   nn.Embedding(sparse=True): gradiente disperso con {g.indices().shape[1]} filas; frente al nuestro {err(dE[g.indices()[0].numpy()], g.values()):.1e}")

# 5. diferencias centrales sobre todas las casillas de E
hh, num = 1e-6, np.zeros_like(Ew)
for i in np.ndindex(Ew.shape):
    Ep, Em = Ew.copy(), Ew.copy(); Ep[i] += hh; Em[i] -= hh
    num[i] = (Modelo(Ep, Ww, bw).perdida(a, y) - Modelo(Em, Ww, bw).perdida(a, y)) / (2 * hh)
print(f"∂E frente a diferencias centrales (h = 1e-6): {err(dE, num):.1e}")

# 6. 30 pasos de descenso de gradiente, NumPy frente a torch.optim.SGD
m = Modelo(Ew, Ww, bw); torch.manual_seed(3); mt = Emb(V, 2); opt = torch.optim.SGD(mt.parameters(), lr=0.5)
for k in range(30):
    sl = slice(64 * k, 64 * k + 64); a, y = P_TR[sl, 0], P_TR[sl, 1]
    m.paso(a, y, 0.5)
    opt.zero_grad(); Fn.cross_entropy(mt(torch.tensor(a)), torch.tensor(y)).backward(); opt.step()
print(f"30 pasos de SGD (lr 0,5): E {err(m.emb.E, mt.E.weight):.1e} · W {err(m.W, mt.out.weight):.1e} · b {err(m.b, mt.out.bias):.1e}")

# 7. la red del juguete (mapa.py), evaluada solo con embedding.py
sd = torch.load("mapa.pt"); mj = Emb(V, 2); mj.load_state_dict(sd)
mn = Modelo(*(sd[k].numpy() for k in ("E.weight", "out.weight", "out.bias")))
print(f"red del juguete, {len(P_TE)} pares de prueba: pérdida NumPy {mn.perdida(P_TE[:, 0], P_TE[:, 1]):.6f}"
      f" · PyTorch {evaluar(mj, P_TE)[0]:.6f}")
