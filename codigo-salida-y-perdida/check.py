"""Verificación de perdidas.py contra torch.nn.functional, el autograd de PyTorch,
sklearn.metrics.log_loss y diferencias centrales (todo en float64)."""
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import log_loss
import perdidas as P

torch.set_default_dtype(torch.float64)
rng = np.random.default_rng(4)
n, K = 5000, 3
yhat, y = rng.normal(32, 6, n), rng.normal(32, 6, n)       # regresión (L/100 km)
z, yb = rng.normal(0, 4, n), rng.integers(0, 2, n)          # logits binarios y etiquetas 0/1
Z, c = rng.normal(0, 4, (n, K)), rng.integers(0, K, n)      # logits de 3 clases y la clase verdadera

def frente_a_torch(nombre, f, df, ref, x, *args):
    t = torch.tensor(x, requires_grad=True)
    v = ref(t, *[torch.tensor(a) for a in args]); v.backward()
    e_v = abs(f(x, *args) - v.item()); e_g = np.max(np.abs(df(x, *args) - t.grad.numpy()))
    print(f"  {nombre:28s} valor {e_v:8.1e}   gradiente {e_g:8.1e}")
    return max(e_v, e_g)

print(f"{n} ejemplos · error absoluto máximo frente a PyTorch {torch.__version__}")
err = [frente_a_torch("MSE vs F.mse_loss", P.mse, P.d_mse, F.mse_loss, yhat, y),
       frente_a_torch("MAE vs F.l1_loss", P.mae, P.d_mae, F.l1_loss, yhat, y),
       frente_a_torch("Huber δ=2 vs F.huber_loss", lambda a, b: P.huber(a, b, 2.0),
                      lambda a, b: P.d_huber(a, b, 2.0), lambda a, b: F.huber_loss(a, b, delta=2.0), yhat, y),
       frente_a_torch("BCE vs F.bce_with_logits", P.bce_logits, P.d_bce_logits,
                      F.binary_cross_entropy_with_logits, z, yb.astype(float)),
       frente_a_torch("BCE pos_weight=13.3", lambda a, b: P.bce_logits(a, b, 13.3),
                      lambda a, b: P.d_bce_logits(a, b, 13.3),
                      lambda a, b: F.binary_cross_entropy_with_logits(a, b, pos_weight=torch.tensor(13.3)),
                      z, yb.astype(float))]
t = torch.tensor(Z, requires_grad=True); v = F.cross_entropy(t, torch.tensor(c)); v.backward()
e = max(abs(P.ce_logits(Z, c) - v.item()), np.max(np.abs(P.d_ce_logits(Z, c) - t.grad.numpy())))
print(f"  {'CE vs F.cross_entropy':28s} valor {abs(P.ce_logits(Z, c) - v.item()):8.1e}   gradiente "
      f"{np.max(np.abs(P.d_ce_logits(Z, c) - t.grad.numpy())):8.1e}"); err.append(e)
for nombre, mio, ref in [("softmax vs F.softmax", P.softmax(Z, axis=1), F.softmax(torch.tensor(Z), 1)),
                         ("softmax T=2.5", P.softmax(Z, 2.5, axis=1), F.softmax(torch.tensor(Z) / 2.5, 1)),
                         ("log_softmax", P.log_softmax(Z, axis=1), F.log_softmax(torch.tensor(Z), 1)),
                         ("logsumexp", P.logsumexp(Z, axis=1), torch.logsumexp(torch.tensor(Z), 1))]:
    e = np.max(np.abs(mio - ref.numpy())); err.append(e); print(f"  {nombre:28s} valor {e:8.1e}")
pb = P.sigmoide(z)
e1 = abs(P.bce_logits(z, yb) - log_loss(yb, pb)); e2 = abs(P.ce_logits(Z, c) - log_loss(c, P.softmax(Z, axis=1)))
print(f"  {'BCE vs sklearn log_loss':28s} valor {e1:8.1e}\n  {'CE vs sklearn log_loss':28s} valor {e2:8.1e}")
print(f"máximo de todo lo anterior: {max(err + [e1, e2]):.1e}")

h = 1e-6                                                    # diferencias centrales en un lote pequeño
z8, y8 = z[:8], yb[:8]
num = np.array([(P.bce_logits(z8 + h * np.eye(8)[i], y8) - P.bce_logits(z8 - h * np.eye(8)[i], y8)) / (2 * h) for i in range(8)])
Z8, c8 = Z[:8], c[:8]
numc = np.array([[(P.ce_logits(Z8 + h * np.eye(24)[3 * i + k].reshape(8, 3), c8)
                   - P.ce_logits(Z8 - h * np.eye(24)[3 * i + k].reshape(8, 3), c8)) / (2 * h) for k in range(3)] for i in range(8)])
print(f"diferencias centrales (h = 1e-6): BCE {np.max(np.abs(num - P.d_bce_logits(z8, y8))):.1e}"
      f"   CE {np.max(np.abs(numc - P.d_ce_logits(Z8, c8))):.1e}")
