"""El gradiente que explota y el recorte. Una RNN de 16 unidades cuya matriz recurrente U empieza como
ρ · (una matriz ortogonal): con ρ < 1 el eco se apaga y con ρ > 1 puede crecer paso a paso. Después, la misma
RNN con ρ = 1,5 entrenada con descenso de gradiente simple (tasa 0,05, lotes de 32, 10 épocas, hueco de 25
lecturas), sin recorte y recortando la norma del gradiente a 1 (Pascanu, Mikolov y Bengio, 2013)."""
import numpy as np
import torch
from torch import nn
from datos import datos
from modelos import Lectora
from bptt import eco

torch.set_num_threads(4)
(X, Y, _), (Xv, Yv, _), (Xt, _, _) = datos(25)
Xtr, Ytr, Xva, Yva = (torch.tensor(a, dtype=torch.float64) for a in (X, Y, Xv, Yv))


def red(rho, seed=0):
    torch.manual_seed(seed); m = Lectora("RNN", 16)
    with torch.no_grad():
        q, _ = torch.linalg.qr(torch.randn(16, 16)); m.rec.weight_hh_l0.copy_(rho * q)
    return m


print("‖∂z/∂h_t‖ a k pasos del final, relativo a k = 0, antes de entrenar (200 tramos de 41 lecturas)")
for rho in (0.5, 0.9, 1.0, 1.5, 3.0):
    gh, _ = eco(red(rho), Xt[:200])
    print(f"ρ = {rho:3.1f}: " + "  ".join(f"k={k}: {gh[-1 - k] / gh[-1]:.1e}" for k in (1, 5, 10, 20, 40)))

perd = nn.functional.binary_cross_entropy_with_logits
print("ρ = 1,5, descenso de gradiente simple · acierto en validación al final de cada época")
for recorte in (None, 1.0):
    m = red(1.5); opt = torch.optim.SGD(m.parameters(), lr=0.05); g = torch.Generator().manual_seed(1)
    normas, saltos, accs = [], [], []
    for ep in range(10):
        orden = torch.randperm(len(Ytr), generator=g)
        for k in range(0, len(Ytr), 32):
            j = orden[k:k + 32]; opt.zero_grad(); perd(m(Xtr[j]), Ytr[j]).backward()
            normas.append(nn.utils.clip_grad_norm_(m.parameters(), recorte if recorte else float("inf")).item())
            antes = [p.detach().clone() for p in m.parameters()]; opt.step()
            saltos.append(float(sum(((p.detach() - a) ** 2).sum() for p, a in zip(m.parameters(), antes)) ** 0.5))
        with torch.no_grad():
            accs.append(((m(Xva) > 0).double() == Yva).double().mean().item())
    print(f"{'recortando a 1' if recorte else 'sin recorte   '}: norma del gradiente mediana {np.median(normas):.2f}, máxima {max(normas):.1f};"
          f" mayor paso de los pesos {max(saltos):.2f}; acierto {' '.join(f'{a:.0%}' for a in accs)}")
