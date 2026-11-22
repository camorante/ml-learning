"""El eco del gradiente: cuánto llega a cada paso t el gradiente de la respuesta final z (el logit), ‖∂z/∂h_t‖,
para la RNN y la LSTM de 16 unidades, antes de entrenar (pesos iniciales de la semilla 0) y ya entrenadas.
En la LSTM también ‖∂z/∂c_t‖, el gradiente que viaja por la cinta. Media de 200 tramos de prueba."""
import json, os
import numpy as np
import torch
from datos import HUECOS, L0, datos
from modelos import Lectora, ESCALA

os.chdir(os.path.dirname(os.path.abspath(__file__)))


def eco(m, X):
    """Recorre la red a mano con los mismos pesos para guardar el gradiente de cada h_t (y c_t)."""
    P = {k: v.detach() for k, v in m.state_dict().items()}
    W, U, b = P["rec.weight_ih_l0"], P["rec.weight_hh_l0"], P["rec.bias_ih_l0"] + P["rec.bias_hh_l0"]
    x = torch.tensor(X)[:, :, None] / ESCALA; n, T, _ = x.shape; H = U.shape[1]
    h, c, hs, cs = torch.zeros(n, H, requires_grad=True), torch.zeros(n, H), [], []
    for t in range(T):
        z = x[:, t] @ W.T + h @ U.T + b
        if m.tipo == "RNN":
            h = torch.tanh(z)
        else:
            i, f, g, o = torch.sigmoid(z[:, :H]), torch.sigmoid(z[:, H:2 * H]), torch.tanh(z[:, 2 * H:3 * H]), torch.sigmoid(z[:, 3 * H:])
            c = f * c + i * g; c.retain_grad(); cs.append(c)
            h = o * torch.tanh(c)
        h.retain_grad(); hs.append(h)
    logit = h @ P["out.weight"][0] + P["out.bias"][0]
    logit.sum().backward()
    gh = np.array([v.grad.norm(dim=1).mean().item() for v in hs])
    gc = np.array([v.grad.norm(dim=1).mean().item() for v in cs]) if cs else None
    return gh, gc


if __name__ == "__main__":
    R = {}
    print("‖∂z/∂h_t‖ a k pasos del final, relativo a k = 0 (media de 200 tramos de prueba)")
    for G in HUECOS:
        _, _, (Xt, _, _) = datos(G)
        for tipo in ("RNN", "LSTM"):
            torch.manual_seed(0); ini = Lectora(tipo, 16)                       # los mismos pesos con que empezó a entrenar
            fin = Lectora(tipo, 16); fin.load_state_dict(torch.load(f"red_{tipo}16_G{G}.pt"))
            for cuando, m in (("inicio", ini), ("entrenada", fin)):
                gh, gc = eco(m, Xt[:200])
                R[f"{tipo}|{cuando}|{G}"] = [float(f"{v:.4g}") for v in gh / gh[-1]]
                if gc is not None:
                    R[f"{tipo}|{cuando}|{G}|c"] = [float(f"{v:.4g}") for v in gc / gh[-1]]
                if G in (50, 100):
                    T = L0 + G; ks = [k for k in (0, 10, 25, 50, 75, 100, T - 1) if k < T]
                    print(f"hueco {G:3d} {tipo:4s} {cuando:9s} " + "  ".join(f"k={k}: {gh[T - 1 - k] / gh[-1]:.1e}" for k in ks))
    json.dump(R, open("bptt.json", "w", encoding="utf-8"), separators=(",", ":"))
