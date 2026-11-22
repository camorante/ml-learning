"""El ejemplo resuelto: un paso de entrenamiento a mano con 3 códigos y embeddings de 2 números, y el coseno
antes y después. Comprueba los gradientes con el autograd de PyTorch."""
import numpy as np
import torch
import torch.nn.functional as Fn

torch.set_default_dtype(torch.float64)
COD = ["P0217", "P0128", "C0035"]
E = np.array([[0.6, 0.2], [0.1, 0.5], [0.4, -0.3]])     # una fila por código
U = np.array([[0.5, 0.5], [0.3, 0.8], [0.7, -0.2]])     # vectores de salida
b = np.zeros(3)                                          # sesgos de salida
a, y, lr = 0, 1, 0.5                                     # visto P0217; el otro código de la visita es P0128

cos = lambda u, v: (u @ v) / (np.linalg.norm(u) * np.linalg.norm(v))
print("antes:  cos(P0217, P0128) = %.4f   cos(P0217, C0035) = %.4f" % (cos(E[0], E[1]), cos(E[0], E[2])))

h = E[a]
z = U @ h + b; z[a] = -np.inf                            # P0217 no puede acompañarse a sí mismo
p = np.exp(z - z[1:].max()); p /= p.sum()
L = -np.log(p[y])
print(f"logits: z(P0128) = {z[1]:.4f}  z(C0035) = {z[2]:.4f}")
print(f"probabilidades: p(P0128) = {p[1]:.5f}  p(C0035) = {p[2]:.5f}   pérdida −ln p(P0128) = {L:.5f}")
dz = p.copy(); dz[y] -= 1
dh = dz @ U                                              # ∂L/∂E[P0217]
dU = np.outer(dz, h); db = dz
print(f"∂L/∂z = ({dz[1]:.5f}, {dz[2]:.5f})   ∂L/∂E[P0217] = ({dh[0]:.5f}, {dh[1]:.5f})")
print(f"∂L/∂u(P0128) = ({dU[1, 0]:.5f}, {dU[1, 1]:.5f})   ∂L/∂u(C0035) = ({dU[2, 0]:.5f}, {dU[2, 1]:.5f})"
      f"   ∂L/∂b = ({db[1]:.5f}, {db[2]:.5f})")

# la misma cuenta con autograd
Et, Ut, bt = (torch.tensor(v, requires_grad=True) for v in (E, U, b))
zt = (Ut @ Et[a] + bt).masked_fill(torch.tensor([True, False, False]), float("-inf"))
Lt = Fn.cross_entropy(zt[None], torch.tensor([y])); Lt.backward()
print(f"PyTorch: pérdida {Lt.item():.5f} · error máximo de los gradientes "
      f"{max(np.abs(Et.grad.numpy()[0] - dh).max(), np.abs(Ut.grad.numpy() - dU).max(), np.abs(bt.grad.numpy() - db).max()):.1e}"
      f" · filas de E con gradiente: {int((Et.grad.abs().sum(1) > 0).sum())} de 3")

E2, U2, b2 = E.copy(), U - lr * dU, b - lr * db
E2[a] -= lr * dh
print(f"después del paso (lr = {lr}): E[P0217] = ({E2[0, 0]:.4f}, {E2[0, 1]:.4f})   u(P0128) = ({U2[1, 0]:.4f}, {U2[1, 1]:.4f})"
      f"   u(C0035) = ({U2[2, 0]:.4f}, {U2[2, 1]:.4f})   b = ({b2[1]:.4f}, {b2[2]:.4f})")
z2 = U2 @ E2[a] + b2
p2 = 1 / (1 + np.exp(-(z2[1] - z2[2])))
print(f"nuevos logits ({z2[1]:.4f}, {z2[2]:.4f}) → p(P0128) = {p2:.4f}, pérdida {-np.log(p2):.4f}")
print("después: cos(P0217, P0128) = %.4f   cos(P0217, C0035) = %.4f" % (cos(E2[0], E2[1]), cos(E2[0], E2[2])))
print(f"normas: |E[P0217]| {np.linalg.norm(E[0]):.5f} → {np.linalg.norm(E2[0]):.5f} · |E[P0128]| {np.linalg.norm(E[1]):.5f}"
      f" · |E[C0035]| {np.linalg.norm(E[2]):.5f}")
