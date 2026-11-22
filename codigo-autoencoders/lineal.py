"""Un autoencoder lineal (64 → 4 → 64, sin activaciones) entrenado con error cuadrático aprende el mismo subespacio
que el PCA con 4 componentes (Baldi y Hornik, 1989), aunque no los mismos pesos."""
import numpy as np
import torch
from scipy.linalg import subspace_angles
from sklearn.decomposition import PCA
from datos import X_TR, X_VA
from modelos import entrenar, reconstruir

k = 4
pca = PCA(k).fit(X_TR)
V = pca.components_.T                            # (64, 4): las 4 direcciones principales, ortonormales


def comparar(net, cuando):
    Wd = net.dec.weight.detach().numpy()         # (64, 4): sus columnas generan todo lo que el decodificador dibuja
    Q, _ = np.linalg.qr(Wd)
    ang = np.degrees(subspace_angles(Wd, V))
    print(f"{cuando}: ángulos principales con el PCA {', '.join(f'{a:.2e}°' for a in ang)}"
          f" · ‖P_AE − P_PCA‖ = {np.linalg.norm(Q @ Q.T - V @ V.T, 2):.1e}")
    return Wd


net, _ = entrenar(X_TR, k=k, m=0, lineal=True, epocas=300, lr=1e-3, seed=0)
comparar(net, "Adam, lotes de 64, 300 épocas")
X = torch.tensor(X_TR)                           # remate: L-BFGS con todo el conjunto hasta converger
opt = torch.optim.LBFGS(net.parameters(), max_iter=500, tolerance_grad=1e-12, tolerance_change=1e-15,
                        history_size=50, line_search_fn="strong_wolfe")
def cierre():
    opt.zero_grad(); perd = ((net(X) - X) ** 2).mean(); perd.backward(); return perd
opt.step(cierre)
Wd = comparar(net, "más L-BFGS con el lote entero ")
e_ae = ((reconstruir(net, X_VA) - X_VA) ** 2).mean()
e_pca = ((pca.inverse_transform(pca.transform(X_VA)) - X_VA) ** 2).mean()
print(f"error en validación: autoencoder lineal {e_ae:.6f} · PCA {e_pca:.6f}")
G = Wd.T @ Wd
print("pero los pesos no son los del PCA (que darían la identidad): Wdᵀ·Wd =")
print(np.array2string(G, precision=2, suppress_small=True))
