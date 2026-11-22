"""Recta o curva: 400 viajes con su velocidad media (km/h) y su consumo (L/100 km), que siguen una U (mucho
consumo en ciudad, el mínimo hacia 70 km/h, algo más en autopista). Un código de 1 número: PCA, autoencoder
lineal 2 → 1 → 2 y autoencoder con tanh 2 → 16 → 1 → 16 → 2, todos con los datos tipificados."""
import json
import numpy as np
import torch
from sklearn.decomposition import PCA
from modelos import entrenar, reconstruir, codigo

rng = np.random.default_rng(1212)
v = rng.uniform(20, 100, 400)
c = 30 + 0.006 * (v - 70) ** 2 + rng.normal(0, 0.8, 400)
D = np.c_[v, c]
mu, sd = D.mean(0), D.std(0)
Z = (D - mu) / sd                                         # tipificar: media 0 y desviación 1 en cada columna
raro = (np.array([[60.0, 35.0]]) - mu) / sd               # un viaje a 60 km/h que gasta 35 L/100 km: un freno que roza

pca = PCA(1).fit(Z)
lin, _ = entrenar(Z, k=1, m=0, lineal=True, epocas=400, lr=1e-2, lote=32, seed=0)
cur, _ = entrenar(Z, k=1, m=16, epocas=400, lr=1e-2, lote=32, seed=0)
modelos = {"pca": lambda X: pca.inverse_transform(pca.transform(X)),
           "lineal": lambda X: reconstruir(lin, X), "curva": lambda X: reconstruir(cur, X)}
out = {"pts": D.round(3).tolist(), "mu": mu.tolist(), "sd": sd.tolist()}
for nm, f in modelos.items():
    e = ((f(Z) - Z) ** 2).mean(1)
    er = float(((f(raro) - raro) ** 2).mean())
    u = float(np.quantile(e, 0.99))
    print(f"{nm:6s}: error medio {e.mean():.4f} · percentil 99 {u:.4f} · viaje raro {er:.4f}"
          f" ({'avisa' if er > u else 'no avisa'})")
    out[nm] = {"rec": (f(Z) * sd + mu).round(3).tolist(), "raro": (f(raro) * sd + mu).round(3)[0].tolist(),
               "e": float(e.mean()), "u": u, "er": er}
d = pca.components_[0]
dl = lin.dec.weight.detach().numpy()[:, 0]
print(f"dirección de la recta: PCA ({d[0]:.4f}, {d[1]:.4f}) · autoencoder lineal"
      f" ({dl[0] / np.linalg.norm(dl):.4f}, {dl[1] / np.linalg.norm(dl):.4f})")
zc = codigo(cur, Z)[:, 0]
zs = np.linspace(zc.min(), zc.max(), 120)[:, None]
with torch.no_grad():                                     # la curva que puede dibujar el decodificador
    out["curva"]["linea"] = (cur.dec(torch.tensor(zs)).numpy() * sd + mu).round(3).tolist()
json.dump(out, open("curva.json", "w", encoding="utf-8"))
