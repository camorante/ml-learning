"""Las direcciones también significan algo. Una flota reparte sus viajes en 8 tipos de tramo: 4 zonas (centro,
polígono, carretera, autopista) × 2 turnos (día, noche). La red solo ve el número del tramo (0…7) y aprende un
embedding de 2 números prediciendo dos cosas de cada viaje: el consumo (L/100 km) y los minutos de ralentí por
hora. De noche hay menos tráfico (gasta menos), pero se espera más en los muelles con el motor encendido."""
import json, os
import numpy as np
import torch
import torch.nn as nn

torch.set_default_dtype(torch.float64)
os.chdir(os.path.dirname(os.path.abspath(__file__)))
ZONAS = ["centro", "polígono", "carretera", "autopista"]
TURNOS = ["día", "noche"]
CONS = {"centro": 38.0, "polígono": 34.0, "carretera": 30.0, "autopista": 28.0}    # L/100 km de día
RAL = {"centro": 16.0, "polígono": 11.0, "carretera": 6.0, "autopista": 3.0}       # min de ralentí por hora de día
NOCHE = (-3.0, 4.0)                                                                # lo que cambia de noche
TRAMOS = [(z, t) for t in TURNOS for z in ZONAS]                                  # 0…3 de día, 4…7 de noche

rng = np.random.default_rng(3)
X, Y = [], []
for c, (z, t) in enumerate(TRAMOS):
    for _ in range(60):                                                           # 60 viajes de cada tramo
        n = t == "noche"
        X.append(c); Y.append([CONS[z] + n * NOCHE[0] + rng.normal(0, 2.0), RAL[z] + n * NOCHE[1] + rng.normal(0, 2.0)])
X, Y = torch.tensor(X), torch.tensor(Y)
mu, sd = Y.mean(0), Y.std(0)
Ys = (Y - mu) / sd                                                                # escalado (tema D7)

torch.manual_seed(0)
m = nn.Sequential(nn.Embedding(8, 2), nn.Linear(2, 2))
opt = torch.optim.Adam(m.parameters(), lr=0.05)
for paso in range(600):                                                           # lote completo, 600 pasos
    loss = ((m(X) - Ys) ** 2).mean()
    opt.zero_grad(); loss.backward(); opt.step()
E = m[0].weight.detach().numpy()
print(f"{len(X)} viajes, 8 tramos · embedding 8×2 + capa lineal 2→2 · error cuadrático medio final {loss.item():.4f} (escalado)")
pred = (m(torch.arange(8)) * sd + mu).detach().numpy()
for c, (z, t) in enumerate(TRAMOS):
    print(f"  {z:9s} {t:5s}  embedding ({E[c, 0]:6.3f}, {E[c, 1]:6.3f})   predice {pred[c, 0]:5.2f} L/100 km, {pred[c, 1]:5.2f} min/h")
print("la flecha «de día a de noche» en cada zona:")
fl = []
for k, z in enumerate(ZONAS):
    v = E[k + 4] - E[k]; fl.append(v)
    print(f"  {z:9s} ({v[0]:6.3f}, {v[1]:6.3f})  largo {np.hypot(*v):.3f}  ángulo {np.degrees(np.arctan2(v[1], v[0])):6.1f}°")
q = E[3] + (E[4] - E[0])
dist = np.linalg.norm(E - q, axis=1)
print(f"autopista de día + (centro de noche − centro de día) = ({q[0]:.3f}, {q[1]:.3f});"
      f" el tramo más cercano: {' '.join(TRAMOS[int(dist.argmin())])}, a {dist.min():.3f}"
      f" (el siguiente, {' '.join(TRAMOS[int(np.argsort(dist)[1])])}, a {np.sort(dist)[1]:.3f})")
json.dump({"zonas": ZONAS, "E": np.round(E, 4).tolist(), "pred": np.round(pred, 2).tolist()},
          open("tramos.json", "w", encoding="utf-8"))
