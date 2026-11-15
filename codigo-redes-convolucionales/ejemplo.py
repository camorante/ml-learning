"""Ejemplo resuelto: el filtro del juguete sobre 10 lecturas (1 s) de la ventana «frenada brusca», cada salida a
mano; después un max-pooling de 2 y la decisión con el umbral del juguete."""
import json
import numpy as np


def suma(w, x, i):
    """Una casilla del mapa, acumulada en el mismo orden que el juguete (JS) y toys.py."""
    a = 0.0
    for k in range(5):
        a += w[k] * x[i + k]
    return a


E = json.load(open("../embed.json", encoding="utf-8"))
w, umbral = E["presets"]["aprendido"], E["umbral"]["aprendido"]
x = E["trazas"]["brusca"]
ini = 33                                                  # lectura 33 = 3,3 s desde el principio de la ventana
tramo = x[ini:ini + 10]
print(f"filtro w = {w}   (suma {sum(w):.2f})   umbral del juguete {umbral:.4f}")
print(f"lecturas {ini}–{ini + 9} (m/s²): {tramo}")
salidas = []
for i in range(6):                                        # 10 lecturas, filtro de 5, paso 1, sin relleno → 6 salidas
    prod = [w[k] * tramo[i + k] for k in range(5)]
    s = suma(w, tramo, i); salidas.append(s)
    print(f"salida {ini + i} ({(ini + i) / 10:.1f} s): " + " ".join(f"{p:+.4f}" for p in prod) + f" = {s:.4f}")
pool = [max(salidas[j], salidas[j + 1]) for j in (0, 2, 4)]
print("max-pooling de 2:", [f"{v:.4f}" for v in pool], "→ el mayor:", f"{max(pool):.4f}",
      "→", "ALERTA" if max(pool) > umbral else "sin alerta")
mapa = [suma(w, x, i) for i in range(len(x) - 4)]
print(f"en toda la ventana: {len(mapa)} salidas, la mayor {max(mapa):.4f} en la posición {int(np.argmax(mapa))};"
      f" {sum(v > umbral for v in mapa)} salidas pasan del umbral")
print(f"la regla de la flota en este medio segundo: media de las lecturas {ini + 1}–{ini + 5} = {np.mean(x[ini + 1:ini + 6]):.3f} m/s²")
print(f"umbral / |suma de los pesos| = {umbral:.4f} / {abs(sum(w)):.2f} = {umbral / abs(sum(w)):.3f} m/s² de deceleración media;"
      f"  mayor suma / |suma| = {max(mapa) / abs(sum(w)):.3f} m/s²")
