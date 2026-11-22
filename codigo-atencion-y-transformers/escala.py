"""Por qué se divide entre √d: con q y k de componentes independientes de media 0 y varianza 1, q·k es una suma
de d términos de varianza 1, así que su varianza es d. Sin dividir, el softmax se satura y su gradiente muere."""
import json
import numpy as np
from atencion import softmax

rng = np.random.default_rng(2017)
n, claves = 20000, 10                          # 20 000 preguntas, cada una frente a 10 claves
print(f"{n} preguntas q frente a {claves} claves k, componentes N(0, 1)")
print("   d   desv. típica de q·k   peso máximo medio    entropía (máx. ln 10 = 2.303)   |gradiente| medio")
print("                 sin dividir   sin dividir / ÷√d   sin dividir / ÷√d               sin dividir / ÷√d")
tabla = {}
for d in (2, 8, 16, 64, 128, 512):
    q = rng.normal(size=(n, 1, d)); k = rng.normal(size=(n, claves, d))
    S = (q * k).sum(-1)                        # (n, 10): las puntuaciones de cada pregunta
    fila = []
    for Z in (S, S / np.sqrt(d)):
        P = softmax(Z)
        H = -(P * np.log(np.maximum(P, 1e-300))).sum(-1)
        J = np.abs(P * (1 - P)).sum(-1)        # tamaño de la diagonal del jacobiano del softmax
        fila.append((P.max(-1).mean(), H.mean(), J.mean()))
    tabla[d] = [round(float(S.std()), 3)] + [round(float(v), 4) for f in fila for v in f]
    print(f"{d:4d}   {S.std():8.2f}  (√d = {np.sqrt(d):5.2f})    {fila[0][0]:.3f} / {fila[1][0]:.3f}"
          f"        {fila[0][1]:.3f} / {fila[1][1]:.3f}                     {fila[0][2]:.3f} / {fila[1][2]:.3f}")
json.dump(tabla, open("escala.json", "w", encoding="utf-8"))
