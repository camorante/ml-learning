"""Dos años de semanas-camión 'en bruto' para el proyecto completo: duplicados, centinelas, huecos,
averías raras y un cambio de sensor en la semana 80."""
import numpy as np, pandas as pd

def flota(n_cam=300, semanas=104, cambio_sensor=80, seed=8):
    rng = np.random.default_rng(seed)
    edad = rng.uniform(1, 12, n_cam); fragil = rng.normal(0, 0.6, n_cam)       # camiones más propensos a averías
    vib_base = rng.normal(3.0, 0.5, n_cam); nuevo_sensor = rng.random(n_cam) < 0.4
    horas = rng.uniform(0, 300, n_cam); filas = []
    for s in range(semanas):
        invierno = np.cos(2 * np.pi * (s - 2) / 52)                         # pico en enero
        km = rng.gamma(4, 450, n_cam); horas = horas + km / 55
        vib = vib_base + 0.0035 * horas + rng.normal(0, 0.35, n_cam)
        temp = 88 + 4 * invierno * -1 + rng.normal(0, 3, n_cam) + 0.004 * horas
        cod = rng.poisson(0.15 + 0.002 * horas)
        z = (-7.4 + 1.5 * (vib - 3) + 0.004 * horas + 0.45 * cod + 0.07 * edad + 0.05 * (temp - 88) + fragil
             + 0.35 * invierno)
        p = 1 / (1 + np.exp(-z)); averia = rng.random(n_cam) < p
        vib_med = vib * np.where(nuevo_sensor & (s >= cambio_sensor), 0.7, 1.0)              # el sensor nuevo lee un 30 % menos
        vib_med = np.where(rng.random(n_cam) < 0.03 + 0.5 * p, np.nan, vib_med)           # el sensor falla más antes de una avería
        t_med = np.where(rng.random(n_cam) < 0.015, rng.choice([-40.0, 255.0], n_cam), temp)  # valores centinela
        for c in range(n_cam):
            filas.append((s, c, edad[c], km[c], horas[c], vib_med[c], t_med[c], cod[c], averia[c]))
        horas = np.where(averia | (horas > rng.uniform(500, 900, n_cam)), 0, horas)      # reparación o mantenimiento
    d = pd.DataFrame(filas, columns=["semana", "camion", "edad", "km", "horas_mant", "vibracion", "temp_motor", "codigos", "averia_sig"])
    d = d.round({"edad": 1, "km": 0, "horas_mant": 1, "vibracion": 3, "temp_motor": 1})
    dup = d.sample(frac=0.04, random_state=seed)                                          # reenvíos
    return pd.concat([d, dup]).sort_values(["semana", "camion"], kind="mergesort").reset_index(drop=True)

if __name__ == "__main__":
    d = flota(); print(len(d), d.averia_sig.mean().round(4)); print(d.describe().round(2))
    print(d.groupby(d.vibracion.isna()).averia_sig.mean())
