"""Datos del tema 3. Los viajes de carga y consumo y el XOR son EXACTAMENTE los del tema 1
(misma función, misma semilla); se añaden la zona normal del motor y el perfil de velocidad de una ruta."""
import numpy as np


def carga_consumo(seed=11):
    """30 viajes: carga (t) y consumo (L/100 km). Lo normal es una franja diagonal;
    los que «no cuadran» (1) quedan por encima o por debajo de ella. (Copia literal del tema 1.)"""
    rng = np.random.default_rng(seed)
    c = np.round(rng.uniform(2, 24, 30), 1)
    base = 24 + 0.55 * c
    desv = np.r_[rng.uniform(-1.3, 1.3, 18), rng.uniform(3.6, 6, 6), -rng.uniform(3.6, 6, 6)]
    q = np.round(base + desv, 1)
    yy = np.r_[np.zeros(18, int), np.ones(12, int)]
    o = rng.permutation(30)
    return np.c_[c, q][o], yy[o]


XOR = (np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float), np.array([0, 1, 1, 0]))

# Zona normal del motor: régimen (rpm) y temperatura del refrigerante (°C) en marcha por carretera.
RPM0, RPM_S, TMP0, TMP_S = 1400.0, 350.0, 89.0, 7.0      # centro y semiejes de la elipse «normal»


def radio(rpm, tmp):
    """Distancia a la zona normal en «semiejes»: < 1 dentro de la elipse, > 1 fuera."""
    return np.hypot((np.asarray(rpm) - RPM0) / RPM_S, (np.asarray(tmp) - TMP0) / TMP_S)


def motor(n_normal=80, n_raras=40, seed=3):
    """Lecturas revisadas por el taller: normales dentro de la elipse (radio < 0,95) y anomalías
    alrededor, en todas direcciones (radio 1,08–1,9): sobrecalentamiento, termostato abierto,
    motor muy revolucionado o forzado a bajas vueltas."""
    rng = np.random.default_rng(seed)
    a = rng.uniform(0, 2 * np.pi, n_normal); r = 0.95 * np.sqrt(rng.uniform(0, 1, n_normal))
    b = rng.uniform(0, 2 * np.pi, n_raras); s = rng.uniform(1.08, 1.9, n_raras)
    rpm = np.r_[RPM0 + RPM_S * r * np.cos(a), RPM0 + RPM_S * s * np.cos(b)]
    tmp = np.r_[TMP0 + TMP_S * r * np.sin(a), TMP0 + TMP_S * s * np.sin(b)]
    y = np.r_[np.zeros(n_normal, int), np.ones(n_raras, int)]
    o = rng.permutation(len(y))
    return np.c_[np.round(rpm, -1), np.round(tmp, 1)][o], y[o]


def motor_nuevas(n=2000, seed=8):
    """Lecturas nuevas repartidas por igual en un disco de radio 1,9 (en semiejes) alrededor del centro;
    anomalía = fuera de la elipse (radio > 1)."""
    rng = np.random.default_rng(seed)
    a = rng.uniform(0, 2 * np.pi, n); r = 1.9 * np.sqrt(rng.uniform(0, 1, n))
    rpm = np.round(RPM0 + RPM_S * r * np.cos(a), -1); tmp = np.round(TMP0 + TMP_S * r * np.sin(a), 1)
    return np.c_[rpm, tmp], (radio(rpm, tmp) > 1).astype(int)


def perfil(km):
    """Velocidad media (km/h) a lo largo de una ruta de 100 km: autovía a 85, un pueblo en el km 22,
    una ciudad entre los km 45 y 55, obras en el km 75 y un puerto de montaña al final."""
    km = np.asarray(km, float)
    s = lambda x, c, w: 1 / (1 + np.exp(-(x - c) / w))
    v = 85 - 35 * (s(km, 20, 0.6) - s(km, 24, 0.6)) - 55 * (s(km, 44, 1.0) - s(km, 56, 1.0))
    v -= 25 * (s(km, 72, 0.5) - s(km, 78, 0.5)) + 20 * s(km, 88, 2.0)
    return v


KM = np.linspace(0, 100, 201)                           # una lectura cada 500 m
