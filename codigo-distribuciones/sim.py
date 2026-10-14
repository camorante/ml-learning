"""Simulador de la flota de §9: consumo por trayecto, alertas de frenada por camión y día,
duración de las paradas y entregas tarde por día."""
import numpy as np


def flota(seed=8):
    rng = np.random.default_rng(seed)
    # consumo (L/100 km) de 6 000 trayectos: 4 000 de larga distancia y 2 000 de reparto urbano
    tipo = np.r_[np.zeros(4000, int), np.ones(2000, int)]
    consumo = np.where(tipo == 0, rng.normal(32, 2.5, 6000), rng.normal(24, 2.0, 6000)).round(1)
    # alertas de frenada brusca: 300 camiones × 60 días; cada conductor tiene su propia tasa
    tasa = rng.gamma(shape=2.0, scale=1.0, size=300)           # media 2 alertas/día, unos más bruscos que otros
    alertas = rng.poisson(tasa[:, None], size=(300, 60))
    # duración de 5 000 paradas (minutos): efectos que se multiplican → lognormal
    paradas = rng.lognormal(np.log(12), 0.8, 5000).round(1)
    # entregas tarde de 20 al día durante 250 días, p = 10 %
    tarde = rng.binomial(20, 0.10, 250)
    return consumo, tipo, alertas, tasa, paradas, tarde
