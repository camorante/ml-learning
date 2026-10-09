"""Viajes de una flota simulada para el tema 'Preparar variables'."""
import numpy as np, pandas as pd

TIPOS = {"frigorífico": 33.0, "furgoneta": 12.0, "rígido": 22.0, "tráiler": 29.0}

def viajes(n=8000, n_camiones=60, n_clientes=1500, dias=240, seed=7):
    rng = np.random.default_rng(seed)
    tipo_cam = rng.choice(list(TIPOS), n_camiones, p=[0.15, 0.35, 0.3, 0.2])
    efic_cam = rng.normal(0, 2.0, n_camiones)            # unos camiones gastan más que otros
    desgaste = rng.uniform(0, 3.0, n_camiones)           # y empeoran con los meses (L/100 km en 240 días)
    pop = 1 / np.arange(1, n_clientes + 1) ** 0.7; pop /= pop.sum()
    efecto_cli = rng.normal(0, 3.0, n_clientes)          # rutas de montaña, ciudad, autovía...
    cam = rng.integers(0, n_camiones, n); cli = rng.choice(n_clientes, n, p=pop)
    dia = np.sort(rng.uniform(0, dias, n))
    hora = np.round(rng.normal(9, 5, n) % 24, 2) % 24
    km = np.clip(np.exp(rng.normal(np.log(140), 0.8, n)), 8, 950)
    tipo = tipo_cam[cam]
    carga = np.clip(rng.uniform(0, 1, n) * np.where(tipo == "furgoneta", 3.5, 24), 0, None)
    consumo = (np.array([TIPOS[t] for t in tipo]) + 0.32 * carga + efecto_cli[cli]
               + 2.5 * np.cos(2 * np.pi * (hora - 14) / 24)            # tráfico: peor a media tarde
               + efic_cam[cam] + desgaste[cam] * dia / dias
               + 9 * np.exp(-km / 40)                                    # trayectos cortos gastan más
               + rng.normal(0, 1.5, n))
    consumo = np.maximum(consumo, 4.0)                                  # ningún camión baja de 4 L/100 km
    return pd.DataFrame(dict(dia=dia.round(3), camion=[f"C{c:02d}" for c in cam], tipo=tipo,
                             cliente=[f"K{c:03d}" for c in cli], hora=hora, km=km.round(1),
                             carga_t=carga.round(2), consumo=consumo.round(2)))

if __name__ == "__main__":
    d = viajes(); print(d.head()); print(d.describe().round(2)); print(d.cliente.value_counts().describe())
    print(d.groupby("tipo").consumo.mean().round(1))
