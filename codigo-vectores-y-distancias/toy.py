"""Los 12 camiones del juguete (medias de un mes): horas de trabajo/día, paradas/día, km/día."""
import numpy as np
T = np.array([
 # horas, paradas, km
 [7.5, 13.5, 180],  # 1 reparto urbano
 [8.0, 13.0, 250],  # 2 reparto urbano
 [6.0, 13.5, 160],  # 3 reparto urbano
 [7.0, 10.5, 300],  # 4 mixto
 [8.0,  6.0, 370],  # 5 regional
 [10.0, 7.0, 470],  # 6 regional
 [8.0,  8.5, 330],  # 7 regional
 [5.5,  5.0, 210],  # 8 lanzadera entre almacenes
 [10.0, 2.5, 660],  # 9 larga distancia
 [12.0, 1.0, 750],  # 10 larga distancia
 [9.0,  2.5, 610],  # 11 larga distancia
 [4.0,  9.5, 130],  # 12 media jornada
])
if __name__ == "__main__":
    from scipy.spatial.distance import cdist
    H = T[:, :2]
    for name, m in [("L2", "euclidean"), ("L1", "cityblock")]:
        D = cdist(H, H, m); np.fill_diagonal(D, np.inf)
        nn = D.argmin(1); srt = np.sort(D, 1)
        print(name, "vecino", (nn + 1).tolist()); print("  d1,d2", [f"{a:.2f}/{b:.2f}" for a, b in srt[:, :2]])
    K = T[:, [2, 1]]
    mu, sd = K.mean(0), K.std(0); Z = (K - mu) / sd
    print("media", mu, "desv", sd)
    for lab, X in [("crudo", K), ("escalado", Z)]:
        D = cdist(X, X); np.fill_diagonal(D, np.inf); srt=np.sort(D,1); print(lab, (D.argmin(1) + 1).tolist(), np.round(srt[:,0],3).tolist())
