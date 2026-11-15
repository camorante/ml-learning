"""Campo receptivo (cuántas lecturas de la entrada ve una neurona) y número de parámetros, capa a capa."""
capas = [("conv k=5", 5, 1), ("max-pool 2", 2, 2), ("conv k=5", 5, 1), ("max-pool 2", 2, 2), ("conv k=5", 5, 1),
         ("max-pool 2", 2, 2), ("conv k=5", 5, 1)]
r, j = 1, 1                                        # campo receptivo y salto (distancia en la entrada entre vecinas)
print(f"{'capa':12s}{'campo':>7s}{'salto':>7s}   segundos (10 lecturas/s)")
for nombre, k, s in capas:
    r, j = r + (k - 1) * j, j * s
    print(f"{nombre:12s}{r:7d}{j:7d}   {r / 10:.1f} s")


def conv(c_in, c_out, k):
    return c_out * c_in * k + c_out


p = [("conv 1→4, k=5", conv(1, 4, 5)), ("conv 4→8, k=5", conv(4, 8, 5)), ("lineal 8→1", 8 + 1)]
print("red convolucional:", ", ".join(f"{a} {b}" for a, b in p), "· total", sum(b for _, b in p))
print("red densa: lineal 64→32", 64 * 32 + 32, "+ lineal 32→1", 32 + 1, "· total", 64 * 32 + 32 + 33)
print("la red convolucional con ventanas de 640 lecturas (64 s): los mismos", sum(b for _, b in p),
      "· la densa:", 640 * 32 + 32 + 33)
print("imagen 224×224×3 → 64 neuronas densas:", f"{224 * 224 * 3 * 64 + 64:,}".replace(",", " "),
      "· → 64 filtros 3×3:", f"{64 * 3 * 3 * 3 + 64:,}".replace(",", " "))
