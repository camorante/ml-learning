import numpy as np

f = lambda z: 1 / (1 + np.exp(-z))          # sigmoide; vale también con números complejos
z = -0.5                                    # el z de la neurona del juguete
exacta = f(z) * (1 - f(z))                  # σ' = σ(1 − σ) = 0.2350037122...
eps = np.finfo(float).eps
print(f"σ'(−0,5) = {exacta:.15f}   ε de la máquina = {eps:.2e}")
print("      h     adelante   central   paso complejo")
for k in range(1, 15, 1):
    h = 10.0 ** -k
    ad = (f(z + h) - f(z)) / h
    ce = (f(z + h) - f(z - h)) / (2 * h)
    co = f(z + 1j * h).imag / h             # sin resta: no hay cancelación
    print(f"  1e-{k:<2d}   {abs(ad - exacta):8.1e}  {abs(ce - exacta):8.1e}  {abs(co - exacta):8.1e}")
print(f"h óptimo teórico: adelante ≈ √ε = {np.sqrt(eps):.1e}   central ≈ ∛ε = {np.cbrt(eps):.1e}")
