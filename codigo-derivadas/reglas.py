import numpy as np

sig = lambda x: 1 / (1 + np.exp(-x))
TABLA = {   # nombre: (f, derivada escrita con la regla)
    "x³":                  (lambda x: x ** 3,               lambda x: 3 * x ** 2),
    "eˣ":                  (np.exp,                         np.exp),
    "log x":               (np.log,                         lambda x: 1 / x),
    "√x":                  (np.sqrt,                        lambda x: 0.5 / np.sqrt(x)),
    "σ(x)":                (sig,                            lambda x: sig(x) * sig(-x)),  # σ(1 − σ)
    "tanh x":              (np.tanh,                        lambda x: 1 - np.tanh(x) ** 2),
    "softplus":            (lambda x: np.log1p(np.exp(x)),  sig),
    "x·eˣ (producto)":     (lambda x: x * np.exp(x),        lambda x: np.exp(x) + x * np.exp(x)),
    "x/(1+x²) (cociente)": (lambda x: x / (1 + x * x),      lambda x: (1 - x * x) / (1 + x * x) ** 2),
    "σ(3x²−1) (cadena)":   (lambda x: sig(3 * x * x - 1),   lambda x: sig(3 * x * x - 1) * sig(1 - 3 * x * x) * 6 * x),
}
x = np.linspace(0.1, 3, 300)
for nombre, (f, d) in TABLA.items():
    paso_complejo = f(x + 1e-30j).imag / 1e-30      # derivada "exacta" numérica (f analítica)
    err = np.max(np.abs(d(x) - paso_complejo) / np.abs(paso_complejo))
    print(f"{nombre:22s} error relativo máx. {err:.1e}")
relu = lambda x: np.maximum(x, 0.0); xr = np.array([-1.0, -1e-3, 1e-3, 1.0, 0.0])
print("ReLU, dif. centrales h=1e-6 en", xr.tolist(), "→",
      ((relu(xr + 1e-6) - relu(xr - 1e-6)) / 2e-6).round(6).tolist())
z = 30.0
print(f"σ'(30): σ(z)·σ(−z) = {sig(z) * sig(-z):.6e}   σ(z)·(1 − σ(z)) = {sig(z) * (1 - sig(z)):.6e}"
      "  ← 1 − σ(30) se cancela")
