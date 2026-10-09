"""Producto punto, proyecciones y similitud coseno, desde cero."""
import numpy as np

def punto(a, b):
    """Suma de productos por parejas: a·b = a1*b1 + a2*b2 + ..."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.shape[-1] != b.shape[-1]:
        raise ValueError("los vectores deben tener la misma longitud")
    return float(sum(x * y for x, y in zip(a, b)))

def norma(a):
    return punto(a, a) ** 0.5

def coseno(a, b, eps=1e-12):
    """cos(a, b) = a·b / (||a|| ||b||); None si algún vector es (casi) cero."""
    na, nb = norma(a), norma(b)
    if na < eps or nb < eps:
        return None
    return punto(a, b) / (na * nb)

def angulo_grados(a, b):
    c = coseno(a, b)
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))

def distancia_coseno(a, b):
    return 1.0 - coseno(a, b)

def proyeccion(a, b):
    """Sombra de a sobre la recta de b: (a·b / b·b) b. Devuelve (proyección, resto ortogonal)."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    p = punto(a, b) / punto(b, b) * b
    return p, a - p

def gram_schmidt(V):
    """Base ortonormal de las filas de V (salta las dependientes)."""
    Q = []
    for v in np.asarray(V, float):
        w = v - sum(punto(v, q) * q for q in Q)
        if norma(w) > 1e-10:
            Q.append(w / norma(w))
    return np.array(Q)

def normalizar_filas(X, eps=1e-12):
    X = np.asarray(X, float); n = np.sqrt((X * X).sum(1, keepdims=True))
    return X / np.maximum(n, eps)

def matriz_coseno(X, Y=None):
    """Todas las similitudes coseno a la vez: normalizar filas y multiplicar X Yᵀ."""
    Xn = normalizar_filas(X); Yn = Xn if Y is None else normalizar_filas(Y)
    return Xn @ Yn.T

def top_k(consultas, base, k=3):
    """Índices y cosenos de los k vectores de 'base' más parecidos a cada consulta."""
    S = matriz_coseno(consultas, base)
    idx = np.argsort(-S, axis=1, kind="stable")[:, :k]
    return idx, np.take_along_axis(S, idx, 1)

def pearson(x, y):
    """Correlación de Pearson = coseno de los vectores centrados."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    return coseno(x - x.mean(), y - y.mean())

def puntuacion(w, x, b):
    """Modelo lineal: w·x + b. El signo dice el lado de la frontera."""
    return punto(w, x) + b

def distancia_frontera(w, x, b):
    """Distancia con signo de x al hiperplano w·x + b = 0."""
    return puntuacion(w, x, b) / norma(w)
