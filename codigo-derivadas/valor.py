"""Derivadas desde cero: diferenciación automática en modo inverso (Valor, al estilo de micrograd
de A. Karpathy), modo directo (Dual), diferencias finitas y gradientes de regresión lineal y logística."""
import math
import numpy as np


class Valor:
    """Un número que recuerda de qué cuentas sale, para poder derivar hacia atrás."""

    def __init__(self, dato, hijos=(), op=""):
        self.dato = float(dato)
        self.grad = 0.0                    # ∂(salida)/∂(este valor), se rellena en atras()
        self._hijos = hijos
        self._op = op
        self._atras = lambda: None         # reparte self.grad entre los hijos (regla de la cadena)

    @staticmethod
    def _v(o):
        return o if isinstance(o, Valor) else Valor(o)

    def __add__(self, o):
        o = Valor._v(o); s = Valor(self.dato + o.dato, (self, o), "+")
        def atras():
            self.grad += s.grad            # ∂s/∂a = 1
            o.grad += s.grad               # ∂s/∂b = 1
        s._atras = atras
        return s

    def __mul__(self, o):
        o = Valor._v(o); s = Valor(self.dato * o.dato, (self, o), "*")
        def atras():
            self.grad += o.dato * s.grad   # ∂(ab)/∂a = b
            o.grad += self.dato * s.grad   # ∂(ab)/∂b = a
        s._atras = atras
        return s

    def __pow__(self, k):                  # k es un número fijo
        s = Valor(self.dato ** k, (self,), f"**{k}")
        def atras():
            self.grad += k * self.dato ** (k - 1) * s.grad
        s._atras = atras
        return s

    def _unaria(self, valor, derivada, op):
        s = Valor(valor, (self,), op)
        def atras():
            self.grad += derivada * s.grad
        s._atras = atras
        return s

    def exp(self):
        e = math.exp(self.dato); return self._unaria(e, e, "exp")

    def log(self):
        return self._unaria(math.log(self.dato), 1 / self.dato, "log")

    def sigmoide(self):
        p = 1 / (1 + math.exp(-self.dato)); return self._unaria(p, p * (1 - p), "σ")

    def tanh(self):
        t = math.tanh(self.dato); return self._unaria(t, 1 - t * t, "tanh")

    def relu(self):
        return self._unaria(max(0.0, self.dato), float(self.dato > 0), "relu")

    def __neg__(self): return self * -1
    def __sub__(self, o): return self + (-Valor._v(o))
    def __truediv__(self, o): return self * Valor._v(o) ** -1
    def __radd__(self, o): return self + o
    def __rmul__(self, o): return self * o
    def __rsub__(self, o): return Valor._v(o) - self
    def __rtruediv__(self, o): return Valor._v(o) / self

    def atras(self):
        """Orden topológico del grafo y regla de la cadena desde la salida hacia las entradas."""
        orden, visto = [], set()
        def visita(v):
            if v not in visto:
                visto.add(v)
                for h in v._hijos:
                    visita(h)
                orden.append(v)
        visita(self)
        for v in orden:
            v.grad = 0.0
        self.grad = 1.0                    # ∂salida/∂salida
        for v in reversed(orden):
            v._atras()

    def __repr__(self):
        return f"Valor(dato={self.dato:.6g}, grad={self.grad:.6g})"


class Dual:
    """Modo directo: cada número lleva su valor y su derivada respecto a UNA dirección de entrada."""

    def __init__(self, v, d=0.0):
        self.v, self.d = float(v), float(d)

    @staticmethod
    def _c(o):
        return o if isinstance(o, Dual) else Dual(o)

    def __add__(self, o): o = Dual._c(o); return Dual(self.v + o.v, self.d + o.d)
    def __mul__(self, o): o = Dual._c(o); return Dual(self.v * o.v, self.d * o.v + self.v * o.d)
    def __pow__(self, k): return Dual(self.v ** k, k * self.v ** (k - 1) * self.d)
    def exp(self): e = math.exp(self.v); return Dual(e, e * self.d)
    def log(self): return Dual(math.log(self.v), self.d / self.v)
    def sigmoide(self): p = 1 / (1 + math.exp(-self.v)); return Dual(p, p * (1 - p) * self.d)
    def tanh(self): t = math.tanh(self.v); return Dual(t, (1 - t * t) * self.d)
    def relu(self): return Dual(max(0.0, self.v), float(self.v > 0) * self.d)
    def __neg__(self): return self * -1
    def __sub__(self, o): return self + (-Dual._c(o))
    def __truediv__(self, o): return self * Dual._c(o) ** -1
    def __radd__(self, o): return self + o
    def __rmul__(self, o): return self * o
    def __rsub__(self, o): return Dual._c(o) - self
    def __rtruediv__(self, o): return Dual._c(o) / self


# funciones que valen para float, Valor y Dual: la misma f se deriva de tres maneras
def exp(x): return x.exp() if hasattr(x, "exp") else math.exp(x)
def log(x): return x.log() if hasattr(x, "log") else math.log(x)
def sigmoide(x): return x.sigmoide() if hasattr(x, "sigmoide") else 1 / (1 + math.exp(-x))
def tanh(x): return x.tanh() if hasattr(x, "tanh") else math.tanh(x)


def grad_inverso(f, x):
    """Gradiente de f: Rⁿ → R en una sola pasada hacia atrás."""
    xs = [Valor(v) for v in x]; y = f(*xs); y.atras()
    return y.dato, np.array([v.grad for v in xs])


def grad_directo(f, x):
    """Gradiente con modo directo: una pasada por cada entrada (n pasadas)."""
    g = []
    for i in range(len(x)):
        g.append(f(*[Dual(v, 1.0 if j == i else 0.0) for j, v in enumerate(x)]).d)
    return np.array(g)


def grad_numerico(f, x, h=1e-5):
    """Diferencias centrales: (f(x + h·eᵢ) − f(x − h·eᵢ)) / 2h. Error O(h²) más redondeo ~ ε/h."""
    x = np.asarray(x, float); g = np.zeros_like(x)
    for i in range(x.size):
        e = np.zeros_like(x); e[i] = h
        g[i] = (f(*(x + e)) - f(*(x - e))) / (2 * h)
    return g


def error_relativo(a, b):
    """‖a − b‖ / max(‖a‖, ‖b‖): la medida habitual de una comprobación de gradiente
    (con normas, no componente a componente: una componente casi nula daría un cociente engañoso)."""
    a, b = np.ravel(np.asarray(a, float)), np.ravel(np.asarray(b, float))
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(a), np.linalg.norm(b), 1e-300))


def sig(z):
    return 0.5 * (1 + np.tanh(0.5 * np.asarray(z, float)))       # σ(z) sin desbordamientos


def grad_mse(X, y, beta):
    """∇ (1/n)‖Xβ − y‖² = (2/n) Xᵀ(Xβ − y)."""
    return 2 / len(y) * X.T @ (X @ beta - y)


def grad_logloss(X, y, beta):
    """∇ de la pérdida logarítmica media = (1/n) Xᵀ(p − y), con p = σ(Xβ)."""
    return X.T @ (sig(X @ beta) - y) / len(y)


def hess_logloss(X, y, beta):
    """Hessiana = (1/n) Xᵀ diag(p(1 − p)) X: la información de Fisher observada por dato."""
    p = sig(X @ beta)
    return (X * (p * (1 - p))[:, None]).T @ X / len(y)
