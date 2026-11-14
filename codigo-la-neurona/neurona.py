"""Una neurona artificial desde cero: suma ponderada + sesgo + activación.
Perceptrón de Rosenblatt (escalón + regla de corrección) y neurona sigmoide entrenada
con entropía cruzada, que es exactamente una regresión logística."""
import numpy as np


def puntuacion(X, w, b):
    """z = w·x + b para cada fila de X (el producto punto más el sesgo)."""
    return np.asarray(X, float) @ np.asarray(w, float) + b


def escalon(z):
    """Salida del perceptrón: 1 si z > 0, 0 si no."""
    return (np.asarray(z) > 0).astype(int)


def sigmoide(z):
    """σ(z) = 1 / (1 + e^(−z)), calculada sin desbordar para |z| grande."""
    z = np.asarray(z, float)
    e = np.exp(-np.abs(z))
    return np.where(z >= 0, 1 / (1 + e), e / (1 + e))


class Perceptron:
    """Regla de Rosenblatt: si el ejemplo cae en el lado equivocado (o justo en la frontera),
    w ← w + η·y·x y b ← b + η·y, con y = +1 (revisar) o −1 (no revisar)."""

    def __init__(self, eta=0.1, epocas=1000, w0=None, b0=0.0):
        self.eta, self.epocas, self.w0, self.b0 = eta, epocas, w0, b0

    def fit(self, X, y, orden=None):
        X = np.asarray(X, float)
        s = np.where(np.asarray(y) == 1, 1.0, -1.0)
        w = np.zeros(X.shape[1]) if self.w0 is None else np.array(self.w0, float)
        b = float(self.b0)
        orden = range(len(X)) if orden is None else orden
        self.correcciones, self.fallos_por_pasada = 0, []
        for _ in range(self.epocas):
            fallos = 0
            for i in orden:
                if s[i] * (X[i] @ w + b) <= 0:          # fallo: corrige hacia el ejemplo
                    w = w + self.eta * s[i] * X[i]
                    b = b + self.eta * s[i]
                    fallos += 1
            self.fallos_por_pasada.append(fallos)
            self.correcciones += fallos
            if fallos == 0:                              # una pasada entera sin fallos: para
                break
        self.w, self.b = w, b
        return self

    def predict(self, X):
        return escalon(puntuacion(X, self.w, self.b))


class NeuronaSigmoide:
    """p = σ(w·x + b) entrenada minimizando la entropía cruzada media
    L = −(1/n) Σ [y log p + (1 − y) log(1 − p)] (+ α/2 ‖w‖² si alfa > 0).
    Es la regresión logística: mismo modelo y misma pérdida."""

    def __init__(self, alfa=0.0):
        self.alfa = alfa

    def perdida(self, X, y, w, b):
        z = puntuacion(X, w, b)
        # −log σ(z) = log(1 + e^(−z)) y −log(1 − σ(z)) = log(1 + e^z): estable con logaddexp
        L = np.mean(y * np.logaddexp(0, -z) + (1 - y) * np.logaddexp(0, z))
        return L + 0.5 * self.alfa * np.dot(w, w)

    def gradiente(self, X, y, w, b):
        """∂L/∂z = p − y por la regla de la cadena; ∇w = Xᵀ(p − y)/n + αw, ∂L/∂b = media(p − y)."""
        X = np.asarray(X, float)
        r = sigmoide(puntuacion(X, w, b)) - y
        return X.T @ r / len(y) + self.alfa * w, r.mean()

    def fit(self, X, y, metodo="newton", eta=0.5, iters=100, tol=1e-12):
        X = np.asarray(X, float); y = np.asarray(y, float)
        Xa = np.c_[X, np.ones(len(X))]                   # el sesgo como peso de una entrada fija 1
        th = np.zeros(Xa.shape[1])
        R = np.diag(np.r_[np.full(X.shape[1], self.alfa), 0.0])   # el sesgo no se regulariza
        for _ in range(iters):
            gw, gb = self.gradiente(X, y, th[:-1], th[-1])
            g = np.r_[gw, gb]
            if metodo == "newton":                       # IRLS: H = XᵀWX/n + αI
                p = sigmoide(Xa @ th)
                H = Xa.T @ (Xa * (p * (1 - p))[:, None]) / len(y) + R
                paso = np.linalg.solve(H, g)
            else:                                        # descenso de gradiente
                paso = eta * g
            th = th - paso
            if np.max(np.abs(paso)) < tol:
                break
        self.w, self.b = th[:-1], th[-1]
        return self

    def predict_proba(self, X):
        return sigmoide(puntuacion(X, self.w, self.b))
