import json
import math
import numpy as np
from manim import *

# Paleta oscura de la serie
BG = "#0E131B"; INK = "#E4EAF3"; MUTED = "#9BA7B9"; LINE = "#2B3545"
MEAN = "#6EA6FF"; SAMPLE = "#AE9FF3"; POINT = "#FF9A55"; GOOD = "#4CC38A"; WARN = "#F08A5D"
FONT = "Inter"
config.background_color = BG
D = json.load(open("../rn/r2/embed.json"))
V, C = np.array(D["v"]), np.array(D["c"])


def T(s, sc=0.45, col=INK, w=None):
    return Text(s, font=FONT, color=col, weight=w or NORMAL).scale(sc)


def fmt(v, d=1):
    s = f"{abs(v):.{d}f}".replace(".", ",")
    return ("−" if v < 0 and round(abs(v), d) != 0 else "") + s


relu = lambda z: np.maximum(z, 0)
sig = lambda z: 1 / (1 + np.exp(-z))


def codos(ks):
    """Recta + codos ReLU ajustados por mínimos cuadrados (como la idea 3 de la página)."""
    B = lambda v: np.column_stack([np.ones_like(v), v / 100] + [relu(v / 100 - k / 100) for k in ks])
    co, *_ = np.linalg.lstsq(B(V), C, rcond=None)
    f = lambda v: float((B(np.array([float(v)])) @ co)[0])
    return f, float(np.sqrt(np.mean((B(V) @ co - C) ** 2)))


class Panel:
    """Ejes 2D sencillos: mapea (x, y) de datos a la escena dentro de una caja."""

    def __init__(self, x0, x1, y0, y1, box, xt=(), xf=None, xl=None, yt=(), yf=None, yl=None):
        self.x0, self.x1, self.y0, self.y1 = x0, x1, y0, y1
        self.L, self.R, self.B, self.Tp = box
        g = VGroup(Line([self.L, self.B, 0], [self.R, self.B, 0], stroke_color=MUTED, stroke_width=2))
        for v in xt:
            g.add(Line(self.P(v, y0), self.P(v, y0) + 0.1 * UP, stroke_color=MUTED, stroke_width=2))
            g.add(T(xf(v) if xf else str(v), 0.28, MUTED).next_to(self.P(v, y0), DOWN, buff=0.1))
        for v in yt:
            g.add(Line(self.P(x0, v), self.P(x1, v), stroke_color=LINE, stroke_width=1.5))
            g.add(T(yf(v) if yf else str(v), 0.28, MUTED).next_to(self.P(x0, v), LEFT, buff=0.1))
        if xl:
            g.add(T(xl, 0.3, INK).next_to([self.R, self.B, 0], DOWN, buff=0.42).align_to([self.R, 0, 0], RIGHT))
        if yl:
            g.add(T(yl, 0.3, MUTED).next_to([self.L, self.Tp, 0], UP, buff=0.12).align_to([self.L, 0, 0], LEFT))
        self.g = g

    def P(self, x, y):
        return np.array([self.L + (x - self.x0) / (self.x1 - self.x0) * (self.R - self.L),
                         self.B + (y - self.y0) / (self.y1 - self.y0) * (self.Tp - self.B), 0])

    def curve(self, f, a, b, col, w=4, n=240):
        return VMobject(stroke_color=col, stroke_width=w).set_points_as_corners(
            [self.P(x, min(self.y1, max(self.y0, f(x)))) for x in np.linspace(a, b, n)])


class Escena(Scene):
    def caption(self, text, sub=None):
        t = Text(text, font=FONT, weight=SEMIBOLD, color=INK).scale(0.62)
        grp = VGroup(t)
        if sub:
            grp.add(Text(sub, font=FONT, color=MUTED).scale(0.42)); grp.arrange(DOWN, buff=0.18)
        grp.to_edge(UP, buff=0.4)
        band = Rectangle(width=config.frame_width + 1, height=1.55, stroke_width=0, fill_color=BG, fill_opacity=1).move_to([0, config.frame_height / 2 - 0.72, 0])
        return VGroup(band, grp).set_z_index(10)

    def mark(self, name):
        self.marks.setdefault(name, []).append(round(self.renderer.time, 3))

    def swap(self, old, new):
        self.mark("caption")
        if old is None:
            self.play(FadeIn(new, shift=0.2 * DOWN), run_time=0.8)
        else:
            self.play(FadeOut(old, shift=0.2 * UP), FadeIn(new, shift=0.2 * DOWN), run_time=0.8)
        return new

    def clear_but(self, keep):
        self.play(*[FadeOut(m) for m in self.mobjects if m is not keep], run_time=0.6)

    def construct(self):
        self.marks = {}
        title = Text("Funciones de activación", font=FONT, weight=BOLD, color=INK).scale(0.9)
        sub = Text("la pieza que deja doblarse a una red neuronal", font=FONT, color=MUTED).scale(0.46)
        VGroup(title, sub).arrange(DOWN, buff=0.35)
        self.mark("title"); self.play(Write(title), run_time=1.4)
        self.mark("subtitle"); self.play(FadeIn(sub, shift=0.2 * UP)); self.wait(1.5)
        self.mark("title_out"); self.play(FadeOut(title), FadeOut(sub))

        # ---------- 1. después de la suma, una curva ----------
        cap = self.swap(None, self.caption("Después de la suma, una curva", "la neurona calcula z = w · x + b y saca f(z)"))
        Pa = Panel(-5, 5, -1.3, 2.3, (-6.0, 1.4, -2.9, 1.8), xt=(-4, -2, 0, 2, 4), xf=lambda v: fmt(v, 0), xl="z (suma que entra)",
                   yt=(-1, 0, 1, 2), yf=lambda v: fmt(v, 0), yl="lo que sale")
        ACTS = [("Escalón", lambda z: 1.0 if z > 0 else 0.0, "0 o 1: sí o no (el perceptrón)"),
                ("Sigmoide", lambda z: float(sig(z)), "entre 0 y 1: una probabilidad"),
                ("Tanh", lambda z: math.tanh(z), "entre −1 y 1, centrada en 0"),
                ("ReLU", lambda z: max(0.0, z), "0 si es negativa; z tal cual si no")]
        self.play(FadeIn(Pa.g), run_time=0.8)
        info = None; cur = None
        for i, (nom, f, rng) in enumerate(ACTS):
            crv = Pa.curve(f, -5, 2.3 if nom == "ReLU" else 5, MEAN, 6, 400)
            card = VGroup(T(nom, 0.62, MEAN, BOLD), T(rng, 0.38, MUTED)).arrange(DOWN, aligned_edge=LEFT, buff=0.18).move_to([4.4, 0.2, 0]).align_to([2.2, 0, 0], LEFT)
            self.mark("act")
            if cur is None:
                self.play(Create(crv), FadeIn(card), run_time=1.0)
            else:
                self.play(Transform(cur, crv), FadeOut(info), FadeIn(card), run_time=1.0)
                crv = cur
            cur, info = crv, card
            self.wait(2.0)
        note = VGroup(T("ReLU: la más usada hoy", 0.38, GOOD, BOLD), T("en las capas ocultas", 0.38, GOOD, BOLD)).arrange(DOWN, aligned_edge=LEFT, buff=0.12).move_to([4.4, -1.5, 0]).align_to([2.2, 0, 0], LEFT)
        self.play(FadeIn(note, shift=0.1 * UP), run_time=0.6); self.wait(1.4)
        self.clear_but(cap)

        # ---------- 2. sin activación, dos capas = una ----------
        cap = self.swap(cap, self.caption("Sin activación, dos capas valen lo mismo que una", "dos neuronas en fila: la velocidad v entra en la primera"))
        e1 = T("0,5 · (−0,8 · v + 56) + 30", 0.6, INK, BOLD)
        e2 = T("= −0,4 · v + 58", 0.6, MEAN, BOLD)
        e3 = T("una sola neurona: siempre una recta", 0.4, MUTED)
        eqs = VGroup(e1, e2, e3).arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to([3.5, 0.5, 0]).align_to([1.4, 0, 0], LEFT)
        Pb = Panel(0, 120, -10, 70, (-6.0, 0.6, -2.9, 1.8), xt=(0, 40, 80, 120), xl="velocidad (km/h)", yt=(0, 30, 60), yl="salida")
        lin = Pb.curve(lambda v: -0.4 * v + 58, 0, 120, MEAN, 6)
        self.mark("formula"); self.play(FadeIn(Pb.g), FadeIn(e1), run_time=0.9); self.wait(0.8)
        self.play(FadeIn(e2, shift=0.1 * DOWN), Create(lin), run_time=1.0)
        self.play(FadeIn(e3), run_time=0.5); self.wait(2.2)
        kink = Pb.curve(lambda v: 0.5 * max(0.0, -0.8 * v + 56) + 30, 0, 120, MEAN, 6, 600)
        r1 = T("con ReLU entre las dos:", 0.46, POINT, BOLD)
        r2 = T("0,5 · máx(0, −0,8 · v + 56) + 30", 0.5, INK, BOLD)
        r3 = T("aparece un codo en 70 km/h", 0.4, MUTED)
        rr = VGroup(r1, r2, r3).arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to(eqs).align_to([1.4, 0, 0], LEFT)
        dk = Dot(Pb.P(70, 30), radius=0.11, color=POINT)
        # el título cambia a la vez que el dibujo pasa al caso con ReLU (sin marca nueva: el audio no se mueve)
        cap2 = self.caption("Con una ReLU en medio, la recta se dobla", "las mismas dos neuronas, con ReLU después de la primera")
        self.mark("kink"); self.play(FadeOut(eqs), FadeIn(rr), Transform(lin, kink),
                                     FadeOut(cap, shift=0.2 * UP), FadeIn(cap2, shift=0.2 * DOWN), run_time=1.2)
        cap = cap2
        self.play(FadeIn(dk, scale=1.6), run_time=0.4); self.wait(2.8)
        self.clear_but(cap)

        # ---------- 3. codos que siguen la U ----------
        cap = self.swap(cap, self.caption("Unos pocos codos siguen la U del consumo", "60 tramos de ruta: consumo según la velocidad media"))
        Pc = Panel(10, 115, 20, 60, (-6.0, 1.6, -2.9, 1.8), xt=(20, 40, 60, 80, 100), xl="velocidad (km/h)", yt=(20, 30, 40, 50, 60), yl="L/100 km")
        dots = VGroup(*[Dot(Pc.P(v, c), radius=0.055, color=POINT) for v, c in zip(V, C)])
        self.mark("dots"); self.play(FadeIn(Pc.g), LaggedStart(*[FadeIn(d) for d in dots], lag_ratio=0.02), run_time=1.2)
        steps = [([], "solo una recta"), ([60], "1 codo en 60 km/h"), ([40, 73], "2 codos: 40 y 73 km/h")]
        fit = None; lab = None; vl = VGroup()
        for ks, txt_ in steps:
            f, e = codos(ks)
            crv = Pc.curve(f, 10, 115, MEAN, 6, 300)
            nl = VGroup(*[DashedLine(Pc.P(k, 20), Pc.P(k, 60), color=SAMPLE, stroke_width=2.5) for k in ks])
            card = VGroup(T(txt_, 0.42, INK, BOLD), T("error medio", 0.34, MUTED), T(fmt(e, 2) + " L/100 km", 0.62, MEAN, BOLD)).arrange(DOWN, aligned_edge=LEFT, buff=0.14).move_to([4.6, 0.6, 0]).align_to([2.4, 0, 0], LEFT)
            self.mark("fit")
            if fit is None:
                self.play(Create(crv), FadeIn(card), run_time=1.0); fit = crv
            else:
                self.play(Transform(fit, crv), FadeOut(lab), FadeOut(vl), FadeIn(nl), FadeIn(card), run_time=1.2)
            lab, vl = card, nl
            self.wait(2.4)
        nn = VGroup(T("cada codo es una neurona ReLU", 0.38, GOOD, BOLD), T("la red del juguete los pone en 40 y 73", 0.34, MUTED)).arrange(DOWN, aligned_edge=LEFT, buff=0.12).move_to([4.6, -1.6, 0]).align_to([2.4, 0, 0], LEFT)
        self.play(FadeIn(nn, shift=0.1 * UP), run_time=0.6); self.wait(1.8)
        self.clear_but(cap)

        # ---------- 4. la pendiente y la corrección que se apaga ----------
        cap = self.swap(cap, self.caption("La pendiente decide si la neurona aprende", "la corrección se multiplica por la pendiente en cada capa"))
        Pd = Panel(-5, 5, 0, 1.15, (-5.9, -0.9, -2.0, 1.6), xt=(-4, 0, 4), xf=lambda v: fmt(v, 0), xl="z", yt=(0, 0.25, 1), yf=lambda v: fmt(v, 2 if v == 0.25 else 0), yl="pendiente")
        ds = Pd.curve(lambda z: float(sig(z) * (1 - sig(z))), -5, 5, SAMPLE, 6, 300)
        dr = VMobject(stroke_color=GOOD, stroke_width=6).set_points_as_corners([Pd.P(-5, 0), Pd.P(0, 0), Pd.P(0, 1), Pd.P(5, 1)])
        l1 = T("sigmoide: como mucho 0,25", 0.36, SAMPLE, BOLD).move_to([-3.4, -3.05, 0]).align_to([-6.2, 0, 0], LEFT)
        l2 = T("ReLU: 1 si está encendida, 0 si no", 0.36, GOOD, BOLD).move_to([-3.4, -3.55, 0]).align_to([-6.2, 0, 0], LEFT)
        self.mark("slopes"); self.play(FadeIn(Pd.g), Create(ds), FadeIn(l1), run_time=1.0)
        self.play(Create(dr), FadeIn(l2), run_time=0.9); self.wait(1.6)
        # barras: lo que llega a cada una de 10 capas (escala logarítmica)
        Pe = Panel(0.5, 10.5, -6, 0, (1.1, 6.3, -2.0, 1.6), xt=(1, 5, 10), xl="capa", yt=(-6, -4, -2, 0), yf=lambda v: "1" if v == 0 else "10" + "⁻" + "⁰¹²³⁴⁵⁶⁷⁸⁹"[-v], yl="corrección que llega")
        self.play(FadeIn(Pe.g), run_time=0.6)
        bs = VGroup()
        for k in range(1, 11):
            val = max(-6, (10 - k) * math.log10(0.25))
            top, bot = Pe.P(k, val), Pe.P(k, -6)
            bs.add(Rectangle(width=0.36, height=max(0.02, top[1] - bot[1]), stroke_width=0, fill_color=WARN if val < -4 else SAMPLE, fill_opacity=0.85).move_to((top + bot) / 2))
        self.mark("bars"); self.play(LaggedStart(*[GrowFromEdge(b, DOWN) for b in bs[::-1]], lag_ratio=0.12), run_time=1.6)
        t1 = T("sigmoide, 10 capas: 0,25⁹ ≈ 0,0000038", 0.36, WARN, BOLD).move_to([3.7, -3.3, 0])
        self.play(FadeIn(t1), run_time=0.5); self.wait(2.2)
        br = VGroup(*[Rectangle(width=0.36, height=Pe.P(k, 0)[1] - Pe.P(k, -6)[1], stroke_width=0, fill_color=GOOD, fill_opacity=0.85).move_to((Pe.P(k, 0) + Pe.P(k, -6)) / 2) for k in range(1, 11)])
        t2 = T("ReLU: llega entera a todas las capas", 0.36, GOOD, BOLD).move_to(t1)
        self.mark("relu"); self.play(Transform(bs, br), FadeOut(t1), FadeIn(t2), run_time=1.2); self.wait(2.6)
        self.clear_but(cap)

        # ---------- Cierre ----------
        self.mark("legend")
        self.play(FadeOut(cap), run_time=0.6)
        e1 = Text("Sin activación, solo rectas", font=FONT, weight=BOLD, color=INK).scale(0.7)
        e2 = VGroup(T("la activación se aplica después de la suma ponderada", 0.42, MUTED),
                    T("sin ella, cien capas equivalen a una", 0.42, MUTED),
                    T("con ReLU, cada neurona aporta un codo", 0.42, MUTED),
                    T("su pendiente decide cuánta corrección pasa", 0.42, MUTED)).arrange(DOWN, buff=0.18)
        VGroup(e1, e2).arrange(DOWN, buff=0.45)
        self.mark("end_text"); self.play(Write(e1), run_time=1.2); self.play(FadeIn(e2, shift=0.2 * UP, lag_ratio=0.15), run_time=1.0)
        self.wait(3.3)
        self.mark("fade"); self.play(FadeOut(VGroup(e1, e2)), run_time=1)
        self.marks["total"] = [round(self.renderer.time, 3)]
        json.dump(self.marks, open("marks.json", "w"), indent=1)
