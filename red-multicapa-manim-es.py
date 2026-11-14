import json
import math
import numpy as np
from manim import *

# Paleta oscura de la serie
BG = "#0E131B"; INK = "#E4EAF3"; MUTED = "#9BA7B9"; LINE = "#2B3545"
MEAN = "#6EA6FF"; SAMPLE = "#AE9FF3"; POINT = "#FF9A55"; GOOD = "#4CC38A"; WARN = "#F08A5D"
FONT = "Inter"
config.background_color = BG
D = json.load(open("../rn/r3/embed.json", encoding="utf-8"))
CC, CQ, CY = np.array(D["cc"]), np.array(D["cq"]), np.array(D["cy"])
MR, MT, MY = np.array(D["mr"]), np.array(D["mt"]), np.array(D["my"])
sig = lambda z: 1 / (1 + np.exp(-z))


def T(s, sc=0.45, col=INK, w=None):
    return Text(s, font=FONT, color=col, weight=w or NORMAL).scale(sc)


def votos(c, q, aA=26.5, aB=21.5, m=0.55):
    return sig(2 * (q - aA - m * c)), sig(2 * (aB + m * c - q))


class Panel:
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

    def recta(self, a, m, col, w=5):
        """Recta consumo = a + m·carga recortada al recuadro."""
        xs = np.linspace(self.x0, self.x1, 400)
        ys = a + m * xs
        k = (ys >= self.y0) & (ys <= self.y1)
        return Line(self.P(xs[k][0], ys[k][0]), self.P(xs[k][-1], ys[k][-1]), stroke_color=col, stroke_width=w)

    def zona(self, a, m, arriba, col=POINT, op=0.16):
        xs = np.linspace(self.x0, self.x1, 120)
        ys = np.clip(a + m * xs, self.y0, self.y1)
        borde = [self.P(x, y) for x, y in zip(xs, ys)]
        tope = [self.P(self.x1, self.y1 if arriba else self.y0), self.P(self.x0, self.y1 if arriba else self.y0)]
        return Polygon(*borde, *tope, stroke_width=0, fill_color=col, fill_opacity=op)


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

    def swap(self, old, new, extra=()):
        self.mark("caption")
        if old is None:
            self.play(FadeIn(new, shift=0.2 * DOWN), *extra, run_time=0.8)
        else:
            self.play(FadeOut(old, shift=0.2 * UP), FadeIn(new, shift=0.2 * DOWN), *extra, run_time=0.8)
        return new

    def clear_but(self, keep):
        self.play(*[FadeOut(m) for m in self.mobjects if m is not keep], run_time=0.6)

    def card(self, lines, x=4.5, y=0.4, left=2.3):
        g = VGroup(*[T(s, sc, col, BOLD if b else None) for s, sc, col, b in lines]).arrange(DOWN, aligned_edge=LEFT, buff=0.16)
        return g.move_to([x, y, 0]).align_to([left, 0, 0], LEFT)

    def construct(self):
        self.marks = {}
        title = Text("La red multicapa", font=FONT, weight=BOLD, color=INK).scale(0.9)
        sub = Text("varias neuronas juntas doblan el espacio", font=FONT, color=MUTED).scale(0.46)
        VGroup(title, sub).arrange(DOWN, buff=0.35)
        self.mark("title"); self.play(Write(title), run_time=1.4)
        self.mark("subtitle"); self.play(FadeIn(sub, shift=0.2 * UP)); self.wait(1.5)
        self.mark("title_out"); self.play(FadeOut(title), FadeOut(sub))

        # ---------- 1. una recta no basta ----------
        cap = self.swap(None, self.caption("Una recta no basta", "30 viajes del tema 1: carga y consumo · 12 no cuadran"))
        Pb = Panel(0, 25, 18, 42, (-6.0, 1.4, -2.9, 1.8), xt=(0, 5, 10, 15, 20, 25), xf=lambda v: f"{v} t", xl="carga",
                   yt=(20, 25, 30, 35, 40), yl="consumo (L/100 km)")
        dots = VGroup(*[Dot(Pb.P(c, q), radius=0.085, color=POINT if y else SAMPLE) for c, q, y in zip(CC, CQ, CY)])
        self.mark("dots"); self.play(FadeIn(Pb.g), LaggedStart(*[FadeIn(d, scale=1.6) for d in dots], lag_ratio=0.03), run_time=1.4)
        zA = Pb.zona(27.5, 0.55, True); lA = Pb.recta(27.5, 0.55, MEAN)
        labA = T("A", 0.5, MEAN, BOLD).next_to(Pb.P(23, 27.5 + 0.55 * 23), UP, buff=0.12)
        c1 = self.card([("una neurona: una recta", 0.42, INK, True), ("avisa solo por encima", 0.36, MUTED, False),
                        ("6 errores", 0.62, WARN, True), ("los 6 de abajo se escapan", 0.36, MUTED, False)])
        self.mark("line"); self.play(FadeIn(zA), Create(lA), FadeIn(labA), run_time=1.0)
        bad = [i for i in range(30) if CY[i] and CQ[i] < 24 + 0.55 * CC[i]]
        rings = VGroup(*[Circle(radius=0.16, stroke_color=WARN, stroke_width=4).move_to(Pb.P(CC[i], CQ[i])) for i in bad])
        self.play(FadeIn(c1), Create(rings), run_time=0.9); self.wait(2.4)

        # ---------- 2. dos neuronas, y la salida junta los votos ----------
        zB = Pb.zona(21.5, 0.55, False); lB = Pb.recta(21.5, 0.55, GOOD)
        labB = T("B", 0.5, GOOD, BOLD).next_to(Pb.P(23, 21.5 + 0.55 * 23), DOWN, buff=0.12)
        c2 = self.card([("neurona A: gasta de más", 0.4, MEAN, True), ("neurona B: gasta de menos", 0.4, GOOD, True),
                        ("cada una traza su recta", 0.36, MUTED, False)])
        cap = self.swap(cap, self.caption("Dos neuronas, dos rectas", "cada neurona oculta vota de un lado de su recta"),
                        extra=(FadeOut(c1), FadeOut(rings), FadeOut(zA), Transform(lA, Pb.recta(26.5, 0.55, MEAN)),
                               labA.animate.next_to(Pb.P(23, 26.5 + 0.55 * 23), UP, buff=0.12)))
        self.mark("lineB"); self.play(Create(lB), FadeIn(labB), FadeIn(c2), run_time=1.0); self.wait(2.0)
        c3 = self.card([("salida: basta un voto", 0.42, POINT, True), ("avisa si A o B votan", 0.36, MUTED, False),
                        ("0 errores", 0.62, GOOD, True), ("3 neuronas, 9 números", 0.36, MUTED, False)])
        cap2 = self.caption("Una neurona de salida junta los votos", "avisa si vota A o vota B")
        self.mark("votes")
        self.play(FadeOut(cap, shift=0.2 * UP), FadeIn(cap2, shift=0.2 * DOWN), FadeIn(Pb.zona(26.5, 0.55, True)),
                  FadeIn(Pb.zona(21.5, 0.55, False)), FadeOut(c2), FadeIn(c3), run_time=1.0)
        cap = cap2; self.wait(3.0)
        self.clear_but(cap)

        # ---------- 3. el espacio oculto ----------
        cap = self.swap(cap, self.caption("La capa oculta dobla el plano", "cada viaje, colocado según lo que votan A y B"))
        Pl = Panel(0, 25, 18, 42, (-6.0, -1.0, -2.9, 1.8), xt=(0, 10, 20), xf=lambda v: f"{v} t", xl="carga",
                   yt=(20, 30, 40), yl="consumo")
        Ph = Panel(-0.08, 1.08, -0.08, 1.08, (1.6, 6.2, -2.9, 1.8), xt=(0, 1), xf=lambda v: f"{int(v * 100)} %", xl="voto de A",
                   yt=(0, 1), yf=lambda v: f"{int(v * 100)} %", yl="voto de B")
        d1 = VGroup(*[Dot(Pl.P(c, q), radius=0.075, color=POINT if y else SAMPLE) for c, q, y in zip(CC, CQ, CY)])
        lines = VGroup(Pl.recta(26.5, 0.55, MEAN, 4), Pl.recta(21.5, 0.55, GOOD, 4))
        self.play(FadeIn(Pl.g), FadeIn(d1), Create(lines), FadeIn(Ph.g), run_time=1.0)
        mov = d1.copy()
        tgt = VGroup(*[Dot(Ph.P(*votos(c, q)), radius=0.075, color=POINT if y else SAMPLE) for c, q, y in zip(CC, CQ, CY)])
        arr = Arrow([-0.85, -0.5, 0], [1.45, -0.5, 0], color=MUTED, stroke_width=4, buff=0)
        self.mark("fold"); self.play(GrowArrow(arr), run_time=0.5)
        self.play(Transform(mov, tgt), run_time=2.4); self.wait(0.8)
        out = DashedLine(Ph.P(-0.08, 0.58), Ph.P(0.58, -0.08), color=POINT, stroke_width=5, dash_length=0.15)
        t1 = VGroup(T("en el espacio oculto,", 0.36, POINT, BOLD), T("una recta basta", 0.36, POINT, BOLD)).arrange(DOWN, aligned_edge=LEFT, buff=0.1).move_to(Ph.P(0.55, 0.6)).align_to(Ph.P(0.3, 0), LEFT)
        self.mark("outline"); self.play(Create(out), FadeIn(t1), run_time=1.0); self.wait(3.0)
        self.clear_but(cap)

        # ---------- 4. una capa es una matriz ----------
        cap = self.swap(cap, self.caption("Una capa es una tabla de pesos", "una fila por neurona: multiplicar la tabla por el viaje"))
        m1 = VGroup(T("neurona A:  −1,1 × carga + 2 × consumo − 53", 0.5, MEAN, BOLD),
                    T("neurona B:   1,1 × carga − 2 × consumo + 43", 0.5, GOOD, BOLD),
                    T("cada puntuación pasa por la sigmoide: votos h_A y h_B", 0.38, MUTED)).arrange(DOWN, aligned_edge=LEFT, buff=0.22)
        m2 = T("salida:  10 × h_A + 10 × h_B − 5", 0.5, POINT, BOLD)
        m3 = T("2 × (2 + 1) + 1 × (2 + 1) = 9 números ajustables", 0.42, GOOD, BOLD)
        VGroup(m1, m2, m3).arrange(DOWN, buff=0.55).move_to([0, -0.7, 0])
        self.mark("matrix"); self.play(Write(m1), run_time=1.4); self.play(FadeIn(m2, shift=0.1 * UP), run_time=0.7)
        self.play(FadeIn(m3), run_time=0.6); self.wait(2.6)
        self.clear_but(cap)

        # ---------- 5. más neuronas, más lados ----------
        cap = self.swap(cap, self.caption("Más neuronas, más lados", "zona normal del motor: revoluciones y temperatura"))
        Pm = Panel(600, 2200, 72, 106, (-6.0, 1.4, -2.9, 1.8), xt=(600, 1400, 2200), xl="revoluciones (rpm)",
                   yt=(75, 90, 105), yf=lambda v: f"{v} °C", yl="temperatura")
        fondo = Polygon(Pm.P(600, 72), Pm.P(2200, 72), Pm.P(2200, 106), Pm.P(600, 106), stroke_width=0, fill_color=POINT, fill_opacity=0.16)

        def zona(m):
            N = D["an"][str(m)]
            W, b, v, c = np.array(N["W"]), np.array(N["b"]), np.array(N["v"]), N["c"]
            p = lambda u, w: sig(c + v @ sig(W @ np.array([u, w]) + b))
            pts = []
            for th in np.linspace(0, 2 * np.pi, 241)[:-1]:
                du, dw = math.cos(th), math.sin(th)
                rmax = min(abs((2200 - 1400) / 350 / du) if abs(du) > 1e-9 else 1e9, abs((106 - 89) / 7 / dw) if abs(dw) > 1e-9 else 1e9,
                           abs((1400 - 600) / 350 / du) if abs(du) > 1e-9 else 1e9, abs((89 - 72) / 7 / dw) if abs(dw) > 1e-9 else 1e9)
                rmax = min(rmax, ((2200 - 1400) / 350 if du > 0 else (1400 - 600) / 350) / max(abs(du), 1e-9),
                           ((106 - 89) / 7 if dw > 0 else (89 - 72) / 7) / max(abs(dw), 1e-9))
                r = rmax
                for k in range(1, 401):
                    rr = rmax * k / 400
                    if p(rr * du, rr * dw) > 0.5:
                        r = rr; break
                pts.append(Pm.P(1400 + 350 * r * du, 89 + 7 * r * dw))
            return Polygon(*pts, stroke_color=SAMPLE, stroke_width=3, fill_color=BG, fill_opacity=1)
        el = ParametricFunction(lambda t: Pm.P(1400 + 350 * math.cos(t), 89 + 7 * math.sin(t)), t_range=[0, 2 * np.pi], color=GOOD, stroke_width=3)
        el = DashedVMobject(el, num_dashes=40)
        md = VGroup(*[Dot(Pm.P(r, t), radius=0.055, color=POINT if y else SAMPLE) for r, t, y in zip(MR, MT, MY)])
        z = zona(1)
        self.play(FadeIn(Pm.g), FadeIn(fondo), FadeIn(z), FadeIn(md), run_time=1.0)
        self.add(el); self.bring_to_front(md)
        card = None
        for m in (1, 3, 6):
            N = D["an"][str(m)]
            nc = self.card([(f"{m} {'neurona' if m == 1 else 'neuronas'}", 0.5, INK, True), (f"{N['e']} errores de 120", 0.4, WARN if N['e'] else GOOD, True),
                            (f"acierto en nuevas: {N['a'] * 100:.2f} %".replace(".", ","), 0.36, MUTED, False), (f"{4 * m + 1} números", 0.36, MUTED, False)])
            self.mark("width")
            if card is None:
                self.play(FadeIn(nc), run_time=0.6)
            else:
                self.play(Transform(z, zona(m)), FadeOut(card), FadeIn(nc), run_time=1.2)
            self.bring_to_front(el, md); card = nc
            self.wait(2.0)
        self.clear_but(cap)

        # ---------- 6. más capas, más pliegues ----------
        cap = self.swap(cap, self.caption("Más capas, más pliegues", "cada capa de 2 neuronas dobla otra vez lo ya doblado"))
        Pd = Panel(0, 1, 0, 1, (-6.0, 1.4, -2.9, 1.8), xt=(0, 0.5, 1), xf=lambda v: f"{v:g}".replace(".", ","), xl="número que entra",
                   yt=(0, 1), yl="número que sale")

        def sierra(L):
            n = 2 ** L
            return VMobject(stroke_color=MEAN, stroke_width=5).set_points_as_corners([Pd.P(k / n, k % 2) for k in range(n + 1)])
        self.play(FadeIn(Pd.g), run_time=0.6)
        cur = None; card = None
        for L in (1, 2, 3, 4):
            s = sierra(L)
            nc = self.card([(f"{L} {'capa' if L == 1 else 'capas'}: {2 ** L} trozos", 0.46, INK, True), (f"{6 * L + 1} números", 0.42, GOOD, True),
                            ("una sola capa necesitaría", 0.34, MUTED, False), (f"{2 ** L - 1} neuronas y {3 * 2 ** L - 2} números", 0.34, WARN if L > 2 else MUTED, True)])
            self.mark("depth")
            if cur is None:
                self.play(Create(s), FadeIn(nc), run_time=1.0); cur = s
            else:
                self.play(Transform(cur, s), FadeOut(card), FadeIn(nc), run_time=1.0)
            card = nc
            self.wait(1.9)
        self.clear_but(cap)

        # ---------- Cierre ----------
        self.mark("legend")
        self.play(FadeOut(cap), run_time=0.6)
        e1 = Text("Varias neuronas doblan el plano", font=FONT, weight=BOLD, color=INK).scale(0.7)
        e2 = VGroup(T("cada neurona oculta traza una recta y vota", 0.42, MUTED),
                    T("la de salida junta los votos con otra recta", 0.42, MUTED),
                    T("en el espacio oculto, una recta basta", 0.42, MUTED),
                    T("más neuronas, más lados; más capas, más pliegues", 0.42, MUTED)).arrange(DOWN, buff=0.18)
        VGroup(e1, e2).arrange(DOWN, buff=0.45)
        self.mark("end_text"); self.play(Write(e1), run_time=1.2); self.play(FadeIn(e2, shift=0.2 * UP, lag_ratio=0.15), run_time=1.0)
        self.wait(3.3)
        self.mark("fade"); self.play(FadeOut(VGroup(e1, e2)), run_time=1)
        self.marks["total"] = [round(self.renderer.time, 3)]
        json.dump(self.marks, open("marks.json", "w"), indent=1)
