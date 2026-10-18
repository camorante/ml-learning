import json
import math
import numpy as np
from manim import *

# Paleta oscura de la serie
BG = "#0E131B"; INK = "#E4EAF3"; MUTED = "#9BA7B9"; LINE = "#2B3545"
MEAN = "#6EA6FF"; SAMPLE = "#AE9FF3"; POINT = "#FF9A55"; GOOD = "#4CC38A"; WARN = "#F08A5D"
FONT = "Inter"
config.background_color = BG
D = json.load(open("../mt/m10/embed.json"))


def T(s, sc=0.45, col=INK, w=None):
    return Text(s, font=FONT, color=col, weight=w or NORMAL).scale(sc)


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def fmt(v, d=2):
    s = f"{abs(v):.{d}f}".replace(".", ",")
    return ("−" if v < 0 and round(abs(v), d) != 0 else "") + s


class Chart:
    """Marco de un gráfico de dispersión: mapea (x, y) de datos a puntos de la escena."""

    def __init__(self, x0, x1, y0, y1, xt, yt, xl, yl, left=-5.6, right=5.6, bottom=-3.0, top=1.6):
        self.x0, self.x1, self.y0, self.y1 = x0, x1, y0, y1
        self.L, self.R, self.B, self.Tp = left, right, bottom, top
        g = VGroup()
        for v in yt:
            g.add(Line(self.P(x0, v), self.P(x1, v), stroke_color=LINE, stroke_width=1.5))
            g.add(T(str(v), 0.32, MUTED).next_to(self.P(x0, v), LEFT, buff=0.15))
        g.add(Line(self.P(x0, y0), self.P(x1, y0), stroke_color=MUTED, stroke_width=2))
        for v in xt:
            g.add(T(str(v), 0.32, MUTED).next_to(self.P(v, y0), DOWN, buff=0.15))
        g.add(T(xl, 0.34, INK).next_to(self.P(x1, y0), DOWN, buff=0.5).align_to(self.P(x1, y0), RIGHT))
        g.add(T(yl, 0.32, MUTED).next_to(self.P(x0, y1), UP, buff=0.12).align_to(self.P(x0, y1), LEFT))
        self.axes = g

    def P(self, x, y):
        return np.array([self.L + (x - self.x0) / (self.x1 - self.x0) * (self.R - self.L),
                         self.B + (y - self.y0) / (self.y1 - self.y0) * (self.Tp - self.B), 0])

    def dots(self, xs, ys, col=POINT, r=0.07):
        return VGroup(*[Dot(self.P(x, y), radius=r, color=col) for x, y in zip(xs, ys)])

    def fit(self, xs, ys, a, b, col=MEAN, w=6):
        sl, ic = np.polyfit(xs, ys, 1)
        return Line(self.P(a, ic + sl * a), self.P(b, ic + sl * b), stroke_color=col, stroke_width=w)


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
        title = Text("Covarianza y correlación", font=FONT, weight=BOLD, color=INK).scale(0.95)
        sub = Text("cuándo dos números se mueven juntos, y por qué eso no es una causa", font=FONT, color=MUTED).scale(0.48)
        VGroup(title, sub).arrange(DOWN, buff=0.35)
        self.mark("title"); self.play(Write(title), run_time=1.3)
        self.mark("subtitle"); self.play(FadeIn(sub, shift=0.2 * UP)); self.wait(1.8)
        self.mark("title_out"); self.play(FadeOut(title), FadeOut(sub))

        X, Y = np.array(D["ex8"][0]), np.array(D["ex8"][1]); mx, my = X.mean(), Y.mean()

        # ---------- 1. la nube y la cruz ----------
        cap = self.swap(None, self.caption("¿Se mueven juntos?", "carga media y consumo de 8 camiones"))
        C = Chart(0, 30, 24, 40, [0, 5, 10, 15, 20, 25, 30], [24, 28, 32, 36, 40], "carga media (toneladas)", "consumo (L/100 km)", right=2.4)
        pts = C.dots(X, Y, POINT, 0.11).set_z_index(3)
        self.play(FadeIn(C.axes), run_time=0.8)
        self.mark("dots"); self.play(LaggedStart(*[GrowFromCenter(p) for p in pts], lag_ratio=0.12), run_time=1.4); self.wait(0.8)
        vx = DashedLine(C.P(mx, 24), C.P(mx, 40), color=MUTED, stroke_width=3, dash_length=0.12)
        hy = DashedLine(C.P(0, my), C.P(30, my), color=MUTED, stroke_width=3, dash_length=0.12)
        ct = VGroup(T("la cruz: las medias", 0.42, INK, BOLD), T("15 t y 32 L/100 km", 0.4, MUTED)).arrange(DOWN, aligned_edge=LEFT, buff=0.14).move_to([4.6, 0.7, 0])
        self.mark("cross"); self.play(Create(vx), Create(hy), FadeIn(ct), run_time=1.0); self.wait(2.0)

        # ---------- 2. rectángulos ----------
        cap = self.swap(cap, self.caption("Rectángulos con signo", "desviación en carga × desviación en consumo"))
        rects = VGroup()
        for x, y in zip(X, Y):
            p = (x - mx) * (y - my); col = MEAN if p > 0 else SAMPLE
            a, b = C.P(min(x, mx), min(y, my)), C.P(max(x, mx), max(y, my))
            rects.add(Rectangle(width=b[0] - a[0], height=b[1] - a[1], stroke_color=col, stroke_width=2, fill_color=col, fill_opacity=0.22).move_to((a + b) / 2))
        info = VGroup(T("azules: a favor", 0.4, MEAN, BOLD), T("suman +132", 0.4, INK), T("violetas: en contra", 0.4, SAMPLE, BOLD), T("suman −2", 0.4, INK),
                      T("covarianza", 0.42, INK, BOLD), T("(132 − 2) / 7 = 18,6", 0.42, POINT, BOLD)).arrange(DOWN, aligned_edge=LEFT, buff=0.16).move_to([4.6, -0.6, 0])
        self.play(FadeOut(ct), run_time=0.4)
        self.mark("rects"); self.play(LaggedStart(*[GrowFromPoint(r, C.P(mx, my)) for r in rects], lag_ratio=0.12), run_time=1.8)
        self.play(FadeIn(info[:4], lag_ratio=0.2), run_time=1.0); self.wait(1.0)
        self.mark("cov"); self.play(FadeIn(info[4:], shift=0.2 * UP), run_time=0.8); self.wait(2.6)

        # ---------- 3. sin unidades ----------
        cap = self.swap(cap, self.caption("Sin unidades: la correlación r", "en kilos la covarianza sale 1 000 veces mayor; r no cambia"))
        kg = VGroup(T("en kilos:", 0.4, MUTED), T("covarianza 18 571", 0.42, INK, BOLD)).arrange(DOWN, aligned_edge=LEFT, buff=0.14)
        zz = VGroup(T("en z (desviaciones típicas):", 0.4, MUTED), T("covarianza = r = 0,92", 0.46, MEAN, BOLD), T("r² = 0,85", 0.42, INK))
        zz.arrange(DOWN, aligned_edge=LEFT, buff=0.14)
        VGroup(kg, zz).arrange(DOWN, aligned_edge=LEFT, buff=0.45).move_to([4.6, -0.4, 0])
        self.play(FadeOut(info), FadeOut(rects), run_time=0.6)
        self.mark("kg"); self.play(FadeIn(kg, shift=0.2 * UP), run_time=0.7); self.wait(1.5)
        fl = C.fit(X, Y, 0, 30)
        self.mark("r"); self.play(Create(fl), FadeIn(zz, shift=0.2 * UP), run_time=1.2); self.wait(2.6)
        self.clear_but(cap)
        rr = [-0.9, -0.5, 0.0, 0.5, 0.9]; rng = np.random.default_rng(5); a = rng.normal(size=60); e = rng.normal(size=60)
        a = (a - a.mean()) / a.std(); e = e - np.polyfit(a, e, 1)[0] * a; e = (e - e.mean()) / e.std()
        minis = VGroup()
        for r in rr:
            b = r * a + math.sqrt(1 - r * r) * e
            box = RoundedRectangle(corner_radius=0.1, width=2.2, height=2.2, stroke_color=LINE, stroke_width=2)
            ds = VGroup(*[Dot(box.get_center() + np.array([0.33 * u, 0.33 * v, 0]), radius=0.035, color=POINT) for u, v in zip(a, b) if abs(u) < 3 and abs(v) < 3])
            lb = T(f"r = {fmt(r, 1)}", 0.42, INK, BOLD).next_to(box, DOWN, buff=0.2)
            minis.add(VGroup(box, ds, lb))
        minis.arrange(RIGHT, buff=0.3).move_to([0, -0.3, 0])
        self.mark("scale"); self.play(LaggedStart(*[FadeIn(m, shift=0.2 * UP) for m in minis], lag_ratio=0.15), run_time=1.6)
        ends = VGroup(T("−1: una recta que baja", 0.38, SAMPLE), T("0: ninguna recta", 0.38, MUTED), T("1: una recta que sube", 0.38, MEAN)).arrange(RIGHT, buff=0.9).next_to(minis, DOWN, buff=0.45)
        self.play(FadeIn(ends), run_time=0.6); self.wait(2.2)
        self.clear_but(cap)

        # ---------- 4. la U ----------
        cap = self.swap(cap, self.caption("r solo ve líneas rectas", "velocidad media del trayecto y consumo: una U"))
        V, Cc = np.array(D["vel"][0]), np.array(D["vel"][1])
        C = Chart(5, 95, 26, 42, [10, 30, 50, 70, 90], [26, 30, 34, 38, 42], "velocidad media (km/h)", "consumo (L/100 km)", right=2.6)
        self.play(FadeIn(C.axes), run_time=0.6)
        self.mark("u"); self.play(FadeIn(C.dots(V, Cc, POINT, 0.06), lag_ratio=0.01), run_time=1.4)
        info = VGroup(T("relación fortísima,", 0.4, INK), T("pero en U", 0.4, INK), T("r = −0,01", 0.5, MEAN, BOLD), T("la mejor recta es plana", 0.38, MUTED)).arrange(DOWN, aligned_edge=LEFT, buff=0.16).move_to([4.7, -0.3, 0])
        self.mark("flat"); self.play(Create(C.fit(V, Cc, 10, 90)), FadeIn(info, lag_ratio=0.2), run_time=1.2); self.wait(3.0)
        self.clear_but(cap)

        # ---------- 5. un atípico ----------
        cap = self.swap(cap, self.caption("Un solo atípico fabrica una correlación", "20 camiones sin relación y un registro erróneo"))
        ax, ay = np.array(D["at"][0][0]), np.array(D["at"][0][1]); ox, oy = D["at0"][0]
        C = Chart(18, 80, 20, 60, [20, 30, 40, 50, 60, 70, 80], [20, 30, 40, 50, 60], "edad del conductor (años)", "consumo (L/100 km)", right=2.6)
        self.play(FadeIn(C.axes), FadeIn(C.dots(ax, ay, POINT, 0.09)), run_time=0.8)
        l0 = C.fit(ax, ay, 18, 80, MUTED, 4)
        i0 = VGroup(T(f"r = {fmt(corr(ax, ay))}", 0.46, INK, BOLD), T("nada", 0.4, MUTED)).arrange(DOWN, aligned_edge=LEFT, buff=0.14).move_to([4.7, 0.6, 0])
        self.play(Create(l0), FadeIn(i0), run_time=0.8); self.wait(0.8)
        AX, AY = np.r_[ax, ox], np.r_[ay, oy]
        od = Dot(C.P(ox, oy), radius=0.13, color=WARN); ring = Circle(0.24, color=WARN, stroke_width=3).move_to(od)
        from scipy.stats import spearmanr
        i1 = VGroup(T(f"con él: r = {fmt(corr(AX, AY))}", 0.46, MEAN, BOLD), T(f"Spearman = {fmt(spearmanr(AX, AY)[0])}", 0.42, POINT, BOLD), T("(r de los puestos)", 0.36, MUTED)).arrange(DOWN, aligned_edge=LEFT, buff=0.14).next_to(i0, DOWN, buff=0.4, aligned_edge=LEFT)
        self.mark("outlier"); self.play(GrowFromCenter(od), Create(ring), run_time=0.6)
        self.play(Transform(l0, C.fit(AX, AY, 18, 80, MEAN, 6)), FadeIn(i1[0]), run_time=1.0); self.wait(0.8)
        self.play(FadeIn(i1[1:]), run_time=0.6); self.wait(2.4)
        self.clear_but(cap)

        # ---------- 6. confusión ----------
        cap = self.swap(cap, self.caption("Correlación no es causalidad", "los camiones viejos van a ciudad, y en ciudad se gasta más"))
        E, K, Co = np.array(D["conf"][0]), np.array(D["conf"][1]), np.array(D["conf"][2])
        C = Chart(0, 16, 24, 44, [0, 4, 8, 12, 16], [24, 28, 32, 36, 40, 44], "edad del camión (años)", "consumo (L/100 km)", right=2.6)
        pts = C.dots(E, Co, POINT, 0.06)
        self.play(FadeIn(C.axes), FadeIn(pts), run_time=0.8)
        lt = C.fit(E, Co, 1, 15)
        i0 = VGroup(T("toda la flota:", 0.4, MUTED), T(f"r = {fmt(corr(E, Co))}", 0.48, MEAN, BOLD), T("¿renovar los viejos?", 0.4, INK)).arrange(DOWN, aligned_edge=LEFT, buff=0.14).move_to([4.7, 0.5, 0])
        self.mark("conf"); self.play(Create(lt), FadeIn(i0), run_time=1.0); self.wait(1.6)
        anims = [p.animate.set_color(POINT if k else SAMPLE) for p, k in zip(pts, K)]
        lc = C.fit(E[K == 1], Co[K == 1], E[K == 1].min(), 15, POINT, 5); la = C.fit(E[K == 0], Co[K == 0], 1, E[K == 0].max(), SAMPLE, 5)
        i1 = VGroup(T(f"ciudad: r = {fmt(corr(E[K == 1], Co[K == 1]))}", 0.42, POINT, BOLD), T(f"autovía: r = {fmt(corr(E[K == 0], Co[K == 0]))}", 0.42, SAMPLE, BOLD),
                    T("la ruta empuja a las dos:", 0.38, MUTED), T("variable de confusión", 0.42, INK, BOLD)).arrange(DOWN, aligned_edge=LEFT, buff=0.14).next_to(i0, DOWN, buff=0.4, aligned_edge=LEFT)
        self.mark("split"); self.play(*anims, FadeOut(lt), run_time=1.0)
        self.play(Create(lc), Create(la), FadeIn(i1, lag_ratio=0.2), run_time=1.2); self.wait(3.4)
        self.clear_but(cap)

        # ---------- Cierre ----------
        self.mark("legend")
        self.play(FadeOut(cap), run_time=0.6)
        e1 = Text("Dibuja la nube antes del número", font=FONT, weight=BOLD, color=INK).scale(0.72)
        e2 = VGroup(T("covarianza: el signo de «se mueven juntos», con unidades", 0.42, MUTED), T("r: la covarianza de las z, de −1 a 1, solo para rectas", 0.42, MUTED),
                    T("un atípico la fabrica; Spearman resiste", 0.42, MUTED), T("y nunca dice qué causa qué", 0.42, MUTED)).arrange(DOWN, buff=0.18)
        VGroup(e1, e2).arrange(DOWN, buff=0.45)
        self.mark("end_text"); self.play(Write(e1), run_time=1.2); self.play(FadeIn(e2, shift=0.2 * UP, lag_ratio=0.15), run_time=1.0)
        self.wait(3.5)
        self.mark("fade"); self.play(FadeOut(VGroup(e1, e2)), run_time=1)
        self.marks["total"] = [round(self.renderer.time, 3)]
        json.dump(self.marks, open("marks.json", "w"), indent=1)
