import json
import math
import numpy as np
from manim import *
from scipy import stats

# Paleta oscura de la serie
BG = "#0E131B"; INK = "#E4EAF3"; MUTED = "#9BA7B9"; LINE = "#2B3545"
MEAN = "#6EA6FF"; SAMPLE = "#AE9FF3"; POINT = "#FF9A55"; GOOD = "#4CC38A"; WARN = "#F08A5D"
FONT = "Inter"
config.background_color = BG
D = json.load(open("../mt/m12/embed.json"))


def T(s, sc=0.45, col=INK, w=None):
    return Text(s, font=FONT, color=col, weight=w or NORMAL).scale(sc)


def fmt(v, d=1):
    s = f"{abs(v):.{d}f}".replace(".", ",")
    return ("−" if v < 0 and round(abs(v), d) != 0 else "") + s


class Axis:
    """Eje horizontal con marcas: mapea un valor a una x de la escena; y libre."""

    def __init__(self, x0, x1, ticks, label, left=-6.0, right=6.0, y=-3.0, fmt_t=None):
        self.x0, self.x1, self.L, self.R, self.y = x0, x1, left, right, y
        g = VGroup(Line([left, y, 0], [right, y, 0], stroke_color=MUTED, stroke_width=2))
        for v in ticks:
            g.add(Line(self.P(v, y), self.P(v, y + 0.1), stroke_color=MUTED, stroke_width=2))
            g.add(T(fmt_t(v) if fmt_t else str(v), 0.3, MUTED).next_to(self.P(v, y), DOWN, buff=0.12))
        g.add(T(label, 0.32, INK).next_to([right, y, 0], DOWN, buff=0.48).align_to([right, 0, 0], RIGHT))
        self.g = g

    def X(self, v):
        return self.L + (v - self.x0) / (self.x1 - self.x0) * (self.R - self.L)

    def P(self, v, y):
        return np.array([self.X(v), y, 0])


def ibar(A, lo, hi, y, col, w=7, m=None):
    g = VGroup(Line(A.P(lo, y), A.P(hi, y), stroke_color=col, stroke_width=w),
               Line(A.P(lo, y - 0.13), A.P(lo, y + 0.13), stroke_color=col, stroke_width=w * 0.6),
               Line(A.P(hi, y - 0.13), A.P(hi, y + 0.13), stroke_color=col, stroke_width=w * 0.6))
    if m is not None:
        g.add(Circle(0.09, color=col, stroke_width=4, fill_color=BG, fill_opacity=1).move_to(A.P(m, y)))
    return g


def tint(x):
    n = len(x); m = float(np.mean(x)); h = stats.t.ppf(0.975, n - 1) * np.std(x, ddof=1) / math.sqrt(n)
    return m, m - h, m + h


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
        title = Text("Intervalos de confianza y bootstrap", font=FONT, weight=BOLD, color=INK).scale(0.9)
        sub = Text("cuánto fiarse de una cifra: con fórmula y remuestreando", font=FONT, color=MUTED).scale(0.48)
        VGroup(title, sub).arrange(DOWN, buff=0.35)
        self.mark("title"); self.play(Write(title), run_time=1.3)
        self.mark("subtitle"); self.play(FadeIn(sub, shift=0.2 * UP)); self.wait(1.8)
        self.mark("title_out"); self.play(FadeOut(title), FadeOut(sub))

        TR = np.array(D["tray"]); rj = np.random.default_rng(3)

        # ---------- 1. una cifra sola no basta ----------
        cap = self.swap(None, self.caption("Una cifra sola no dice cuánto fiarse", "consumo de los trayectos de un camión (L/100 km)"))
        A = Axis(20, 46, [20, 24, 28, 32, 36, 40, 44], "consumo (L/100 km)", left=-6.0, right=6.0, y=-3.0)
        self.play(FadeIn(A.g), run_time=0.6)
        d10 = VGroup(*[Dot(A.P(v, 0.35 + 1.25 * rj.random()), radius=0.1, color=POINT) for v in TR[:10]])
        self.mark("dots"); self.play(LaggedStart(*[GrowFromCenter(p) for p in d10], lag_ratio=0.1), run_time=1.2)
        m, lo, hi = tint(TR[:10])
        lab10 = T(f"10 trayectos: media {fmt(m)}", 0.4, INK, BOLD).move_to(A.P(m, -0.15))
        self.mark("mean"); self.play(FadeIn(lab10, shift=0.1 * UP), run_time=0.6); self.wait(0.6)
        b10 = ibar(A, lo, hi, -0.75, MEAN, 8, m)
        t10 = T(f"± {fmt((hi - lo) / 2)}: de {fmt(lo)} a {fmt(hi)}", 0.38, MEAN, BOLD).next_to(b10, RIGHT, buff=0.3)
        self.mark("ci10"); self.play(Create(b10), FadeIn(t10), run_time=1.0); self.wait(1.6)
        d1k = VGroup(*[Dot(A.P(min(45.8, max(20.2, v)), 0.35 + 1.25 * rj.random()), radius=0.028, color=POINT, fill_opacity=0.5) for v in TR])
        m2, lo2, hi2 = tint(TR)
        lab1k = T(f"1 000 trayectos: media {fmt(m2)}", 0.4, INK, BOLD).move_to(A.P(m2, -1.4))
        b1k = ibar(A, lo2, hi2, -2.0, GOOD, 8, m2)
        t1k = T(f"± {fmt((hi2 - lo2) / 2, 2)}", 0.38, GOOD, BOLD).next_to(b1k, RIGHT, buff=0.3)
        self.mark("ci1000"); self.play(FadeOut(d10), FadeIn(d1k, lag_ratio=0.002), run_time=1.4)
        self.play(FadeIn(lab1k), Create(b1k), FadeIn(t1k), run_time=1.0); self.wait(2.4)
        self.clear_but(cap)

        # ---------- 2. estimación ± margen ----------
        cap = self.swap(cap, self.caption("Estimación ± margen", "margen = 1,96 × desviación típica / √n"))
        A = Axis(26, 39, [26, 28, 30, 32, 34, 36, 38], "consumo (L/100 km)", left=-6.2, right=2.2, y=-3.0)
        self.play(FadeIn(A.g), run_time=0.5)

        def bell(n):
            se = 4.5 / math.sqrt(n); h = 1.96 * se
            xs = np.linspace(26, 39, 260); top = 1.6
            pts = [A.P(x, -2.2 + (top + 2.2 - 0.3) * math.exp(-0.5 * ((x - 32.4) / se) ** 2)) for x in xs]
            curve = VMobject(stroke_color=MEAN, stroke_width=4).set_points_smoothly(pts)
            sh = [A.P(x, -2.2 + (top + 2.2 - 0.3) * math.exp(-0.5 * ((x - 32.4) / se) ** 2)) for x in np.linspace(max(26, 32.4 - h), min(39, 32.4 + h), 120)]
            area = Polygon(*sh, A.P(min(39, 32.4 + h), -2.2), A.P(max(26, 32.4 - h), -2.2), stroke_width=0, fill_color=MEAN, fill_opacity=0.22)
            bar = ibar(A, 32.4 - h, 32.4 + h, -2.6, MEAN, 7, 32.4)
            info = VGroup(T(f"n = {n:,}".replace(",", " "), 0.5, INK, BOLD), T("error estándar 4,5 / √" + f"{n:,}".replace(",", " ") + f" = {fmt(se, 2)}", 0.36, MUTED),
                          T(f"margen ± {fmt(h, 2)}", 0.48, MEAN, BOLD)).arrange(DOWN, aligned_edge=LEFT, buff=0.18).move_to([4.6, 0.2, 0])
            return VGroup(area, curve, bar, info)

        g = bell(10)
        self.mark("bell"); self.play(FadeIn(g), run_time=1.0); self.wait(1.6)
        for n in (100, 1000):
            g2 = bell(n); self.mark("bell2"); self.play(Transform(g, g2), run_time=1.2); self.wait(1.4)
        note = T("× 4 datos → margen ÷ 2", 0.4, POINT, BOLD).move_to([4.6, -1.6, 0])
        self.play(FadeIn(note, shift=0.2 * UP), run_time=0.6); self.wait(1.6)
        self.clear_but(cap)

        # ---------- 3. qué significa 95 % ----------
        cap = self.swap(cap, self.caption("¿Qué quiere decir «95 %»?", "de cada 100 intervalos construidos así, unos 95 atrapan el valor real"))
        A = Axis(25, 39, [26, 28, 30, 32, 34, 36, 38], "consumo (L/100 km)", left=-6.2, right=2.6, y=-3.2)
        r = np.random.default_rng(8); X = r.normal(32, 4.5, (40, 10))
        real = DashedLine(A.P(32, -3.2), A.P(32, 2.15), color=INK, stroke_width=3, dash_length=0.12)
        rl = T("valor real 32,0", 0.34, INK, BOLD).next_to(A.P(32, -3.2), DOWN, buff=0.5).shift(0 * LEFT)
        self.play(FadeIn(A.g), Create(real), run_time=0.7)
        bars = VGroup(); ok = 0
        for k in range(40):
            mm, l, h = tint(X[k]); hit = l <= 32 <= h; ok += hit; y = 2.0 - k * 0.128
            bars.add(Line(A.P(max(25, l), y), A.P(min(39, h), y), stroke_color=MEAN if hit else WARN, stroke_width=3 if hit else 6))
        self.mark("stack"); self.play(LaggedStart(*[Create(b) for b in bars], lag_ratio=0.06), run_time=3.0)
        cnt = VGroup(T(f"{ok} de 40", 0.62, MEAN, BOLD), T("atrapan el valor real", 0.38, INK), T(f"{40 - ok} fallan (en rojo)", 0.38, WARN, BOLD),
                     T("y no se sabe cuáles", 0.36, MUTED)).arrange(DOWN, aligned_edge=LEFT, buff=0.18).move_to([4.9, 0.2, 0])
        self.mark("count"); self.play(FadeIn(cnt, lag_ratio=0.2), run_time=1.0); self.wait(1.4)
        n2 = VGroup(T("el 95 % es del método,", 0.36, POINT, BOLD), T("no de tu intervalo", 0.36, POINT, BOLD)).arrange(DOWN, aligned_edge=LEFT, buff=0.1).next_to(cnt, DOWN, buff=0.45, aligned_edge=LEFT)
        self.play(FadeIn(n2), run_time=0.6); self.wait(2.0)
        self.clear_but(cap)

        # ---------- 4. la t ----------
        cap = self.swap(cap, self.caption("Pocos datos: la t de Student", "la desviación también sale de la muestra: colas más gordas, intervalo más ancho"))
        A = Axis(-4.5, 4.5, [-4, -2, 0, 2, 4], "distancia al valor real (errores estándar)", left=-6.2, right=2.2, y=-2.9, fmt_t=lambda v: fmt(v, 0))
        sy = lambda d: -2.9 + d / 0.42 * 4.6
        xs = np.linspace(-4.5, 4.5, 300)
        nrm = DashedVMobject(VMobject(stroke_color=MUTED, stroke_width=3).set_points_smoothly([A.P(x, sy(stats.norm.pdf(x))) for x in xs]), num_dashes=60)
        tc = VMobject(stroke_color=MEAN, stroke_width=4.5).set_points_smoothly([A.P(x, sy(stats.t.pdf(x, 4))) for x in xs])
        tails = VGroup(*[Polygon(*[A.P(x, sy(stats.t.pdf(x, 4))) for x in np.linspace(a, b, 60)], A.P(b, -2.9), A.P(a, -2.9), stroke_width=0, fill_color=WARN, fill_opacity=0.45) for a, b in ((-4.5, -1.96), (1.96, 4.5))])
        self.play(FadeIn(A.g), run_time=0.5)
        self.mark("tcurve"); self.play(Create(nrm), run_time=0.8); self.play(Create(tc), run_time=0.9)
        lz = VGroup(*[DashedLine(A.P(s * 1.96, -2.9), A.P(s * 1.96, 1.9), color=MUTED, stroke_width=2.5) for s in (-1, 1)])
        lt = VGroup(*[Line(A.P(s * 2.776, -2.9), A.P(s * 2.776, 1.9), color=MEAN, stroke_width=3.5) for s in (-1, 1)])
        info = VGroup(T("con 5 trayectos:", 0.4, INK, BOLD), T("usar 1,96 acierta el 88 %", 0.38, WARN, BOLD), T("t = 2,78 acierta el 95 %", 0.38, MEAN, BOLD),
                      T("con 30 trayectos, t = 2,05", 0.34, MUTED), T("con 1 000, t ≈ 1,96", 0.34, MUTED)).arrange(DOWN, aligned_edge=LEFT, buff=0.18).move_to([4.7, 0.0, 0])
        self.play(FadeIn(tails), Create(lz), FadeIn(info[:2]), run_time=1.0); self.wait(1.0)
        self.play(Create(lt), FadeIn(info[2]), run_time=0.8); self.wait(0.8)
        self.play(FadeIn(info[3:]), run_time=0.6); self.wait(2.0)
        self.clear_but(cap)

        # ---------- 5. bootstrap ----------
        cap = self.swap(cap, self.caption("Bootstrap: remuestrear tus propios datos", "12 viajes de reparto: sacar 12 con reemplazo, mil veces, y mirar cuánto baila la mediana"))
        E = np.array(D["eur12"]); toks = VGroup()
        for i, e in enumerate(E):
            box = RoundedRectangle(corner_radius=0.12, width=0.92, height=0.62, stroke_color=WARN if i == 11 else LINE, stroke_width=2.5)
            toks.add(VGroup(box, T(f"{int(e)} €", 0.34, INK, BOLD).move_to(box)))
        toks.arrange(RIGHT, buff=0.12).move_to([0, 1.55, 0])
        self.mark("tokens"); self.play(LaggedStart(*[FadeIn(t, shift=0.1 * UP) for t in toks], lag_ratio=0.05), run_time=1.0)
        A = Axis(28, 76, [30, 40, 50, 60, 70], "mediana del remuestreo (€)", left=-6.0, right=6.0, y=-3.1)
        self.play(FadeIn(A.g), run_time=0.4)
        idx = np.random.default_rng(1).integers(0, 12, (1000, 12)); med = np.median(E[idx], 1)
        cnts = None; mlab = None
        for j in range(3):
            c = np.bincount(idx[j], minlength=12)
            new = VGroup(*[T(f"×{c[i]}", 0.32, POINT if c[i] > 1 else (MEAN if c[i] else MUTED), BOLD).next_to(toks[i], DOWN, buff=0.12) for i in range(12)])
            nl = T(f"remuestreo {j + 1}: mediana {fmt(med[j])} €", 0.4, INK, BOLD).move_to([0, 0.15, 0])
            anims = [a for i in range(12) for a in (toks[i][0].animate.set_stroke(opacity=0.3 if c[i] == 0 else 1), toks[i][1].animate.set_opacity(0.3 if c[i] == 0 else 1))]
            self.mark("resample")
            if cnts is None:
                self.play(*anims, FadeIn(new), FadeIn(nl), run_time=0.7)
            else:
                self.play(*anims, Transform(cnts, new), Transform(mlab, nl), run_time=0.7)
            if cnts is None: cnts, mlab = new, nl
            self.wait(0.7)
        nb = 48; hcount, edges = np.histogram(med, bins=nb, range=(28, 76)); hm = hcount.max()
        bars = VGroup(*[Rectangle(width=(A.X(edges[1]) - A.X(edges[0])) * 0.85, height=max(0.001, 2.3 * c / hm), stroke_width=0, fill_color=POINT, fill_opacity=0.8)
                        .move_to(A.P((edges[k] + edges[k + 1]) / 2, -3.1 + 1.15 * c / hm)) for k, c in enumerate(hcount) if c])
        self.mark("hist"); self.play(FadeOut(mlab), LaggedStart(*[GrowFromEdge(b, DOWN) for b in bars], lag_ratio=0.03), run_time=1.6)
        lo, hi = np.quantile(med, [0.025, 0.975])
        band = Rectangle(width=A.X(hi) - A.X(lo), height=2.5, stroke_color=MEAN, stroke_width=3, fill_color=MEAN, fill_opacity=0.12).move_to(A.P((lo + hi) / 2, -1.85))
        bl = VGroup(T(f"mediana real de los 12: 43 €", 0.38, INK, BOLD), T(f"95 % central: de {fmt(lo)} a {fmt(hi, 0)} €", 0.4, MEAN, BOLD), T("sin ninguna fórmula", 0.36, MUTED)).arrange(DOWN, aligned_edge=LEFT, buff=0.14).move_to([3.7, -1.2, 0])
        self.play(FadeIn(band), FadeIn(bl, lag_ratio=0.2), run_time=1.0); self.wait(2.6)
        self.clear_but(cap)

        # ---------- 6. dos depósitos ----------
        cap = self.swap(cap, self.caption("Dos depósitos: ¿la diferencia incluye el 0?", "Sur − Norte, consumo medio por trayecto (L/100 km)"))
        A = Axis(-4, 6, [-4, -2, 0, 2, 4, 6], "diferencia Sur − Norte (L/100 km)", left=-5.8, right=5.8, y=-2.8, fmt_t=lambda v: ("+" if v > 0 else "") + fmt(v, 0))
        zero = Line(A.P(0, -2.8), A.P(0, 1.9), color=INK, stroke_width=3)
        zl = T("0: sin diferencia", 0.34, INK).next_to(A.P(0, 1.9), UP, buff=0.1)
        self.play(FadeIn(A.g), Create(zero), FadeIn(zl), run_time=0.7)
        No, Su = np.array(D["dep"][0]), np.array(D["dep"][1])

        def diff(n):
            a, b = No[:n], Su[:n]; r = stats.ttest_ind(b, a, equal_var=False).confidence_interval(); return b.mean() - a.mean(), r.low, r.high
        d, l, h = diff(20)
        b1 = ibar(A, l, h, 0.4, MUTED, 9, d)
        l1 = T(f"20 trayectos por depósito: +{fmt(d, 1)} (de {fmt(l, 1)} a +{fmt(h, 1)})", 0.38, MUTED, BOLD).next_to(b1, UP, buff=0.3).add_background_rectangle(color=BG, opacity=1, buff=0.06)
        l1b = T("incluye el 0: no se puede decir quién gasta más", 0.36, INK).next_to(b1, DOWN, buff=0.3).add_background_rectangle(color=BG, opacity=1, buff=0.06)
        self.mark("diff1"); self.play(Create(b1), FadeIn(l1), FadeIn(l1b), run_time=1.0); self.wait(2.0)
        d, l, h = diff(200)
        b2 = ibar(A, l, h, -1.5, GOOD, 9, d)
        l2 = T(f"200 trayectos: +{fmt(d, 1)} (de +{fmt(l, 1)} a +{fmt(h, 1)})", 0.38, GOOD, BOLD).next_to(b2, UP, buff=0.3).add_background_rectangle(color=BG, opacity=1, buff=0.06)
        l2b = T("ya no incluye el 0: unos 2 150 € al año por camión", 0.36, INK).next_to(b2, DOWN, buff=0.3).add_background_rectangle(color=BG, opacity=1, buff=0.06)
        self.mark("diff2"); self.play(Create(b2), FadeIn(l2), FadeIn(l2b), run_time=1.0); self.wait(2.8)
        self.clear_but(cap)

        # ---------- Cierre ----------
        self.mark("legend")
        self.play(FadeOut(cap), run_time=0.6)
        e1 = Text("Una cifra sin margen no dice cuánto fiarse", font=FONT, weight=BOLD, color=INK).scale(0.68)
        e2 = VGroup(T("medias: media ± t · s / √n", 0.42, MUTED), T("porcentajes: Wilson, no Wald", 0.42, MUTED),
                    T("cualquier otra cifra: bootstrap", 0.42, MUTED), T("y remuestrea lo independiente: camiones, no trayectos", 0.42, MUTED)).arrange(DOWN, buff=0.18)
        VGroup(e1, e2).arrange(DOWN, buff=0.45)
        self.mark("end_text"); self.play(Write(e1), run_time=1.2); self.play(FadeIn(e2, shift=0.2 * UP, lag_ratio=0.15), run_time=1.0)
        self.wait(3.5)
        self.mark("fade"); self.play(FadeOut(VGroup(e1, e2)), run_time=1)
        self.marks["total"] = [round(self.renderer.time, 3)]
        json.dump(self.marks, open("marks.json", "w"), indent=1)
