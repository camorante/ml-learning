import json
import numpy as np
from numpy.polynomial import Polynomial
from manim import *

# Dark palette of the Visual ML series
BG = "#0E131B"
INK = "#E4EAF3"
MUTED = "#9BA7B9"
LINE = "#2B3545"
MEAN = "#6EA6FF"
SAMPLE = "#AE9FF3"
POINT = "#FF9A55"
GOOD = "#4CC38A"
FONT = "Inter"
config.background_color = BG

# Datos 1: casi lineales (viajes en taxi)
PA = np.array([(1, 2.1), (2, 2.6), (3, 3.9), (4, 4.1), (5, 5.2), (6, 5.0), (7, 6.4), (8, 6.9), (9, 8.1)])
A_BEST = np.polyfit(PA[:, 0], PA[:, 1], 1)  # [pendiente, intercepto]

# Datos 2: forma de U
truth = lambda x: 2.9 + 0.17 * (x - 4.6) ** 2
rng = np.random.default_rng(4)
BX = np.linspace(0.8, 9.0, 10)
BY = truth(BX) + rng.normal(0, 0.35, len(BX))
NX = np.array([1.25, 2.2, 3.05, 5.75, 7.55, 8.5])  # datos nuevos
NY = truth(NX) + rng.normal(0, 0.3, len(NX))
XS = np.linspace(0.55, 9.25, 300)


def sse(m, b, P):
    return float(np.sum((P[:, 1] - (m * P[:, 0] + b)) ** 2))


class Regresion(Scene):
    def caption(self, text, sub=None):
        t = Text(text, font=FONT, weight=SEMIBOLD, color=INK).scale(0.62)
        grp = VGroup(t)
        if sub:
            grp.add(Text(sub, font=FONT, color=MUTED).scale(0.42))
            grp.arrange(DOWN, buff=0.18)
        return grp.to_edge(UP, buff=0.4)

    def swap(self, old, new):
        self.mark("caption")
        if old is None:
            self.play(FadeIn(new, shift=0.2 * DOWN), run_time=0.8)
        else:
            self.play(FadeOut(old, shift=0.2 * UP), FadeIn(new, shift=0.2 * DOWN), run_time=0.8)
        return new

    def mark(self, name):
        self.marks.setdefault(name, []).append(round(self.renderer.time, 3))

    def construct(self):
        self.marks = {}
        # ---------- Título ----------
        title = Text("Linear and polynomial regression", font=FONT, weight=BOLD, color=INK).scale(1.0)
        sub = Text("finding the curve that best sums up the data", font=FONT, color=MUTED).scale(0.5)
        VGroup(title, sub).arrange(DOWN, buff=0.35)
        self.mark("title")
        self.play(Write(title), run_time=1.4)
        self.mark("subtitle")
        self.play(FadeIn(sub, shift=0.2 * UP))
        self.wait(1.2)
        self.mark("title_out")
        self.play(FadeOut(title), FadeOut(sub))

        ax = Axes(x_range=[0, 10, 1], y_range=[0, 10, 1], x_length=12, y_length=5.2,
                  axis_config={"color": LINE, "stroke_width": 2, "include_ticks": False}, tips=False).shift(0.55 * DOWN)
        self.mark("axes")
        self.play(Create(ax), run_time=1)

        def dot(x, y, hollow=False):
            d = Dot(ax.c2p(x, y), radius=0.11, color=POINT)
            if hollow:
                d = Circle(radius=0.1, color=SAMPLE, stroke_width=4).move_to(ax.c2p(x, y))
            else:
                d.set_stroke(BG, width=4, background=True)
            return d

        def curve(f, color=MEAN, width=5):
            ys = np.clip(f(XS), 0.15, 9.85)
            m = VMobject(stroke_color=color, stroke_width=width)
            m.set_points_smoothly([ax.c2p(x, y) for x, y in zip(XS, ys)])
            return m

        def residual_group(f, P, squares=False):
            g = VGroup()
            for x, y in P:
                a, b = ax.c2p(x, y), ax.c2p(x, float(f(x)))
                if squares:
                    s = abs(a[1] - b[1])
                    sq = Rectangle(width=s, height=s, stroke_color=SAMPLE, stroke_width=1.5,
                                   fill_color=SAMPLE, fill_opacity=0.18)
                    sq.move_to([a[0] + s / 2, (a[1] + b[1]) / 2, 0])
                    g.add(sq)
                g.add(Line(a, b, color=SAMPLE, stroke_width=4))
            return g

        # ---------- 1. Dos perillas ----------
        cap = self.swap(None, self.caption("A straight line has only two knobs", "slope and starting point"))
        dots_a = VGroup(*[dot(x, y) for x, y in PA])
        self.mark("points_a")
        self.play(LaggedStart(*[GrowFromCenter(d) for d in dots_a], lag_ratio=0.15), run_time=1.6)
        m, b = ValueTracker(0.05), ValueTracker(7.2)
        line = always_redraw(lambda: curve(lambda x: m.get_value() * x + b.get_value()))
        res = always_redraw(lambda: residual_group(lambda x: m.get_value() * x + b.get_value(), PA))
        err_lbl = Text("total error", font=FONT, color=MUTED).scale(0.42)
        err = always_redraw(lambda: Text(f"{sse(m.get_value(), b.get_value(), PA):.1f}", font=FONT, weight=SEMIBOLD,
                                         color=INK).scale(0.7).next_to(err_lbl, DOWN, buff=0.12))
        VGroup(err_lbl).move_to(ax.c2p(1.6, 9.2))
        self.mark("line_in")
        self.play(Create(line), FadeIn(res), FadeIn(err_lbl), FadeIn(err), run_time=1)
        self.bring_to_front(dots_a)
        self.mark("move")
        self.play(m.animate.set_value(1.25), b.animate.set_value(0.2), run_time=1.6, rate_func=smooth)
        self.mark("move")
        self.play(m.animate.set_value(0.55), b.animate.set_value(3.2), run_time=1.3, rate_func=smooth)
        self.mark("settle")
        self.play(m.animate.set_value(A_BEST[0]), b.animate.set_value(A_BEST[1]), run_time=1.4, rate_func=smooth)
        self.wait(0.4)

        # ---------- 2. Cuadrados ----------
        cap = self.swap(cap, self.caption("The best line: smallest squared errors",
                                          "the total area of the squares is as small as possible"))
        f_best = lambda x: A_BEST[0] * x + A_BEST[1]
        self.remove(res)
        res_sq = always_redraw(lambda: residual_group(lambda x: m.get_value() * x + b.get_value(), PA, squares=True))
        self.add(res_sq)
        self.bring_to_front(dots_a)
        self.mark("squares")
        self.play(m.animate.set_value(0.95), b.animate.set_value(0.0), run_time=1.3, rate_func=smooth)
        self.wait(0.3)
        self.mark("settle")
        self.play(m.animate.set_value(A_BEST[0]), b.animate.set_value(A_BEST[1]), run_time=1.3, rate_func=smooth)
        self.wait(0.6)
        self.remove(res_sq)
        res_static = residual_group(f_best, PA, squares=True)
        sq = VGroup()
        self.add(res_static)
        self.bring_to_front(dots_a)

        # ---------- 3. Datos curvos ----------
        cap = self.swap(cap, self.caption("If the data curves, the line fails",
                                          "the errors form a pattern"))
        lin_b = np.polyfit(BX, BY, 1)
        f_lin = lambda x: lin_b[0] * x + lin_b[1]
        dots_b = VGroup(*[dot(x, y) for x, y in zip(BX, BY)])
        line_static = curve(f_best)
        self.remove(line, err, err_lbl)
        self.add(line_static)
        self.mark("morph_data")
        self.play(FadeOut(dots_a), FadeOut(res_static), FadeOut(sq), run_time=0.6)
        self.play(LaggedStart(*[GrowFromCenter(d) for d in dots_b], lag_ratio=0.08),
                  Transform(line_static, curve(f_lin)), run_time=1.4)
        res_b = residual_group(f_lin, np.c_[BX, BY])
        self.mark("pattern")
        self.play(FadeIn(res_b), run_time=0.7)
        self.bring_to_front(dots_b)
        self.wait(1.2)

        cap = self.swap(cap, self.caption("The parabola: one more knob", "degree 2: it can bend"))
        p2 = Polynomial.fit(BX, BY, 2)
        self.mark("parabola")
        self.play(Transform(line_static, curve(p2)), Transform(res_b, residual_group(p2, np.c_[BX, BY])), run_time=1.5)
        self.bring_to_front(dots_b)
        self.wait(1.2)

        # ---------- 4. Más grados ----------
        cap = self.swap(cap, self.caption("Higher degree, more flexible…", "until it passes through every point"))
        self.play(FadeOut(res_b), run_time=0.4)
        for d in (4, 6, 9):
            pd = Polynomial.fit(BX, BY, d)
            self.mark("degree")
            self.play(Transform(line_static, curve(pd, POINT if d == 9 else MEAN)), run_time=1.0)
            self.bring_to_front(dots_b)
            self.wait(0.25)

        cap = self.swap(cap, self.caption("Memorizing is not learning", "on new data, the overly flexible curve fails"))
        p9 = Polynomial.fit(BX, BY, 9)
        new = VGroup(*[dot(x, y, hollow=True) for x, y in zip(NX, NY)])
        self.mark("new_points")
        self.play(LaggedStart(*[GrowFromCenter(d) for d in new], lag_ratio=0.15), run_time=1.0)
        res_n = residual_group(p9, np.c_[NX, NY])
        self.mark("wobble")
        self.play(FadeIn(res_n), run_time=0.6)
        self.wait(1.3)
        self.mark("back")
        self.play(Transform(line_static, curve(p2)), Transform(res_n, residual_group(p2, np.c_[NX, NY])), run_time=1.4)
        self.bring_to_front(dots_b, new)
        self.wait(0.8)

        # ---------- Cierre ----------
        self.mark("legend")
        self.play(FadeOut(cap), FadeOut(res_n), run_time=0.6)
        k1 = VGroup(Line(LEFT * 0.3, RIGHT * 0.3, color=MEAN, stroke_width=6),
                    Text("model", font=FONT, color=INK).scale(0.42)).arrange(RIGHT, buff=0.2)
        k2 = VGroup(Line(UP * 0.15, DOWN * 0.15, color=SAMPLE, stroke_width=5),
                    Text("errors", font=FONT, color=INK).scale(0.42)).arrange(RIGHT, buff=0.2)
        k3 = VGroup(Dot(radius=0.1, color=POINT), Text("data to learn from", font=FONT, color=INK).scale(0.42)).arrange(RIGHT, buff=0.2)
        k4 = VGroup(Circle(radius=0.09, color=SAMPLE, stroke_width=4), Text("new data", font=FONT, color=INK).scale(0.42)).arrange(RIGHT, buff=0.2)
        legend = VGroup(k1, k2, k3, k4).arrange(RIGHT, buff=0.6).to_edge(UP, buff=0.5)
        end = Text("The best curve is the simplest one that explains the data.", font=FONT, weight=SEMIBOLD,
                   color=INK).scale(0.5).to_edge(DOWN, buff=0.35)
        self.play(FadeIn(legend, shift=0.2 * DOWN), run_time=0.8)
        self.mark("end_text")
        self.play(Write(end), run_time=1.5)
        self.wait(2.3)
        self.mark("fade")
        self.play(*[FadeOut(mm) for mm in self.mobjects], run_time=1)
        self.marks["total"] = [round(self.renderer.time, 3)]
        with open("marks.json", "w") as fh:
            json.dump(self.marks, fh, indent=1)
