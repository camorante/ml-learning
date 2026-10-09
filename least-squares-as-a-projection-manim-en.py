import json
import numpy as np
from manim import *

# Dark palette of the series
BG = "#0E131B"; INK = "#E4EAF3"; MUTED = "#9BA7B9"; LINE = "#2B3545"
MEAN = "#6EA6FF"; SAMPLE = "#AE9FF3"; POINT = "#FF9A55"; GOOD = "#4CC38A"; WARN = "#F08A5D"
FONT = "Inter"
config.background_color = BG

D = json.load(open("../../mt/m4/embed.json"))
X = np.array(D["x"]); Y = np.array(D["y"]); B = D["b"]


def T(s, sc=0.45, col=INK, w=None):
    return Text(s, font=FONT, color=col, weight=w or NORMAL).scale(sc)


def fit(x, y):
    A = np.c_[np.ones_like(x), x]
    return np.linalg.lstsq(A, y, rcond=None)[0]


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
        title = Text("Least squares as a projection", font=FONT, weight=BOLD, color=INK).scale(0.9)
        sub = Text("why the best line has a formula", font=FONT, color=MUTED).scale(0.5)
        VGroup(title, sub).arrange(DOWN, buff=0.35)
        self.mark("title"); self.play(Write(title), run_time=1.3)
        self.mark("subtitle"); self.play(FadeIn(sub, shift=0.2 * UP)); self.wait(1.3)
        self.mark("title_out"); self.play(FadeOut(title), FadeOut(sub))

        # ---------- 1. the line and its squares ----------
        cap = self.swap(None, self.caption("The line with the smallest sum of squares", "10 trips: fuel use (L/100 km) versus load (t)"))
        ax = Axes(x_range=[0, 25, 5], y_range=[22, 40, 3], x_length=7.6, y_length=4.6, tips=False,
                  axis_config={"color": MUTED, "stroke_width": 2, "include_ticks": False}).move_to([-2.2, -0.9, 0])
        h = ValueTracker(28.0); p = ValueTracker(0.0)
        dots = VGroup(*[Dot(ax.c2p(x, y), radius=0.08, color=POINT) for x, y in zip(X, Y)]).set_z_index(3)
        def squares():
            g = VGroup()
            for x, y in zip(X, Y):
                yh = h.get_value() + p.get_value() * x
                a, b = ax.c2p(x, y), ax.c2p(x, yh); s = abs(a[1] - b[1])
                sq = Square(max(s, 0.001), stroke_color=SAMPLE, stroke_width=1, stroke_opacity=0.6, fill_color=SAMPLE, fill_opacity=0.16)
                sq.move_to([a[0] + (s / 2 if x < 20 else -s / 2), (a[1] + b[1]) / 2, 0]); g.add(sq)
                g.add(Line(a, b, color=SAMPLE, stroke_width=3))
            return g
        sq = always_redraw(squares)
        ln = always_redraw(lambda: Line(ax.c2p(0, h.get_value()), ax.c2p(25, h.get_value() + 25 * p.get_value()), color=MEAN, stroke_width=5))
        def readout():
            r = Y - h.get_value() - p.get_value() * X; s = float(r @ r)
            v = VGroup(T("sum of squares", 0.42, MUTED), T(f"{s:.1f}", 1.0, SAMPLE if s > 12.4 else GOOD, BOLD)).arrange(DOWN, buff=0.2)
            return v.move_to([4.6, 0.2, 0])
        ro = always_redraw(readout)
        self.mark("pts"); self.play(Create(ax), FadeIn(dots, lag_ratio=0.1), run_time=1.2)
        self.play(FadeIn(sq), Create(ln), FadeIn(ro), run_time=0.8); self.wait(0.6)
        self.mark("move"); self.play(p.animate.set_value(0.4), run_time=1.6); self.play(h.animate.set_value(26.0), run_time=1.0)
        self.mark("best"); self.play(h.animate.set_value(B[0]), p.animate.set_value(B[1]), run_time=0.9)
        nt = VGroup(T("26.20 + 0.41 × load", 0.5, GOOD, BOLD), T("no line does better", 0.42, GOOD)).arrange(DOWN, buff=0.18).move_to([4.6, -1.3, 0])
        self.play(FadeIn(nt), run_time=0.5); self.wait(2.0)
        sq.clear_updaters(); ro.clear_updaters(); ln.clear_updaters()
        self.clear_but(cap)

        # ---------- 2. an odd data point ----------
        cap = self.swap(cap, self.caption("One odd data point pulls the line", "squares punish big errors very hard"))
        ax = Axes(x_range=[0, 25, 5], y_range=[22, 52, 6], x_length=7.6, y_length=4.6, tips=False,
                  axis_config={"color": MUTED, "stroke_width": 2, "include_ticks": False}).move_to([-2.2, -0.9, 0])
        e = ValueTracker(0.0)
        def ys():
            y = Y.copy(); y[-1] += e.get_value(); return y
        def lad(y):
            best = None
            for i in range(10):
                for j in range(i + 1, 10):
                    s = (y[j] - y[i]) / (X[j] - X[i]); a = y[i] - s * X[i]; c = np.abs(y - a - s * X).sum()
                    if best is None or c < best[0] - 1e-12: best = (c, a, s)
            return best[1], best[2]
        dots = always_redraw(lambda: VGroup(*[Dot(ax.c2p(x, y), radius=0.08, color=POINT) for x, y in zip(X, ys())]).set_z_index(3))
        ls = always_redraw(lambda: (lambda b: Line(ax.c2p(0, b[0]), ax.c2p(25, b[0] + 25 * b[1]), color=MEAN, stroke_width=5))(fit(X, ys())))
        la = always_redraw(lambda: (lambda b: DashedLine(ax.c2p(0, b[0]), ax.c2p(25, b[0] + 25 * b[1]), color=SAMPLE, stroke_width=4))(lad(ys())))
        leg = VGroup(T("squares", 0.45, MEAN, BOLD), T("absolute values", 0.45, SAMPLE, BOLD)).arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to([4.5, 0.6, 0])
        self.play(Create(ax), FadeIn(dots), Create(ls), Create(la), FadeIn(leg), run_time=1.0); self.wait(0.6)
        self.mark("outlier"); self.play(e.animate.set_value(15), run_time=2.2, rate_func=smooth)
        ring = Circle(0.25, color=WARN, stroke_width=4).move_to(ax.c2p(X[-1], Y[-1] + 15))
        nt = VGroup(T("an error of 15 liters", 0.42, WARN), T("counts as 225", 0.42, WARN, BOLD)).arrange(DOWN, aligned_edge=LEFT, buff=0.15).move_to([4.5, -1.2, 0])
        self.play(Create(ring), FadeIn(nt), run_time=0.6); self.wait(2.2)
        for m in (dots, ls, la): m.clear_updaters()
        self.clear_but(cap)

        # ---------- 3. the shadow ----------
        cap = self.swap(cap, self.caption("The data are an arrow; the prediction, its shadow", "3 trips = 3 numbers = an arrow in 3D"))
        O = np.array([-4.2, -2.4, 0]); u = np.array([5.6, 0.7, 0]); v = np.array([1.9, 1.15, 0])
        plane = Polygon(O - 0.4 * u - 0.3 * v, O + 1.25 * u - 0.3 * v, O + 1.25 * u + 1.1 * v, O - 0.4 * u + 1.1 * v,
                        stroke_color=MEAN, stroke_width=2, fill_color=MEAN, fill_opacity=0.15)
        pl_lab = T("all predictions of \"height + slope × load\"", 0.36, MEAN).next_to(plane, DOWN, buff=0.15)
        H = O + 0.95 * u + 0.55 * v; Yv = H + np.array([0, 2.6, 0])
        ya = Arrow(O, Yv, buff=0, color=POINT, stroke_width=7, max_tip_length_to_length_ratio=0.07)
        ha = Arrow(O, H, buff=0, color=GOOD, stroke_width=7, max_tip_length_to_length_ratio=0.08)
        rr = DashedLine(H, Yv, color=SAMPLE, stroke_width=5)
        ra = VMobject(color=INK, stroke_width=2).set_points_as_corners([H + np.array([0, 0.3, 0]), H + np.array([0, 0.3, 0]) - 0.3 * u / np.linalg.norm(u), H - 0.3 * u / np.linalg.norm(u)])
        self.mark("plane"); self.play(DrawBorderThenFill(plane), FadeIn(pl_lab), run_time=1.0)
        self.mark("arrow"); self.play(GrowArrow(ya), FadeIn(T("data (26, 33, 34)", 0.45, POINT, BOLD).next_to(Yv, LEFT, buff=0.2)), run_time=0.9); self.wait(0.5)
        self.mark("shadow"); self.play(Create(rr), run_time=0.7); self.play(GrowArrow(ha), Create(ra), run_time=0.8)
        self.play(FadeIn(T("shadow (27, 31, 35)", 0.45, GOOD, BOLD).next_to(H, RIGHT, buff=0.25).shift(0.2 * DOWN)), run_time=0.5)
        info = VGroup(T("length² of the error =", 0.45, MUTED), T("sum of squares = 6", 0.5, SAMPLE, BOLD)).arrange(DOWN, aligned_edge=LEFT, buff=0.15).move_to([4.75, 1.55, 0])
        self.play(FadeIn(info), run_time=0.6); self.wait(2.2)
        self.clear_but(cap)

        # ---------- 4. normal equations ----------
        cap = self.swap(cap, self.caption("The error is perpendicular to each column", "dot product 0  →  normal equations"))
        rows = VGroup(
            VGroup(T("errors", 0.5, MUTED), T("(−1, 2, −1)", 0.6, SAMPLE, BOLD)).arrange(RIGHT, buff=0.4),
            VGroup(T("· column of ones", 0.5, MUTED), T("−1 + 2 − 1 = 0", 0.6, GOOD, BOLD)).arrange(RIGHT, buff=0.4),
            VGroup(T("· load (4, 12, 20)", 0.5, MUTED), T("−4 + 24 − 20 = 0", 0.6, GOOD, BOLD)).arrange(RIGHT, buff=0.4),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.55).move_to([0, 0.5, 0])
        self.mark("eq1"); self.play(FadeIn(rows[0], shift=0.2 * RIGHT), run_time=0.6); self.wait(0.3)
        self.mark("eq2"); self.play(FadeIn(rows[1], shift=0.2 * RIGHT), run_time=0.6); self.wait(0.3)
        self.mark("eq3"); self.play(FadeIn(rows[2], shift=0.2 * RIGHT), run_time=0.6); self.wait(0.5)
        ne = T("Xᵀ X β = Xᵀ y", 0.9, INK, BOLD).move_to([0, -1.9, 0])
        ns = T("two equations, two unknowns: height 25, slope 0.5", 0.42, MUTED).next_to(ne, DOWN, buff=0.3)
        self.mark("normal"); self.play(Write(ne), run_time=1.0); self.play(FadeIn(ns), run_time=0.5); self.wait(2.2)
        self.clear_but(cap)

        # ---------- 5. more columns ----------
        cap = self.swap(cap, self.caption("More columns: training always improves", "pure noise columns, 30 trips"))
        tr = [D["tr0"]] + D["tr"][:26]; te = [D["te0"]] + D["te"][:26]
        ax = Axes(x_range=[1, 27, 5], y_range=[0, 4.5, 1], x_length=8.4, y_length=4.2, tips=False,
                  axis_config={"color": MUTED, "stroke_width": 2, "include_ticks": False}).move_to([-1.6, -0.9, 0])
        c_tr = ax.plot_line_graph(list(range(1, 28)), tr, line_color=MEAN, add_vertex_dots=False, stroke_width=5)
        c_te = ax.plot_line_graph(list(range(1, 28)), te, line_color=POINT, add_vertex_dots=False, stroke_width=5)
        xl = T("model columns →", 0.38, MUTED).next_to(ax, DOWN, buff=0.15)
        l1 = T("training", 0.45, MEAN, BOLD).move_to([4.6, -1.9, 0]); l2 = T("new trips", 0.45, POINT, BOLD).move_to([4.6, 0.9, 0])
        self.play(Create(ax), FadeIn(xl), run_time=0.6)
        self.mark("curves"); self.play(Create(c_tr), Create(c_te), run_time=2.4); self.play(FadeIn(l1), FadeIn(l2), run_time=0.5)
        nt = T("set data aside before you look", 0.45, GOOD, BOLD).move_to([0.4, 1.3, 0])
        self.play(FadeIn(nt), run_time=0.5); self.wait(2.0)
        self.clear_but(cap)

        # ---------- 6. twin columns ----------
        cap = self.swap(cap, self.caption("Twin columns: wild coefficients", "load and total weight when all trucks weigh the same empty"))
        ax = Axes(x_range=[-2, 2, 1], y_range=[-2, 2, 1], x_length=4.9, y_length=4.9, tips=False,
                  axis_config={"color": MUTED, "stroke_width": 2, "include_ticks": False}).move_to([-3.0, -0.9, 0])
        wk = [(1.23, -0.75), (1.40, -0.84), (0.20, 0.29), (-1.04, 1.58), (0.47, -0.03), (0.48, 0.06), (1.20, -0.65), (1.67, -1.17)]
        rd = [(0.27, 0.20), (0.36, 0.21), (0.23, 0.26), (0.19, 0.35), (0.22, 0.22), (0.28, 0.26), (0.33, 0.22), (0.29, 0.22)]
        dl = DashedLine(ax.c2p(-1.5, 2.0), ax.c2p(2.0, -1.5), color=MUTED, stroke_width=3)
        truth = Cross(scale_factor=0.14, stroke_color=GOOD, stroke_width=6).move_to(ax.c2p(0, 0.5))
        dd = VGroup(*[Dot(ax.c2p(a, b), radius=0.1, color=SAMPLE) for a, b in wk])
        xl = T("per tonne of load →", 0.36, MUTED).next_to(ax, DOWN, buff=0.12); yl = T("per tonne of weight ↑", 0.36, MUTED).next_to(ax, UP, buff=0.08).align_to(ax, LEFT)
        self.play(Create(ax), FadeIn(xl), FadeIn(yl), Create(dl), FadeIn(truth), run_time=0.8)
        self.mark("dots"); self.play(LaggedStart(*[FadeIn(d, scale=1.5) for d in dd], lag_ratio=0.15), run_time=1.6)
        info = VGroup(T("8 weeks of trips:", 0.45, MUTED), T("load from −1.04 to 1.67", 0.5, WARN, BOLD),
                      T("the sum always ≈ 0.5", 0.5, INK)).arrange(DOWN, aligned_edge=LEFT, buff=0.2).move_to([3.3, 0.4, 0])
        self.play(FadeIn(info), run_time=0.6); self.wait(1.0)
        rl = T("with ridge: stable", 0.5, MEAN, BOLD).next_to(info, DOWN, buff=0.5).align_to(info, LEFT)
        self.mark("ridge"); self.play(*[d.animate.move_to(ax.c2p(a, b)).set_color(MEAN) for d, (a, b) in zip(dd, rd)], FadeIn(rl), run_time=1.2); self.wait(2.0)
        self.clear_but(cap)

        # ---------- Closing ----------
        self.mark("legend")
        self.play(FadeOut(cap), run_time=0.6)
        e1 = Text("The best prediction is a shadow.", font=FONT, weight=BOLD, color=INK).scale(0.72)
        e2 = Text("The error sits at a right angle to everything the model can explain.", font=FONT, color=MUTED).scale(0.42)
        VGroup(e1, e2).arrange(DOWN, buff=0.4)
        self.mark("end_text"); self.play(Write(e1), run_time=1.4); self.play(FadeIn(e2, shift=0.2 * UP), run_time=0.8)
        self.wait(2.4)
        self.mark("fade"); self.play(FadeOut(VGroup(e1, e2)), run_time=1)
        self.marks["total"] = [round(self.renderer.time, 3)]
        json.dump(self.marks, open("marks.json", "w"), indent=1)
