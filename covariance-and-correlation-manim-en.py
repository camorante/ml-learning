import json
import math
import numpy as np
from manim import *

# Dark palette of the series
BG = "#0E131B"; INK = "#E4EAF3"; MUTED = "#9BA7B9"; LINE = "#2B3545"
MEAN = "#6EA6FF"; SAMPLE = "#AE9FF3"; POINT = "#FF9A55"; GOOD = "#4CC38A"; WARN = "#F08A5D"
FONT = "Inter"
config.background_color = BG
D = json.load(open("../../mt/m10/embed.json"))


def T(s, sc=0.45, col=INK, w=None):
    return Text(s, font=FONT, color=col, weight=w or NORMAL).scale(sc)


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def fmt(v, d=2):
    s = f"{abs(v):.{d}f}"
    return ("−" if v < 0 and round(abs(v), d) != 0 else "") + s


class Chart:
    """Frame of a scatter chart: maps data (x, y) to scene points."""

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
        title = Text("Covariance and correlation", font=FONT, weight=BOLD, color=INK).scale(0.95)
        sub = Text("when two numbers move together, and why that is not a cause", font=FONT, color=MUTED).scale(0.48)
        VGroup(title, sub).arrange(DOWN, buff=0.35)
        self.mark("title"); self.play(Write(title), run_time=1.3)
        self.mark("subtitle"); self.play(FadeIn(sub, shift=0.2 * UP)); self.wait(1.8)
        self.mark("title_out"); self.play(FadeOut(title), FadeOut(sub))

        X, Y = np.array(D["ex8"][0]), np.array(D["ex8"][1]); mx, my = X.mean(), Y.mean()

        # ---------- 1. the cloud and the cross ----------
        cap = self.swap(None, self.caption("Do they move together?", "average load and consumption of 8 trucks"))
        C = Chart(0, 30, 24, 40, [0, 5, 10, 15, 20, 25, 30], [24, 28, 32, 36, 40], "average load (tonnes)", "consumption (L/100 km)", right=2.4)
        pts = C.dots(X, Y, POINT, 0.11).set_z_index(3)
        self.play(FadeIn(C.axes), run_time=0.8)
        self.mark("dots"); self.play(LaggedStart(*[GrowFromCenter(p) for p in pts], lag_ratio=0.12), run_time=1.4); self.wait(0.8)
        vx = DashedLine(C.P(mx, 24), C.P(mx, 40), color=MUTED, stroke_width=3, dash_length=0.12)
        hy = DashedLine(C.P(0, my), C.P(30, my), color=MUTED, stroke_width=3, dash_length=0.12)
        ct = VGroup(T("the cross: the means", 0.42, INK, BOLD), T("15 t and 32 L/100 km", 0.4, MUTED)).arrange(DOWN, aligned_edge=LEFT, buff=0.14).move_to([4.6, 0.7, 0])
        self.mark("cross"); self.play(Create(vx), Create(hy), FadeIn(ct), run_time=1.0); self.wait(2.0)

        # ---------- 2. rectangles ----------
        cap = self.swap(cap, self.caption("Rectangles with a sign", "load deviation × consumption deviation"))
        rects = VGroup()
        for x, y in zip(X, Y):
            p = (x - mx) * (y - my); col = MEAN if p > 0 else SAMPLE
            a, b = C.P(min(x, mx), min(y, my)), C.P(max(x, mx), max(y, my))
            rects.add(Rectangle(width=b[0] - a[0], height=b[1] - a[1], stroke_color=col, stroke_width=2, fill_color=col, fill_opacity=0.22).move_to((a + b) / 2))
        info = VGroup(T("blue: in favor", 0.4, MEAN, BOLD), T("add up to +132", 0.4, INK), T("purple: against", 0.4, SAMPLE, BOLD), T("add up to −2", 0.4, INK),
                      T("covariance", 0.42, INK, BOLD), T("(132 − 2) / 7 = 18.6", 0.42, POINT, BOLD)).arrange(DOWN, aligned_edge=LEFT, buff=0.16).move_to([4.6, -0.6, 0])
        self.play(FadeOut(ct), run_time=0.4)
        self.mark("rects"); self.play(LaggedStart(*[GrowFromPoint(r, C.P(mx, my)) for r in rects], lag_ratio=0.12), run_time=1.8)
        self.play(FadeIn(info[:4], lag_ratio=0.2), run_time=1.0); self.wait(1.0)
        self.mark("cov"); self.play(FadeIn(info[4:], shift=0.2 * UP), run_time=0.8); self.wait(2.6)

        # ---------- 3. no units ----------
        cap = self.swap(cap, self.caption("No units: the correlation r", "in kilograms the covariance is 1,000 times larger; r doesn't change"))
        kg = VGroup(T("in kilograms:", 0.4, MUTED), T("covariance 18,571", 0.42, INK, BOLD)).arrange(DOWN, aligned_edge=LEFT, buff=0.14)
        zz = VGroup(T("in z (standard deviations):", 0.4, MUTED), T("covariance = r = 0.92", 0.46, MEAN, BOLD), T("r² = 0.85", 0.42, INK))
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
        ends = VGroup(T("−1: a line going down", 0.38, SAMPLE), T("0: no line", 0.38, MUTED), T("1: a line going up", 0.38, MEAN)).arrange(RIGHT, buff=0.9).next_to(minis, DOWN, buff=0.45)
        self.play(FadeIn(ends), run_time=0.6); self.wait(2.2)
        self.clear_but(cap)

        # ---------- 4. the U ----------
        cap = self.swap(cap, self.caption("r only sees straight lines", "average trip speed and consumption: a U"))
        V, Cc = np.array(D["vel"][0]), np.array(D["vel"][1])
        C = Chart(5, 95, 26, 42, [10, 30, 50, 70, 90], [26, 30, 34, 38, 42], "average speed (km/h)", "consumption (L/100 km)", right=2.6)
        self.play(FadeIn(C.axes), run_time=0.6)
        self.mark("u"); self.play(FadeIn(C.dots(V, Cc, POINT, 0.06), lag_ratio=0.01), run_time=1.4)
        info = VGroup(T("a very strong relationship,", 0.4, INK), T("but U-shaped", 0.4, INK), T("r = −0.01", 0.5, MEAN, BOLD), T("the best line is flat", 0.38, MUTED)).arrange(DOWN, aligned_edge=LEFT, buff=0.16).move_to([4.7, -0.3, 0])
        self.mark("flat"); self.play(Create(C.fit(V, Cc, 10, 90)), FadeIn(info, lag_ratio=0.2), run_time=1.2); self.wait(3.0)
        self.clear_but(cap)

        # ---------- 5. an outlier ----------
        cap = self.swap(cap, self.caption("A single outlier creates a correlation", "20 trucks with no relationship and one bad record"))
        ax, ay = np.array(D["at"][0][0]), np.array(D["at"][0][1]); ox, oy = D["at0"][0]
        C = Chart(18, 80, 20, 60, [20, 30, 40, 50, 60, 70, 80], [20, 30, 40, 50, 60], "driver age (years)", "consumption (L/100 km)", right=2.6)
        self.play(FadeIn(C.axes), FadeIn(C.dots(ax, ay, POINT, 0.09)), run_time=0.8)
        l0 = C.fit(ax, ay, 18, 80, MUTED, 4)
        i0 = VGroup(T(f"r = {fmt(corr(ax, ay))}", 0.46, INK, BOLD), T("nothing", 0.4, MUTED)).arrange(DOWN, aligned_edge=LEFT, buff=0.14).move_to([4.7, 0.6, 0])
        self.play(Create(l0), FadeIn(i0), run_time=0.8); self.wait(0.8)
        AX, AY = np.r_[ax, ox], np.r_[ay, oy]
        od = Dot(C.P(ox, oy), radius=0.13, color=WARN); ring = Circle(0.24, color=WARN, stroke_width=3).move_to(od)
        from scipy.stats import spearmanr
        i1 = VGroup(T(f"with it: r = {fmt(corr(AX, AY))}", 0.46, MEAN, BOLD), T(f"Spearman = {fmt(spearmanr(AX, AY)[0])}", 0.42, POINT, BOLD), T("(r of the ranks)", 0.36, MUTED)).arrange(DOWN, aligned_edge=LEFT, buff=0.14).next_to(i0, DOWN, buff=0.4, aligned_edge=LEFT)
        self.mark("outlier"); self.play(GrowFromCenter(od), Create(ring), run_time=0.6)
        self.play(Transform(l0, C.fit(AX, AY, 18, 80, MEAN, 6)), FadeIn(i1[0]), run_time=1.0); self.wait(0.8)
        self.play(FadeIn(i1[1:]), run_time=0.6); self.wait(2.4)
        self.clear_but(cap)

        # ---------- 6. confounding ----------
        cap = self.swap(cap, self.caption("Correlation is not causation", "old trucks go to the city, and city driving burns more"))
        E, K, Co = np.array(D["conf"][0]), np.array(D["conf"][1]), np.array(D["conf"][2])
        C = Chart(0, 16, 24, 44, [0, 4, 8, 12, 16], [24, 28, 32, 36, 40, 44], "truck age (years)", "consumption (L/100 km)", right=2.6)
        pts = C.dots(E, Co, POINT, 0.06)
        self.play(FadeIn(C.axes), FadeIn(pts), run_time=0.8)
        lt = C.fit(E, Co, 1, 15)
        i0 = VGroup(T("whole fleet:", 0.4, MUTED), T(f"r = {fmt(corr(E, Co))}", 0.48, MEAN, BOLD), T("replace the old ones?", 0.4, INK)).arrange(DOWN, aligned_edge=LEFT, buff=0.14).move_to([4.7, 0.5, 0])
        self.mark("conf"); self.play(Create(lt), FadeIn(i0), run_time=1.0); self.wait(1.6)
        anims = [p.animate.set_color(POINT if k else SAMPLE) for p, k in zip(pts, K)]
        lc = C.fit(E[K == 1], Co[K == 1], E[K == 1].min(), 15, POINT, 5); la = C.fit(E[K == 0], Co[K == 0], 1, E[K == 0].max(), SAMPLE, 5)
        i1 = VGroup(T(f"city: r = {fmt(corr(E[K == 1], Co[K == 1]))}", 0.42, POINT, BOLD), T(f"highway: r = {fmt(corr(E[K == 0], Co[K == 0]))}", 0.42, SAMPLE, BOLD),
                    T("the route pushes both:", 0.38, MUTED), T("confounding variable", 0.42, INK, BOLD)).arrange(DOWN, aligned_edge=LEFT, buff=0.14).next_to(i0, DOWN, buff=0.4, aligned_edge=LEFT)
        self.mark("split"); self.play(*anims, FadeOut(lt), run_time=1.0)
        self.play(Create(lc), Create(la), FadeIn(i1, lag_ratio=0.2), run_time=1.2); self.wait(3.4)
        self.clear_but(cap)

        # ---------- Closing ----------
        self.mark("legend")
        self.play(FadeOut(cap), run_time=0.6)
        e1 = Text("Draw the cloud before the number", font=FONT, weight=BOLD, color=INK).scale(0.72)
        e2 = VGroup(T("covariance: the sign of \"they move together\", with units", 0.42, MUTED), T("r: the covariance of the z's, from −1 to 1, only for lines", 0.42, MUTED),
                    T("an outlier creates it; Spearman holds up", 0.42, MUTED), T("and it never says what causes what", 0.42, MUTED)).arrange(DOWN, buff=0.18)
        VGroup(e1, e2).arrange(DOWN, buff=0.45)
        self.mark("end_text"); self.play(Write(e1), run_time=1.2); self.play(FadeIn(e2, shift=0.2 * UP, lag_ratio=0.15), run_time=1.0)
        self.wait(3.5)
        self.mark("fade"); self.play(FadeOut(VGroup(e1, e2)), run_time=1)
        self.marks["total"] = [round(self.renderer.time, 3)]
        json.dump(self.marks, open("marks.json", "w"), indent=1)
