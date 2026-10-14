import json
import math
import numpy as np
from manim import *

# Dark palette of the series
BG = "#0E131B"; INK = "#E4EAF3"; MUTED = "#9BA7B9"; LINE = "#2B3545"
MEAN = "#6EA6FF"; SAMPLE = "#AE9FF3"; POINT = "#FF9A55"; GOOD = "#4CC38A"; WARN = "#F08A5D"
FONT = "Inter"
config.background_color = BG


def T(s, sc=0.45, col=INK, w=None):
    return Text(s, font=FONT, color=col, weight=w or NORMAL).scale(sc)


def npdf(x, m, s):
    return np.exp(-0.5 * ((x - m) / s) ** 2) / (s * math.sqrt(2 * math.pi))


def binom(n, p):
    return np.array([math.comb(n, k) * p**k * (1 - p) ** (n - k) for k in range(n + 1)])


def pois(l, km):
    return np.array([math.exp(-l) * l**k / math.factorial(k) for k in range(km + 1)])


class Chart:
    """Chart frame: maps data (x, y) to scene points."""

    def __init__(self, x0, x1, y1, xt, yt, xl, yl, fy, left=-5.6, right=5.6, bottom=-3.1, top=1.55):
        self.x0, self.x1, self.y1 = x0, x1, y1
        self.L, self.R, self.B, self.Tp = left, right, bottom, top
        g = VGroup()
        for v in yt:
            g.add(Line(self.P(x0, v), self.P(x1, v), stroke_color=LINE, stroke_width=1.5))
            g.add(T(fy(v), 0.32, MUTED).next_to(self.P(x0, v), LEFT, buff=0.15))
        g.add(Line(self.P(x0, 0), self.P(x1, 0), stroke_color=MUTED, stroke_width=2))
        for v in xt:
            g.add(T(str(v), 0.32, MUTED).next_to(self.P(v, 0), DOWN, buff=0.15))
        g.add(T(xl, 0.34, INK).next_to(self.P(x1, 0), DOWN, buff=0.5).align_to(self.P(x1, 0), RIGHT))
        if yl:
            g.add(T(yl, 0.32, MUTED).next_to(self.P(x0, y1), UP, buff=0.12).align_to(self.P(x0, y1), LEFT))
        self.axes = g

    def P(self, x, y):
        return np.array([self.L + (x - self.x0) / (self.x1 - self.x0) * (self.R - self.L),
                         self.B + min(y, self.y1) / self.y1 * (self.Tp - self.B), 0])

    def bar(self, a, b, h, col, op=0.7):
        h = max(h, 1e-4)
        p0, p1 = self.P(a, 0), self.P(b, h)
        r = Rectangle(width=p1[0] - p0[0], height=max(0.001, p1[1] - p0[1]), stroke_width=1.5, stroke_color=col, fill_color=col, fill_opacity=op)
        return r.move_to([(p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2, 0])

    def curve(self, f, a, b, col, w=5, n=300, dash=False):
        xs = np.linspace(a, b, n)
        c = VMobject(stroke_color=col, stroke_width=w).set_points_as_corners([self.P(x, f(x)) for x in xs])
        return DashedVMobject(c, num_dashes=60) if dash else c

    def area(self, f, a, b, col, op=0.6, n=160):
        xs = np.linspace(a, b, n)
        pts = [self.P(a, 0)] + [self.P(x, f(x)) for x in xs] + [self.P(b, 0)]
        return Polygon(*pts, stroke_width=0, fill_color=col, fill_opacity=op)

    def vline(self, x, col, h=None, w=3, dash=True):
        l = Line(self.P(x, 0), self.P(x, h if h is not None else self.y1), stroke_color=col, stroke_width=w)
        return DashedLine(l.get_start(), l.get_end(), color=col, stroke_width=w, dash_length=0.12) if dash else l


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

    def fit_right(self, g, xmax=6.6):
        if g.get_right()[0] > xmax:
            g.shift((g.get_right()[0] - xmax) * LEFT)
        return g

    def clear_but(self, keep):
        self.play(*[FadeOut(m) for m in self.mobjects if m is not keep], run_time=0.6)

    def construct(self):
        self.marks = {}
        title = Text("Distributions", font=FONT, weight=BOLD, color=INK).scale(1.0)
        sub = Text("what shape the data has and when each one shows up", font=FONT, color=MUTED).scale(0.5)
        VGroup(title, sub).arrange(DOWN, buff=0.35)
        self.mark("title"); self.play(Write(title), run_time=1.3)
        self.mark("subtitle"); self.play(FadeIn(sub, shift=0.2 * UP)); self.wait(1.8)
        self.mark("title_out"); self.play(FadeOut(title), FadeOut(sub))

        # ---------- 1. from data to shape ----------
        cap = self.swap(None, self.caption("From data to shape", "fuel used on each trip: how often each value comes up"))
        C = Chart(22, 42, 30, [22, 26, 30, 34, 38, 42], [0, 10, 20, 30], "liters per 100 km", "% of trips", lambda v: f"{v}%")
        rng = np.random.default_rng(3)
        data = rng.normal(32, 2.5, 5000)

        def hist(n):
            x = np.clip(data[:n], 22, 41.999)
            h = np.histogram(x, bins=np.arange(22, 43, 1))[0] / n * 100
            return VGroup(*[C.bar(22 + i + 0.06, 23 + i - 0.06, h[i], POINT, 0.6) for i in range(20)]), x.mean()

        bars, m = hist(20)
        ml = C.vline(m, SAMPLE); mt = T(f"mean {m:.1f}", 0.36, SAMPLE, BOLD).next_to(ml, UP, buff=0.08)
        nt = T("20 trips", 0.5, INK, BOLD).move_to([4.2, 1.2, 0])
        self.play(FadeIn(C.axes), run_time=0.8)
        self.mark("hist"); self.play(FadeIn(bars, lag_ratio=0.05), FadeIn(ml), FadeIn(mt), FadeIn(nt), run_time=1.2); self.wait(1.2)
        for n in (200, 5000):
            nb, m = hist(n)
            nml = C.vline(m, SAMPLE); nmt = T(f"mean {m:.1f}", 0.36, SAMPLE, BOLD).next_to(nml, UP, buff=0.08)
            nnt = T(f"{n:,} trips", 0.5, INK, BOLD).move_to([4.2, 1.2, 0])
            self.mark("more"); self.play(Transform(bars, nb), Transform(ml, nml), Transform(mt, nmt), Transform(nt, nnt), run_time=1.2); self.wait(1.0)
        bell = C.curve(lambda x: 100 * npdf(x, 32, 2.5), 22, 42, MEAN, 6)
        bt = T("the underlying shape:\nthe distribution", 0.42, MEAN, BOLD).move_to([4.2, 0.2, 0])
        self.mark("bell"); self.play(Create(bell), FadeIn(bt), run_time=1.4); self.wait(3.0)
        self.clear_but(cap)

        # ---------- 2. the normal ----------
        cap = self.swap(cap, self.caption("The normal: a center μ and a width σ", "a sum of many small effects: wind, load, traffic, driver"))
        C = Chart(20, 44, 22, [20, 24, 28, 32, 36, 40, 44], [0, 10, 20], "liters per 100 km", None, lambda v: f"{v}")
        f = lambda x: 100 * npdf(x, 32, 2.5)
        self.play(FadeIn(C.axes), Create(C.curve(f, 20, 44, MEAN, 6)), run_time=1.0)
        labs = [("68%", 1), ("95%", 2), ("99.7%", 3)]
        for i, (t, k) in enumerate(labs):
            band = Polygon(C.P(32 - k * 2.5, 0), C.P(32 + k * 2.5, 0), C.P(32 + k * 2.5, 22), C.P(32 - k * 2.5, 22), stroke_width=0, fill_color=MEAN, fill_opacity=0.10).set_z_index(-1)
            y = 1.25 - i * 0.42
            seg = Line([C.P(32 - k * 2.5, 0)[0], y, 0], [C.P(32 + k * 2.5, 0)[0], y, 0], stroke_color=MEAN, stroke_width=3)
            lt = T(f"±{k}σ: {t}", 0.36, INK, BOLD).next_to(seg, RIGHT, buff=0.15)
            self.mark("band"); self.play(FadeIn(band), Create(seg), FadeIn(lt), run_time=0.8); self.wait(0.5)
        tail = C.area(f, 37, 44, POINT, 0.85)
        vl = C.vline(37, POINT, 14, 4, dash=False)
        zt = VGroup(T("a 37 L trip", 0.4, POINT, BOLD), T("z = (37 − 32) / 2.5 = 2", 0.4, INK), T("only 2.3% use more", 0.4, INK)).arrange(DOWN, aligned_edge=LEFT, buff=0.12).move_to([4.0, -0.9, 0])
        self.mark("z"); self.play(Create(vl), FadeIn(tail), run_time=0.8); self.play(FadeIn(zt, shift=0.2 * RIGHT), run_time=0.7); self.wait(3.5)
        self.clear_but(cap)

        # ---------- 3. the binomial ----------
        cap = self.swap(cap, self.caption("The binomial: how many of n come out \"yes\"", "20 deliveries a day, each one late with p = 10%"))
        C = Chart(-0.6, 12.6, 30, list(range(0, 13)), [0, 10, 20, 30], "late deliveries in the day", "% of days", lambda v: f"{v}%", right=3.0)
        B = binom(20, 0.1)
        bb = VGroup(*[C.bar(k - 0.36, k + 0.36, 100 * B[k], MEAN, 0.6) for k in range(13)])
        self.play(FadeIn(C.axes), run_time=0.6)
        self.mark("bars"); self.play(LaggedStart(*[GrowFromEdge(b, DOWN) for b in bb], lag_ratio=0.08), run_time=1.4)
        info = VGroup(T("most common: 1 or 2", 0.4, INK), T("6 or more: 1.1% of days", 0.42, POINT, BOLD), T("about 3 days a year", 0.4, INK),
                      T("at $15 per delay:", 0.38, MUTED), T("$30 a day on average", 0.4, INK)).arrange(DOWN, aligned_edge=LEFT, buff=0.18).move_to([4.9, -0.6, 0])
        self.fit_right(info)
        ml = C.vline(2, SAMPLE, 30); mt = T("mean n × p = 2", 0.42, SAMPLE, BOLD).next_to(info, UP, buff=0.3, aligned_edge=LEFT)
        self.play(FadeIn(ml), FadeIn(mt), run_time=0.6)
        self.play(FadeIn(info[0]), run_time=0.5)
        self.mark("tail"); self.play(*[bb[k].animate.set_fill(POINT, 0.95).set_stroke(POINT) for k in range(6, 13)], FadeIn(info[1:3]), run_time=0.8)
        self.play(FadeIn(info[3:]), run_time=0.6); self.wait(3.2)

        # ---------- 4. the Poisson ----------
        cap = self.swap(cap, self.caption("The Poisson: counting rare things over time", "a truck's harsh-braking alerts, λ = 2 a day on average"))
        Pp = pois(2, 12)
        pb = VGroup(*[C.bar(k - 0.36, k + 0.36, 100 * Pp[k], MEAN, 0.6) for k in range(13)])
        xl2 = T("alerts in the day", 0.34, INK).move_to(C.axes[-2]).align_to(C.axes[-2], RIGHT)
        self.mark("morph"); self.play(FadeOut(info), FadeOut(mt), Transform(bb, pb), Transform(C.axes[-2], xl2), run_time=1.2)
        rings = VGroup(*[Circle(0.09, color=SAMPLE, stroke_width=4).move_to(C.P(k, 100 * B[k])) for k in range(8)])
        info2 = VGroup(T("binomial (20, 0.1)", 0.4, SAMPLE, BOLD), T("almost the same: many tries,", 0.38, MUTED), T("each one very rare", 0.38, MUTED),
                       T("mean = variance = λ = 2", 0.42, INK, BOLD), T("7 or more alerts: 0.45%", 0.42, POINT, BOLD), T("1 day in 221", 0.4, INK)).arrange(DOWN, aligned_edge=LEFT, buff=0.18).move_to([4.8, -0.5, 0])
        self.fit_right(info2)
        self.play(FadeIn(rings), FadeIn(info2[:3]), run_time=0.8); self.wait(2.0)
        self.play(FadeIn(info2[3]), run_time=0.5)
        self.mark("ptail"); self.play(*[bb[k].animate.set_fill(POINT, 0.95).set_stroke(POINT) for k in range(7, 13)], *[bb[k].animate.set_fill(MEAN, 0.6).set_stroke(MEAN) for k in range(6, 7)], FadeIn(info2[4:]), run_time=0.8)
        self.wait(3.2)
        self.clear_but(cap)

        # ---------- 5. long tails ----------
        cap = self.swap(cap, self.caption("Long tails: when the bell misleads you", "stop duration: almost all short, a few very long"))
        C = Chart(-30, 120, 7, [-30, 0, 30, 60, 90, 120], [0, 2, 4, 6], "minutes stopped", None, lambda v: f"{v}", right=2.6)
        s, mu = 0.8, math.log(12); mean = 12 * math.exp(s * s / 2); sd = mean * math.sqrt(math.exp(s * s) - 1)
        fl = lambda x: 0 if x <= 0 else 100 * npdf(math.log(x), mu, s) / x
        fn = lambda x: 100 * npdf(x, mean, sd)
        imp = Polygon(C.P(-30, 0), C.P(0, 0), C.P(0, 7), C.P(-30, 7), stroke_width=0, fill_color=WARN, fill_opacity=0.14)
        it = T("impossible", 0.36, WARN, BOLD).move_to((C.P(-30, 3.2) + C.P(0, 3.2)) / 2)
        self.play(FadeIn(C.axes), run_time=0.6)
        self.mark("lognorm"); self.play(Create(C.curve(fl, 0.05, 120, MEAN, 6, 400)), run_time=1.2)
        med = C.vline(12, INK, 6.2, 2); mdt = T("median 12", 0.34, INK, BOLD).next_to(med, UP, buff=0.06).shift(0.55 * LEFT)
        avg = C.vline(mean, INK, 5.2, 2); avt = T("mean 16.5", 0.34, INK, BOLD).next_to(avg, UP, buff=0.06).shift(0.5 * RIGHT)
        self.play(FadeIn(med), FadeIn(mdt), FadeIn(avg), FadeIn(avt), run_time=0.7); self.wait(1.0)
        info3 = VGroup(T("bell with the same mean", 0.38, SAMPLE, BOLD), T("and standard deviation:", 0.38, SAMPLE, BOLD),
                       T("14.5% \"negative\" stops", 0.38, INK), T("longer than 1 hour:", 0.38, MUTED), T("real 2.2%", 0.42, POINT, BOLD), T("bell 0.27%", 0.42, SAMPLE, BOLD)).arrange(DOWN, aligned_edge=LEFT, buff=0.16).move_to([5.0, -0.5, 0])
        self.fit_right(info3, 6.5)
        self.mark("wrong"); self.play(Create(C.curve(fn, -30, 120, SAMPLE, 4, 300, dash=True)), FadeIn(imp), FadeIn(it), FadeIn(info3[:3]), run_time=1.2); self.wait(1.2)
        self.play(FadeIn(C.area(fl, 60, 120, POINT, 0.9)), FadeIn(info3[3:]), run_time=0.8); self.wait(3.8)
        self.clear_but(cap)

        # ---------- Closing ----------
        self.mark("legend")
        self.play(FadeOut(cap), run_time=0.6)
        e1 = Text("Do you count or measure?", font=FONT, weight=BOLD, color=INK).scale(0.8)
        e2 = VGroup(T("count \"yeses\" out of n tries: binomial", 0.42, MUTED), T("count events over time: Poisson", 0.42, MUTED),
                    T("measure, many effects add up: normal", 0.42, MUTED), T("measure, effects multiply: long tail", 0.42, MUTED)).arrange(DOWN, buff=0.18)
        VGroup(e1, e2).arrange(DOWN, buff=0.45)
        self.mark("end_text"); self.play(Write(e1), run_time=1.2); self.play(FadeIn(e2, shift=0.2 * UP, lag_ratio=0.15), run_time=1.0)
        self.wait(3.5)
        self.mark("fade"); self.play(FadeOut(VGroup(e1, e2)), run_time=1)
        self.marks["total"] = [round(self.renderer.time, 3)]
        json.dump(self.marks, open("marks.json", "w"), indent=1)
