import json
import numpy as np
from manim import *

# Dark palette of the series
BG = "#0E131B"; INK = "#E4EAF3"; MUTED = "#9BA7B9"; LINE = "#2B3545"
MEAN = "#6EA6FF"; SAMPLE = "#AE9FF3"; POINT = "#FF9A55"; GOOD = "#4CC38A"; WARN = "#F08A5D"
FONT = "Inter"
config.background_color = BG
P = np.array(json.load(open("../../mt/m11/embed.json"))["dur"], float)
MU, SIG = P.mean(), P.std()


def T(s, sc=0.45, col=INK, w=None):
    return Text(s, font=FONT, color=col, weight=w or NORMAL).scale(sc)


def fmt(v, d=1):
    return f"{v:.{d}f}"


class Chart:
    """Histogram axes: maps data (x, y) to scene points."""

    def __init__(self, x0, x1, y1, xt, xl, title=None, left=-5.6, right=5.6, bottom=-3.0, top=1.6):
        self.x0, self.x1, self.y1 = x0, x1, y1
        self.L, self.R, self.B, self.Tp = left, right, bottom, top
        g = VGroup(Line(self.P(x0, 0), self.P(x1, 0), stroke_color=MUTED, stroke_width=2))
        for v in xt:
            g.add(T(str(v), 0.3, MUTED).next_to(self.P(v, 0), DOWN, buff=0.12))
        if xl:
            g.add(T(xl, 0.32, INK).next_to(self.P(x1, 0), DOWN, buff=0.45).align_to(self.P(x1, 0), RIGHT))
        if title:
            g.add(T(title, 0.34, MUTED).move_to(self.P(x0, y1)).align_to(self.P(x0, y1), LEFT).shift(0.15 * UP))
        self.axes = g

    def P(self, x, y):
        return np.array([self.L + (x - self.x0) / (self.x1 - self.x0) * (self.R - self.L),
                         self.B + y / self.y1 * (self.Tp - self.B), 0])

    def bars(self, counts, x0, bw, col, op=0.75, ymax=None):
        ymax = ymax or self.y1
        g = VGroup()
        for k, c in enumerate(counts):
            h = max(min(c, ymax), 1e-3)
            a, b = self.P(x0 + k * bw, 0), self.P(x0 + (k + 1) * bw, h)
            g.add(Rectangle(width=max(0.01, b[0] - a[0] - 0.02), height=max(0.001, b[1] - a[1]), stroke_width=0,
                            fill_color=col, fill_opacity=op).move_to([(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, 0]))
        return g

    def curve(self, f, col=MEAN, w=5, dash=False):
        xs = np.linspace(self.x0, self.x1, 300)
        pts = [self.P(x, min(f(x), self.y1 * 1.02)) for x in xs]
        c = VMobject(stroke_color=col, stroke_width=w).set_points_smoothly(pts)
        return DashedVMobject(c, num_dashes=60) if dash else c

    def vline(self, x, col=MEAN, w=4):
        return Line(self.P(x, 0), self.P(x, self.y1), stroke_color=col, stroke_width=w)


def hist(v, x0, x1, bw):
    nb = int(round((x1 - x0) / bw))
    k = np.clip(((np.asarray(v) - x0) / bw).astype(int), 0, nb - 1)
    return np.bincount(k, minlength=nb).astype(float)


def npdf(x, m, s):
    return np.exp(-0.5 * ((x - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))


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
        rng = np.random.default_rng(11)
        title = Text("Sampling and the central limit theorem", font=FONT, weight=BOLD, color=INK).scale(0.85)
        sub = Text("why averages turn into a bell, and how much they move", font=FONT, color=MUTED).scale(0.48)
        VGroup(title, sub).arrange(DOWN, buff=0.35)
        self.mark("title"); self.play(Write(title), run_time=1.3)
        self.mark("subtitle"); self.play(FadeIn(sub, shift=0.2 * UP)); self.wait(1.8)
        self.mark("title_out"); self.play(FadeOut(title), FadeOut(sub))

        # ---------- 1. the population ----------
        cap = self.swap(None, self.caption("The population: 12,000 stops", "half last under 19 minutes; a few last hours"))
        PC = hist(P, 0, 120, 4)
        C1 = Chart(0, 120, PC.max() * 1.1, [0, 20, 40, 60, 80, 100, 120], "stop duration (minutes)", right=2.3, bottom=-3.0, top=1.5)
        pb = C1.bars(PC, 0, 4, MUTED, 0.55)
        self.play(FadeIn(C1.axes), run_time=0.6)
        self.mark("pop"); self.play(LaggedStart(*[GrowFromEdge(b, DOWN) for b in pb], lag_ratio=0.03), run_time=1.6)
        ml = C1.vline(MU)
        info = VGroup(T("true mean", 0.42, MEAN, BOLD), T(f"μ = {fmt(MU)} min", 0.44, INK), T("standard deviation", 0.38, MUTED), T(f"σ = {fmt(SIG)} min", 0.42, INK),
                      T("skewed, with a long tail", 0.38, MUTED)).arrange(DOWN, aligned_edge=LEFT, buff=0.16).move_to([0, -0.4, 0]).align_to(np.array([2.9, 0, 0]), LEFT)
        self.play(Create(ml), FadeIn(info, lag_ratio=0.2), run_time=1.0); self.wait(2.2)

        # ---------- 2. one sample, one mean ----------
        cap = self.swap(cap, self.caption("One sample, one mean", "16 random stops: each sample gives a different mean"))
        Cs = Chart(0, 120, PC.max() * 1.1, [0, 20, 40, 60, 80, 100, 120], "stop duration (minutes)", right=2.3, bottom=0.15, top=2.0)
        Cm = Chart(0, 120, 1, [0, 20, 40, 60, 80, 100, 120], "mean of each sample (minutes)", title="the sample means", right=2.3, bottom=-3.1, top=-1.55)
        self.play(FadeOut(info), ReplacementTransform(C1.axes, Cs.axes), ReplacementTransform(pb, Cs.bars(PC, 0, 4, MUTED, 0.55)),
                  ReplacementTransform(ml, Cs.vline(MU)), run_time=1.0)
        self.play(FadeIn(Cm.axes), run_time=0.5)
        means = []; chips = []
        last_dots, last_tick = None, None
        for k in range(4):
            x = P[rng.integers(0, len(P), 16)]
            m = x.mean(); means.append(m)
            ys = Cs.B + 0.72 * (Cs.Tp - Cs.B) + rng.uniform(-0.15, 0.15, 16)
            dots = VGroup(*[Dot([Cs.P(min(v, 119), 0)[0], y, 0], radius=0.06, color=POINT) for v, y in zip(x, ys)])
            tick = Line(Cs.P(m, 0) + 0.45 * DOWN, Cm.P(m, 0), stroke_color=POINT, stroke_width=4)
            lab = T(f"mean {fmt(m)}", 0.36, POINT, BOLD).next_to(Cm.P(m, 0.6), RIGHT, buff=0.1)
            chip = Rectangle(width=0.12, height=0.2, stroke_width=0, fill_color=POINT, fill_opacity=0.9).move_to(Cm.P(m, 0.1 + 0.16 * sum(abs(mm - m) < 2 for mm in means[:-1])))
            olds = [o for o in (last_dots, last_tick) if o is not None]
            if k == 0:
                self.mark("sample")
            self.play(*[FadeOut(o) for o in olds], LaggedStart(*[GrowFromCenter(d) for d in dots], lag_ratio=0.04), run_time=0.9 if k == 0 else 0.6)
            self.play(Create(tick), FadeIn(lab), run_time=0.5)
            self.play(FadeIn(chip, shift=0.3 * DOWN), FadeOut(lab), run_time=0.4)
            last_dots, last_tick = dots, tick; chips.append(chip)
        self.wait(0.6)

        # ---------- 3. many samples ----------
        cap = self.swap(cap, self.caption("Many samples: the means pile up", "around the true mean, and much tighter than the stops"))
        self.play(FadeOut(last_dots), FadeOut(last_tick), *[FadeOut(c) for c in chips], run_time=0.4)
        M16 = P[rng.integers(0, len(P), (3000, 16))].mean(1)
        bw = 2
        prev = None
        for i, k in enumerate([30, 300, 3000]):
            H = hist(M16[:k], 0, 120, bw)
            Cm.y1 = 1.0
            b = Cm.bars(H / H.max() * 0.92, 0, bw, POINT, 0.8)
            if i == 0:
                self.mark("pile")
            self.play(Transform(prev, b) if prev else FadeIn(b), run_time=0.8)
            if prev is None:
                prev = b
        sc = 0.92 / hist(M16, 0, 120, bw).max() * 3000 * bw
        bell = Cm.curve(lambda x: sc * npdf(x, MU, SIG / 4), MEAN, 4)
        mv = Cm.vline(MU, MEAN, 3)
        info = VGroup(T("3,000 means of 16", 0.4, POINT, BOLD), T(f"they wobble {fmt(M16.std())} min", 0.42, INK), T(f"σ / √16 = {fmt(SIG)} / 4 = {fmt(SIG / 4)}", 0.4, MEAN, BOLD),
                      T("the standard error", 0.38, MUTED)).arrange(DOWN, aligned_edge=LEFT, buff=0.16).move_to([0, -1.0, 0]).align_to(np.array([2.9, 0, 0]), LEFT)
        self.mark("bell"); self.play(Create(bell), Create(mv), FadeIn(info, lag_ratio=0.2), run_time=1.2); self.wait(2.6)
        self.clear_but(cap)

        # ---------- 4. the square-root law ----------
        cap = self.swap(cap, self.caption("The square-root law", "4 times more data, half the error"))
        C = Chart(0, 60, 0.115, [0, 10, 20, 30, 40, 50, 60], "sample mean (minutes)", right=2.3)
        self.play(FadeIn(C.axes), Create(C.vline(MU, MUTED, 2)), run_time=0.6)
        cols = [SAMPLE, POINT, MEAN]
        rows = VGroup()
        for i, n in enumerate([4, 16, 64]):
            s = SIG / np.sqrt(n)
            cv = C.curve(lambda x, s=s: npdf(x, MU, s), cols[i], 5)
            rows.add(T(f"n = {n}:  ±{fmt(s)} min   (${4 * n})", 0.42, cols[i], BOLD))
            if i == 0:
                rows.arrange(DOWN, aligned_edge=LEFT).move_to([0, 0.5, 0]).align_to(np.array([2.9, 0, 0]), LEFT)
            else:
                rows[-1].next_to(rows[-2], DOWN, aligned_edge=LEFT, buff=0.3)
            self.mark("root"); self.play(Create(cv), FadeIn(rows[-1]), run_time=1.0); self.wait(0.8)
        note = VGroup(T("standard error = σ / √n", 0.42, INK, BOLD), T("each stop checked: $4", 0.36, MUTED)).arrange(DOWN, aligned_edge=LEFT, buff=0.14).next_to(rows, DOWN, aligned_edge=LEFT, buff=0.5)
        self.play(FadeIn(note), run_time=0.6); self.wait(2.2)
        self.clear_but(cap)

        # ---------- 5. TCL ----------
        cap = self.swap(cap, self.caption("From any shape to a bell", "the central limit theorem: means of n stops"))
        C = Chart(0, 120, 1.0, [0, 20, 40, 60, 80, 100, 120], "mean of n stops (minutes)", right=2.3)
        self.play(FadeIn(C.axes), run_time=0.5)
        prev, lab, bl = None, None, None
        for n in [1, 5, 30, 100]:
            M = P[rng.integers(0, len(P), (4000, n))].mean(1)
            H = hist(M, 0, 120, 2); b = C.bars(H / H.max() * 0.92, 0, 2, POINT, 0.8)
            sc = 0.92 / H.max() * 4000 * 2
            nb = C.curve(lambda x, s=SIG / np.sqrt(n), sc=sc: sc * npdf(x, MU, s), MEAN, 4)
            sk = ((M - M.mean()) ** 3).mean() / M.std() ** 3
            nl = VGroup(T(f"n = {n}", 0.6, INK, BOLD), T(f"skewness {fmt(sk, 2)}", 0.42, POINT, BOLD), T("0 = symmetric", 0.36, MUTED)).arrange(DOWN, aligned_edge=LEFT, buff=0.16).move_to([0, 0.2, 0]).align_to(np.array([2.9, 0, 0]), LEFT)
            self.mark("tcl")
            if prev is None:
                self.play(FadeIn(b), FadeIn(nl), run_time=0.8); prev, lab = b, nl
                bl = nb; self.play(Create(bl), run_time=0.6)
            else:
                self.play(Transform(prev, b), Transform(lab, nl), Transform(bl, nb), run_time=1.0)
            self.wait(1.2)
        self.wait(1.0)
        self.clear_but(cap)

        # ---------- 6. when it fails ----------
        cap = self.swap(cap, self.caption("When it fails", "the theorem has fine print"))
        items = [("Very heavy tails", "delays from incidents: the standard deviation is infinite", WARN),
                 ("Data that look alike", "30 consecutive stops of one truck are worth 2 independent ones", SAMPLE),
                 ("Biased samples", "if long stops get lost: 20.9 instead of 28.6", POINT)]
        rows = VGroup()
        for h, s, c in items:
            r = VGroup(Dot(radius=0.11, color=c), VGroup(T(h, 0.5, c, BOLD), T(s, 0.4, MUTED)).arrange(DOWN, aligned_edge=LEFT, buff=0.12)).arrange(RIGHT, buff=0.35, aligned_edge=UP)
            rows.add(r)
        rows.arrange(DOWN, aligned_edge=LEFT, buff=0.55).move_to([-0.2, -0.5, 0])
        for i, r in enumerate(rows):
            self.mark("fail"); self.play(FadeIn(r, shift=0.2 * RIGHT), run_time=0.7); self.wait(1.3)
        warn = T("more data does not fix a biased sample", 0.46, INK, BOLD).next_to(rows, DOWN, buff=0.6)
        self.play(FadeIn(warn), run_time=0.6); self.wait(2.2)
        self.clear_but(cap)

        # ---------- Closing ----------
        self.mark("legend")
        self.play(FadeOut(cap), run_time=0.6)
        e1 = Text("A mean is an estimate that wobbles", font=FONT, weight=BOLD, color=INK).scale(0.7)
        e2 = VGroup(T("right on average, wobbling σ / √n", 0.42, MUTED), T("half the error costs 4 times more data", 0.42, MUTED),
                    T("with large n, the means form a bell", 0.42, MUTED), T("if the data are independent and the sample is not biased", 0.42, MUTED)).arrange(DOWN, buff=0.18)
        VGroup(e1, e2).arrange(DOWN, buff=0.45)
        self.mark("end_text"); self.play(Write(e1), run_time=1.2); self.play(FadeIn(e2, shift=0.2 * UP, lag_ratio=0.15), run_time=1.0)
        self.wait(3.5)
        self.mark("fade"); self.play(FadeOut(VGroup(e1, e2)), run_time=1)
        self.marks["total"] = [round(self.renderer.time, 3)]
        json.dump(self.marks, open("marks.json", "w"), indent=1)
