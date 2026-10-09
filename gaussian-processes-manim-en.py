import numpy as np
from manim import *

# Dark palette of the HTML explainer (procesos-gaussianos.html)
BG = "#0E131B"
INK = "#E4EAF3"
MUTED = "#9BA7B9"
LINE = "#2B3545"
MEAN = "#6EA6FF"
SAMPLE = "#AE9FF3"
POINT = "#FF9A55"
GOOD = "#4CC38A"
WARN = "#F08A5D"
FONT = "Inter"

config.background_color = BG

# ---------- Gaussian process (RBF kernel) ----------
XS = np.linspace(0, 10, 160)
LS, SF, NOISE = 1.1, 1.0, 1e-4
N_SAMPLES = 7


def rbf(a, b):
    d = a[:, None] - b[None, :]
    return SF**2 * np.exp(-0.5 * (d / LS) ** 2)


def f_true(x):
    return np.sin(0.9 * x) + 0.35 * np.cos(2.1 * x)


OBS_X = np.array([1.4, 3.0, 6.3, 8.4])
OBS_Y = f_true(OBS_X)

rng = np.random.default_rng(7)
Z = rng.standard_normal((len(XS), N_SAMPLES))


def posterior(k):
    """Mean, std and samples using the first k data points (same Z => smooth transitions)."""
    Kss = rbf(XS, XS)
    if k == 0:
        mu, cov = np.zeros_like(XS), Kss
    else:
        X, Y = OBS_X[:k], OBS_Y[:k]
        K = rbf(X, X) + NOISE * np.eye(k)
        Ks = rbf(XS, X)
        Kinv = np.linalg.inv(K)
        mu = Ks @ Kinv @ Y
        cov = Kss - Ks @ Kinv @ Ks.T
    cov = 0.5 * (cov + cov.T) + 1e-6 * np.eye(len(XS))
    L = np.linalg.cholesky(cov)
    samples = mu[:, None] + L @ Z
    sd = np.sqrt(np.clip(np.diag(cov), 0, None))
    return mu, sd, np.clip(samples, -2.9, 2.9)


class ProcesosGaussianos(Scene):
    def caption(self, text, sub=None):
        t = Text(text, font=FONT, weight=SEMIBOLD, color=INK).scale(0.62)
        grp = VGroup(t)
        if sub:
            s = Text(sub, font=FONT, color=MUTED).scale(0.42)
            grp.add(s)
            grp.arrange(DOWN, buff=0.18)
        grp.to_edge(UP, buff=0.4)
        return grp

    def swap_caption(self, old, new):
        if old is None:
            self.play(FadeIn(new, shift=0.2 * DOWN), run_time=0.8)
        else:
            self.play(FadeOut(old, shift=0.2 * UP), FadeIn(new, shift=0.2 * DOWN), run_time=0.8)
        return new

    def construct(self):
        # ---------- Title ----------
        title = Text("Gaussian processes", font=FONT, weight=BOLD, color=INK).scale(1.15)
        sub = Text("how to predict a curve… and know when you don't know", font=FONT, color=MUTED).scale(0.5)
        VGroup(title, sub).arrange(DOWN, buff=0.35)
        self.play(Write(title), run_time=1.4)
        self.play(FadeIn(sub, shift=0.2 * UP))
        self.wait(1.2)
        self.play(FadeOut(title), FadeOut(sub))

        # ---------- Axes ----------
        ax = Axes(
            x_range=[0, 10, 1], y_range=[-3, 3, 1],
            x_length=12, y_length=5.2,
            axis_config={"color": LINE, "stroke_width": 2, "include_ticks": False},
            tips=False,
        ).shift(0.55 * DOWN)

        def curve(ys, color, width=2.5, opacity=1.0):
            pts = [ax.c2p(x, y) for x, y in zip(XS, ys)]
            m = VMobject(stroke_color=color, stroke_width=width, stroke_opacity=opacity)
            m.set_points_smoothly(pts)
            return m

        def band(mu, sd):
            up = [ax.c2p(x, min(y, 3)) for x, y in zip(XS, mu + 2 * sd)]
            lo = [ax.c2p(x, max(y, -3)) for x, y in zip(XS, mu - 2 * sd)]
            return Polygon(*up, *lo[::-1], stroke_width=0, fill_color=MEAN, fill_opacity=0.16)

        def samples_group(S):
            return VGroup(*[curve(S[:, i], SAMPLE, 2, 0.75) for i in range(N_SAMPLES)])

        def dot_at(i):
            d = Dot(ax.c2p(OBS_X[i], OBS_Y[i]), radius=0.11, color=POINT)
            d.set_stroke(BG, width=4, background=True)
            return d

        self.play(Create(ax), run_time=1)

        # ---------- 1. Before seeing data ----------
        cap = self.swap_caption(None, self.caption(
            "Before seeing any data, many curves are possible",
            "each purple line is a shape the function could have"))
        mu, sd, S = posterior(0)
        bnd = band(mu, sd)
        smp = samples_group(S)
        self.play(FadeIn(bnd), run_time=0.8)
        self.play(LaggedStart(*[Create(c) for c in smp], lag_ratio=0.18), run_time=3)
        self.wait(1.5)

        # ---------- 2. Each data point rules out curves ----------
        cap = self.swap_caption(cap, self.caption(
            "Each data point rules out curves",
            "only the ones that pass through the measured points survive"))
        dots = VGroup()
        for k in range(1, len(OBS_X) + 1):
            d = dot_at(k - 1)
            dots.add(d)
            self.play(GrowFromCenter(d), Flash(d, color=POINT, line_length=0.18, flash_radius=0.25), run_time=0.6)
            mu, sd, S = posterior(k)
            self.play(Transform(smp, samples_group(S)), Transform(bnd, band(mu, sd)), run_time=1.4)
            self.bring_to_front(dots)
            self.wait(0.3)
        self.wait(1)

        # ---------- 3. The best bet ----------
        cap = self.swap_caption(cap, self.caption(
            "The average of the curves is the best bet",
            "the blue line is the prediction"))
        mean_line = curve(mu, MEAN, 5)
        self.play(smp.animate.set_stroke(opacity=0.18), Create(mean_line), run_time=1.6)
        self.bring_to_front(dots)
        self.wait(1.2)

        # ---------- 4. The zone of doubt ----------
        cap = self.swap_caption(cap, self.caption(
            "The zone of doubt: knowing when you don't know",
            "the band widens far from the data"))
        self.play(FadeOut(smp), bnd.animate.set_fill(opacity=0.26), run_time=0.8)

        # Point of most doubt between data points, and point of most certainty
        gap_mask = (XS > OBS_X[1]) & (XS < OBS_X[2])
        xi = XS[gap_mask][np.argmax(sd[gap_mask])]
        i_g = np.argmin(abs(XS - xi))
        top, bot = ax.c2p(xi, mu[i_g] + 2 * sd[i_g]), ax.c2p(xi, mu[i_g] - 2 * sd[i_g])
        doubt = DoubleArrow(bot, top, buff=0, color=WARN, stroke_width=4,
                            max_tip_length_to_length_ratio=0.12)
        doubt_lbl = Text("most unsure here", font=FONT, weight=MEDIUM, color=WARN).scale(0.42)
        doubt_lbl.next_to(top, RIGHT, buff=0.2).shift(0.45 * DOWN)

        sure_dot = dots[0]
        sure_lbl = Text("sure here", font=FONT, weight=MEDIUM, color=GOOD).scale(0.42)
        sure_lbl.move_to(ax.c2p(OBS_X[0] + 0.3, -2.2))
        sure_arrow = Arrow(sure_lbl.get_top(), sure_dot.get_bottom(), buff=0.1, color=GOOD,
                           stroke_width=4, max_tip_length_to_length_ratio=0.15)

        self.play(GrowFromCenter(doubt), FadeIn(doubt_lbl, shift=0.1 * LEFT))
        self.play(GrowArrow(sure_arrow), FadeIn(sure_lbl, shift=0.1 * UP))
        self.wait(2.5)

        # ---------- Closing ----------
        self.play(*[FadeOut(m) for m in [doubt, doubt_lbl, sure_arrow, sure_lbl, cap]], run_time=0.6)
        k1 = VGroup(Line(ORIGIN, 0.6 * RIGHT, color=MEAN, stroke_width=6),
                    Text("prediction", font=FONT, color=INK).scale(0.42)).arrange(RIGHT, buff=0.2)
        k2 = VGroup(Rectangle(width=0.6, height=0.3, stroke_width=0, fill_color=MEAN, fill_opacity=0.35),
                    Text("how unsure", font=FONT, color=INK).scale(0.42)).arrange(RIGHT, buff=0.2)
        k3 = VGroup(Dot(radius=0.1, color=POINT),
                    Text("measured data", font=FONT, color=INK).scale(0.42)).arrange(RIGHT, buff=0.2)
        legend = VGroup(k1, k2, k3).arrange(RIGHT, buff=0.7).to_edge(UP, buff=0.5)
        end = Text("A Gaussian process predicts, and tells you how sure it is.",
                   font=FONT, weight=SEMIBOLD, color=INK).scale(0.5).to_edge(DOWN, buff=0.35)
        self.play(FadeIn(legend, shift=0.2 * DOWN), run_time=0.8)
        self.play(Write(end), run_time=1.5)
        self.wait(2.5)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1)
