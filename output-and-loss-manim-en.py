import json
import math
import numpy as np
from manim import *

# Dark palette of the series
BG = "#0E131B"; INK = "#E4EAF3"; MUTED = "#9BA7B9"; LINE = "#2B3545"
MEAN = "#6EA6FF"; SAMPLE = "#AE9FF3"; POINT = "#FF9A55"; GOOD = "#4CC38A"; WARN = "#F08A5D"
FONT = "Inter"
config.background_color = BG
D = json.load(open("../../rn/r4-en/embed.json", encoding="utf-8"))
U40 = (np.array(D["x40"]) - 30) / 30
Y40 = np.array(D["y40"])
B40, W40 = D["b40"], D["w40"]


def T(s, sc=0.45, col=INK, w=None):
    return Text(s, font=FONT, color=col, weight=w or NORMAL).scale(sc)


def fmt(v, d=1):
    s = f"{abs(v):,.{d}f}"
    return ("−" if v < 0 and round(abs(v), d) != 0 else "") + s


sig = lambda z: 1 / (1 + np.exp(-z))
def softmax(z):
    e = np.exp(np.array(z) - max(z)); return e / e.sum()
def loss40(w):
    z = B40 + w * U40
    return float(np.mean(np.where(Y40 == 1, np.log1p(np.exp(-z)), np.log1p(np.exp(z)))))
def slope40(w):
    return float(np.mean((sig(B40 + w * U40) - Y40) * U40))


class Panel:
    """Simple 2D axes: maps data (x, y) to the scene inside a box."""

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


def bar(P, x, v, col, w=0.5, base=0.0):
    top, bot = P.P(x, v), P.P(x, base)
    h = abs(top[1] - bot[1])
    return Rectangle(width=w, height=max(h, 0.02), stroke_width=0, fill_color=col, fill_opacity=0.88).move_to((top + bot) / 2)


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

    def recap(self, old, new):
        """Changes the caption without a new mark (the drawing changes meaning within the same part)."""
        return [FadeOut(old, shift=0.2 * UP), FadeIn(new, shift=0.2 * DOWN)]

    def clear_but(self, keep):
        self.play(*[FadeOut(m) for m in self.mobjects if m is not keep], run_time=0.6)

    def construct(self):
        self.marks = {}
        title = Text("Output and loss", font=FONT, weight=BOLD, color=INK).scale(0.95)
        sub = Text("what the network answers and how it gets graded", font=FONT, color=MUTED).scale(0.46)
        VGroup(title, sub).arrange(DOWN, buff=0.35)
        self.mark("title"); self.play(Write(title), run_time=1.4)
        self.mark("subtitle"); self.play(FadeIn(sub, shift=0.2 * UP)); self.wait(1.5)
        self.mark("title_out"); self.play(FadeOut(title), FadeOut(sub))

        # ---------- 1. one network, three heads ----------
        cap = self.swap(None, self.caption("One network, three heads", "each question needs its own last activation"))
        body = RoundedRectangle(width=2.6, height=3.6, corner_radius=0.2, stroke_color=MUTED, stroke_width=2.5).move_to([-4.6, -0.5, 0])
        btxt = VGroup(T("shared", 0.42, INK, BOLD), T("body", 0.42, INK, BOLD), T("trip data", 0.3, MUTED)).arrange(DOWN, buff=0.12).move_to(body)
        self.play(Create(body), FadeIn(btxt), run_time=0.8)
        V = D["viaje"]; q = softmax(V["logits"]); p = float(sig(V["z_averia"]))
        HEADS = [("How much fuel?", "linear", f"{fmt(V['consumo_pred'])} L/100 km", MEAN),
                 ("Breakdown in 30 days?", "sigmoid", f"{fmt(100 * p, 0)}%", POINT),
                 ("What kind of trip?", "softmax", f"urb. {fmt(100 * q[0], 0)}% · reg. {fmt(100 * q[1], 0)}% · hwy {fmt(100 * q[2], 0)}%", GOOD)]
        for i, (qq, act, ans, col) in enumerate(HEADS):
            y = 1.0 - 1.5 * i
            box = RoundedRectangle(width=1.9, height=0.8, corner_radius=0.15, stroke_color=col, stroke_width=3).move_to([-1.0, y - 0.5, 0])
            at = T(act, 0.36, col, BOLD).move_to(box)
            arr = Arrow(body.get_right(), box.get_left(), buff=0.1, stroke_width=3, color=MUTED, max_tip_length_to_length_ratio=0.12)
            ql = T(qq, 0.36, INK, BOLD).move_to([0.4, y - 0.28, 0], aligned_edge=LEFT)
            al = T(ans, 0.38, col, BOLD).move_to([0.4, y - 0.74, 0], aligned_edge=LEFT)
            self.mark("head")
            self.play(GrowArrow(arr), Create(box), FadeIn(at), run_time=0.7)
            self.play(FadeIn(ql, shift=0.1 * RIGHT), FadeIn(al, shift=0.1 * RIGHT), run_time=0.6)
            self.wait(1.3)
        self.wait(1.2)
        self.clear_but(cap)

        # ---------- 2. softmax ----------
        cap = self.swap(cap, self.caption("Softmax: exponentiate and split", "the z values for urban, regional and highway: 0.4, 1.6 and 0.9"))
        Z = np.array(V["logits"]); E = np.exp(Z); PR = E / E.sum()
        COLS = [SAMPLE, MEAN, GOOD]
        P1 = Panel(0, 3, 0, 2, (-6.3, -2.6, -2.2, 1.7), yt=(0, 1, 2), yf=lambda v: fmt(v, 0), yl="1 · the z values")
        P2 = Panel(0, 3, 0, 6, (-1.8, 1.9, -2.2, 1.7), yt=(0, 2, 4, 6), yf=lambda v: fmt(v, 0), yl="2 · eᶻ")
        P3 = Panel(0, 3, 0, 1, (2.9, 6.6, -2.2, 1.7), yt=(0, 0.5, 1), yf=lambda v: fmt(100 * v, 0) + "%", yl="3 · ÷ sum = %")
        names = ["urb.", "reg.", "hwy"]
        def bars(P, vals, d, fmtf):
            g = VGroup()
            for k in range(3):
                b = bar(P, k + 0.5, vals[k], COLS[k], 0.62).set_z_index(2)
                g.add(b, T(fmtf(vals[k]), 0.3, INK, BOLD).next_to(b, UP, buff=0.08))
            return g
        labs = VGroup(*[T(names[k], 0.28, MUTED).next_to(P.P(k + 0.5, P.y0), DOWN, buff=0.12) for P in (P1, P2, P3) for k in range(3)])
        b1 = bars(P1, Z, 1, lambda v: fmt(v, 1))
        self.mark("smx"); self.play(FadeIn(P1.g), FadeIn(labs[:3]), FadeIn(b1), run_time=0.9); self.wait(1.0)
        b2 = bars(P2, E, 2, lambda v: fmt(v, 2))
        self.mark("smx"); self.play(FadeIn(P2.g), FadeIn(labs[3:6]), TransformFromCopy(b1, b2), run_time=1.1)
        s1 = T(f"sum {fmt(E.sum(), 2)}", 0.34, INK, BOLD).move_to([0.05, 2.1, 0])
        self.play(FadeIn(s1), run_time=0.4); self.wait(0.8)
        b3 = bars(P3, PR, 3, lambda v: fmt(100 * v, 0) + "%")
        self.mark("smx"); self.play(FadeIn(P3.g), FadeIn(labs[6:]), TransformFromCopy(b2, b3), run_time=1.1)
        self.wait(1.4)
        Zs = Z + 1; Es = np.exp(Zs)
        P2b = Panel(0, 3, 0, 15, (-1.8, 1.9, -2.2, 1.7), yt=(0, 5, 10, 15), yf=lambda v: fmt(v, 0), yl="2 · eᶻ")
        P1b = Panel(0, 3, 0, 3, (-6.3, -2.6, -2.2, 1.7), yt=(0, 1, 2, 3), yf=lambda v: fmt(v, 0), yl="1 · the z values")
        n1 = bars(P1b, Zs, 1, lambda v: fmt(v, 1)); n2 = bars(P2b, Es, 2, lambda v: fmt(v, 2))
        cap2 = self.caption("Adding the same to all of them changes nothing", "+1 to all three z: each eᶻ is multiplied by 2.7, and so is the sum")
        s2 = T(f"sum {fmt(Es.sum(), 2)}", 0.34, INK, BOLD).move_to(s1)
        self.mark("shift")
        self.play(*self.recap(cap, cap2), Transform(P1.g, P1b.g), Transform(b1, n1), Transform(P2.g, P2b.g), Transform(b2, n2), Transform(s1, s2), run_time=1.2)
        cap = cap2
        same = T("the same percentages", 0.36, GOOD, BOLD).move_to([4.75, -3.25, 0])
        self.play(Indicate(b3, color=GOOD, scale_factor=1.05), FadeIn(same), run_time=1.0); self.wait(1.8)
        self.clear_but(cap)

        # ---------- 3. the loss ----------
        cap = self.swap(cap, self.caption("The loss: a single number to bring down", "fuel use: the squared error (it was 33.0 L/100 km)"))
        Pm = Panel(20, 50, 0, 200, (-6.2, -0.6, -2.6, 1.6), xt=(20, 30, 40, 50), xl="prediction (L/100 km)", yt=(0, 100, 200), yl="penalty")
        par = Pm.curve(lambda v: (v - 33) ** 2, 20, 47, MEAN, 5)
        tr = ValueTracker(31.2)
        dm = always_redraw(lambda: Dot(Pm.P(tr.get_value(), (tr.get_value() - 33) ** 2), radius=0.11, color=POINT))
        lm = always_redraw(lambda: T(f"({fmt(tr.get_value())} − 33.0)² = {fmt((tr.get_value() - 33) ** 2, 1)}", 0.36, POINT, BOLD).move_to([-3.4, -3.45, 0]))
        self.mark("mse"); self.play(FadeIn(Pm.g), Create(par), FadeIn(dm), FadeIn(lm), run_time=1.0); self.wait(0.8)
        self.play(tr.animate.set_value(46.0), run_time=1.6)
        n1 = VGroup(T("13 liters of error:", 0.34, MUTED), T("penalty 169", 0.34, MUTED)).arrange(DOWN, aligned_edge=LEFT, buff=0.1).move_to([-3.4, 0.3, 0])
        self.play(FadeIn(n1), run_time=0.5); self.wait(0.8)
        Pc = Panel(0, 1, 0, 6, (0.9, 6.4, -2.6, 1.6), xt=(0, 0.5, 1), xf=lambda v: fmt(100 * v, 0) + "%", xl="probability given to the breakdown", yt=(0, 2, 4, 6), yl="penalty −ln p")
        ce = Pc.curve(lambda v: -math.log(max(v, 1e-9)), 0.0025, 1, WARN, 5, 400)
        tp = ValueTracker(p)
        dc = always_redraw(lambda: Dot(Pc.P(tp.get_value(), -math.log(tp.get_value())), radius=0.11, color=POINT))
        lc = always_redraw(lambda: T(f"−ln {fmt(100 * tp.get_value(), 1)}% = {fmt(-math.log(tp.get_value()), 2)}", 0.36, POINT, BOLD).move_to([3.65, -3.45, 0]))
        cap2 = self.caption("Confident and wrong is very expensive", "breakdown: the penalty is −ln of the probability it gave to what happened")
        self.mark("ce"); self.play(*self.recap(cap, cap2), FadeIn(Pc.g), Create(ce), FadeIn(dc), FadeIn(lc), run_time=1.2)
        cap = cap2; self.wait(1.0)
        self.mark("wrong"); self.play(tp.animate.set_value(0.01), run_time=1.8)
        n2 = VGroup(T("it gave 1% and it broke down:", 0.34, WARN, BOLD), T("4.61, no ceiling", 0.34, WARN, BOLD)).arrange(DOWN, aligned_edge=LEFT, buff=0.1).move_to([2.2, 0.9, 0], aligned_edge=LEFT)
        self.play(FadeIn(n2), run_time=0.5); self.wait(2.4)
        self.clear_but(cap)

        # ---------- 4. the push ----------
        cap = self.swap(cap, self.caption("The push: p − y", "a truck that did break down: what each loss corrects in z"))
        Pg = Panel(-8, 4, 0, 1.05, (-6.0, 1.4, -2.7, 1.6), xt=(-8, -4, 0, 4), xf=lambda v: fmt(v, 0), xl="z of the breakdown head", yt=(0, 0.5, 1), yf=lambda v: fmt(v, 1 if v % 1 else 0), yl="size of the push")
        gce = Pg.curve(lambda z: 1 - sig(z), -8, 4, MEAN, 5, 300)
        gms = Pg.curve(lambda z: 2 * (1 - sig(z)) ** 2 * sig(z), -8, 4, SAMPLE, 5, 300)
        self.mark("push"); self.play(FadeIn(Pg.g), Create(gce), run_time=1.0)
        l1 = VGroup(T("cross-entropy:", 0.36, MEAN, BOLD), T("1 − p", 0.36, MEAN, BOLD)).arrange(DOWN, aligned_edge=LEFT, buff=0.1).move_to([2.1, 1.0, 0], aligned_edge=LEFT)
        self.play(FadeIn(l1), run_time=0.4)
        l2 = VGroup(T("squared error:", 0.36, SAMPLE, BOLD), T("2(1 − p) · p(1 − p)", 0.36, SAMPLE, BOLD)).arrange(DOWN, aligned_edge=LEFT, buff=0.1).move_to([2.1, -0.4, 0], aligned_edge=LEFT)
        self.play(Create(gms), FadeIn(l2), run_time=1.0); self.wait(0.6)
        z0 = -4.6; p0 = float(sig(z0))
        d1 = Dot(Pg.P(z0, 1 - p0), radius=0.11, color=MEAN); d2 = Dot(Pg.P(z0, 2 * (1 - p0) ** 2 * p0), radius=0.11, color=SAMPLE)
        vl = DashedLine(Pg.P(z0, 0), Pg.P(z0, 1.05), color=POINT, stroke_width=2.5)
        n3 = VGroup(T(f"with p = 1%: {fmt(1 - p0, 2)} vs {fmt(2 * (1 - p0) ** 2 * p0, 2)}", 0.36, POINT, BOLD),
                    T("the square barely corrects", 0.34, MUTED), T("its worst mistake", 0.34, MUTED)).arrange(DOWN, aligned_edge=LEFT, buff=0.1).move_to([2.1, -2.1, 0], aligned_edge=LEFT)
        self.mark("dot"); self.play(Create(vl), FadeIn(d1, scale=1.6), FadeIn(d2, scale=1.6), run_time=0.7)
        self.play(FadeIn(n3), run_time=0.6); self.wait(2.6)
        self.clear_but(cap)

        # ---------- 5. the landscape ----------
        cap = self.swap(cap, self.caption("Averaged over the data, a landscape", "40 trucks: average loss according to the weight w of one neuron"))
        Pl = Panel(-8, 10, 0.2, 1.9, (-6.0, 1.6, -2.7, 1.6), xt=(-8, -4, 0, 4, 8), xf=lambda v: fmt(v, 0), xl="weight w", yt=(0.4, 0.8, 1.2, 1.6), yf=lambda v: fmt(v, 1), yl="average loss")
        land = Pl.curve(loss40, -8, 10, INK, 4.5, 300)
        self.mark("land"); self.play(FadeIn(Pl.g), Create(land), run_time=1.2)
        w = -6.0
        ball = Dot(Pl.P(w, loss40(w)), radius=0.14, color=POINT)
        info = VGroup(T("each step:", 0.36, INK, BOLD), T("w ← w − 10 · slope", 0.36, POINT, BOLD)).arrange(DOWN, aligned_edge=LEFT, buff=0.12).move_to([2.4, 1.0, 0], aligned_edge=LEFT)
        self.play(FadeIn(ball, scale=1.5), FadeIn(info), run_time=0.6)
        self.mark("steps")
        for _ in range(10):
            w = w - 10 * slope40(w)
            self.play(ball.animate.move_to(Pl.P(w, loss40(w))), run_time=0.4)
        fondo = VGroup(T(f"the bottom: w = {fmt(W40, 2)}", 0.36, GOOD, BOLD), T("more km, more breakdowns", 0.34, MUTED)).arrange(DOWN, aligned_edge=LEFT, buff=0.12).move_to([2.4, -0.6, 0], aligned_edge=LEFT)
        t5 = VGroup(T("walking this landscape down", 0.34, MUTED), T("with thousands of weights:", 0.34, MUTED), T("backpropagation (topic 5)", 0.34, MUTED)).arrange(DOWN, aligned_edge=LEFT, buff=0.1).move_to([2.4, -2.0, 0], aligned_edge=LEFT)
        self.play(FadeIn(fondo), run_time=0.6); self.play(FadeIn(t5), run_time=0.6); self.wait(2.2)
        self.clear_but(cap)

        # ---------- Closing ----------
        self.mark("legend")
        self.play(FadeOut(cap), run_time=0.6)
        e1 = Text("Answer in its language, punish with a number", font=FONT, weight=BOLD, color=INK).scale(0.62)
        e2 = VGroup(T("quantities: linear output and squared error", 0.42, MUTED),
                    T("yes or no: sigmoid; several options: softmax", 0.42, MUTED),
                    T("probabilities: cross-entropy, push p − y", 0.42, MUTED),
                    T("the average loss is the landscape training walks down", 0.42, MUTED)).arrange(DOWN, buff=0.18)
        VGroup(e1, e2).arrange(DOWN, buff=0.45)
        self.mark("end_text"); self.play(Write(e1), run_time=1.2); self.play(FadeIn(e2, shift=0.2 * UP, lag_ratio=0.15), run_time=1.0)
        self.wait(3.3)
        self.mark("fade"); self.play(FadeOut(VGroup(e1, e2)), run_time=1)
        self.marks["total"] = [round(self.renderer.time, 3)]
        json.dump(self.marks, open("marks.json", "w"), indent=1)
