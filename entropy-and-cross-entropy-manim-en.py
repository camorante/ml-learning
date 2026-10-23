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


def fmt(v, d=2):
    s = f"{abs(v):.{d}f}"
    return ("−" if v < 0 and round(abs(v), d) != 0 else "") + s


def H2(p):
    return 0.0 if p <= 0 or p >= 1 else -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


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
        return VMobject(stroke_color=col, stroke_width=w).set_points_smoothly([self.P(x, f(x)) for x in np.linspace(a, b, n)])


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
        title = Text("Entropy and cross-entropy", font=FONT, weight=BOLD, color=INK).scale(0.95)
        sub = Text("measuring surprise: the loss of classifiers", font=FONT, color=MUTED).scale(0.48)
        VGroup(title, sub).arrange(DOWN, buff=0.35)
        self.mark("title"); self.play(Write(title), run_time=1.3)
        self.mark("subtitle"); self.play(FadeIn(sub, shift=0.2 * UP)); self.wait(1.4)
        self.mark("title_out"); self.play(FadeOut(title), FadeOut(sub))

        # ---------- 1. surprise: yes-or-no questions ----------
        cap = self.swap(None, self.caption("Surprise is measured in yes-or-no questions", "one of 16 trucks sends an alert: which one?"))
        cw, ch, gx, gy = 1.05, 0.95, -6.0, 1.75
        trucks = VGroup()
        for i in range(16):
            c, r = i % 4, i // 4
            col = POINT if i == 11 else LINE
            trucks.add(VGroup(RoundedRectangle(width=0.62, height=0.38, corner_radius=0.06, fill_color=col, fill_opacity=1, stroke_width=0),
                              Dot([-0.16, -0.25, 0], radius=0.06, color=col if i == 11 else MUTED), Dot([0.16, -0.25, 0], radius=0.06, color=col if i == 11 else MUTED))
                          .move_to([gx + (c + 0.5) * cw, gy - (r + 0.5) * ch, 0]))
        self.play(LaggedStart(*[FadeIn(t, scale=0.8) for t in trucks], lag_ratio=0.04), run_time=1.0)
        # bisection: columns 0-4 → 2-4 (vertical), rows 0-4 → 2-4, columns 2-4 → 3-4, rows 2-4 → 2-3
        c0, c1, r0, r1 = 0, 4, 0, 4
        X = lambda c: gx + c * cw; Y = lambda r: gy - r * ch
        steps = [("v", 2, "c"), ("h", 2, "r"), ("v", 3, "c"), ("h", 3, "r")]
        counter = T("16", 0.9, INK, BOLD).move_to([3.6, 1.0, 0])
        clab = T("possible trucks", 0.36, MUTED).next_to(counter, DOWN, buff=0.15)
        self.play(FadeIn(counter), FadeIn(clab), run_time=0.4)
        n = 16
        for q, (kind, m, _) in enumerate(steps, 1):
            if kind == "v":
                ln = DashedLine([X(m), Y(r0), 0], [X(m), Y(r1), 0], color=MEAN, stroke_width=4)
                c0 = m
            else:
                ln = DashedLine([X(c0), Y(m), 0], [X(c1), Y(m), 0], color=MEAN, stroke_width=4)
                r1 = m if q == 4 else r1
                r0 = m if q == 2 else r0
            n //= 2
            badge = VGroup(Circle(radius=0.17, fill_color=MEAN, fill_opacity=1, stroke_color=BG, stroke_width=3), T(str(q), 0.3, BG, BOLD))
            a0, a1 = (np.array([X(m), Y(r0 if q != 4 else 2), 0]), np.array([X(m), Y(4), 0])) if kind == "v" else (np.array([X(c0), Y(m), 0]), np.array([X(c1), Y(m), 0]))
            badge.move_to(a0 + 0.3 * (a1 - a0))
            fade = [t.animate.set_opacity(0.25) for i, t in enumerate(trucks) if i != 11 and not (c0 <= i % 4 < c1 and r0 <= i // 4 < r1)]
            nc = T(str(n), 0.9, INK if n > 1 else POINT, BOLD).move_to(counter)
            self.mark("q"); self.play(Create(ln), FadeIn(badge), Transform(counter, nc), *fade, run_time=0.6)
        res = T("4 questions = 4 bits", 0.55, POINT, BOLD).move_to([3.6, -0.2, 0])
        f1 = T("log₂(1/p): 1 in 16 → 4 bits", 0.38, INK).next_to(res, DOWN, buff=0.3)
        self.mark("bits"); self.play(FadeIn(res), FadeIn(f1), run_time=0.6)
        rows = VGroup(T("no incidents (90%): 0.15 bits", 0.34, MUTED), T("accident (0.5%): 7.64 bits", 0.34, SAMPLE, BOLD)).arrange(DOWN, aligned_edge=LEFT, buff=0.15).next_to(f1, DOWN, buff=0.35)
        self.play(FadeIn(rows, shift=0.1 * UP), run_time=0.6); self.wait(3.4)
        self.clear_but(cap)

        # ---------- 2. entropy: average surprise ----------
        cap = self.swap(cap, self.caption("Entropy: the average surprise", "will the delivery be late? · fair coin versus loaded coin"))
        Pe = Panel(0, 1, 0, 1.1, (-6.0, 1.2, -2.9, 1.9), xt=(0, 0.25, 0.5, 0.75, 1), xf=lambda v: f"{int(v * 100)}%", xl="deliveries that are late",
                   yt=(0, 0.5, 1), yf=lambda v: fmt(v, 1), yl="bits")
        cv = Pe.curve(H2, 0.0005, 0.9995, MEAN, 4, 300)
        self.play(FadeIn(Pe.g), Create(cv), run_time=1.0)
        p = ValueTracker(0.5)
        dot = always_redraw(lambda: Dot(Pe.P(p.get_value(), H2(p.get_value())), radius=0.12, color=POINT))
        val = always_redraw(lambda: T(f"{fmt(H2(p.get_value()), 2)} bits", 0.7, POINT, BOLD).move_to([4.3, 1.0, 0]))
        pl = always_redraw(lambda: T(f"late: {int(round(p.get_value() * 100))}%", 0.4, INK).move_to([4.3, 0.25, 0]))
        self.add(dot, val, pl)
        t1 = T("fair coin: 1 bit", 0.36, MUTED).move_to([4.3, -0.5, 0])
        self.mark("coin"); self.play(FadeIn(t1), run_time=0.4); self.wait(1.4)
        self.play(p.animate.set_value(0.1), run_time=2.0, rate_func=smooth)
        t2 = T("loaded (10%): 0.47 bits", 0.36, MUTED).move_to([4.3, -1.0, 0])
        self.play(FadeIn(t2), run_time=0.4); self.wait(1.2)
        t3 = VGroup(T("the fleet report (90% no incidents):", 0.32, INK), T("0.59 bits out of a possible 2", 0.42, GOOD, BOLD)).arrange(DOWN, buff=0.12).move_to([4.3, -2.0, 0])
        self.mark("flota"); self.play(FadeIn(t3, shift=0.1 * UP), run_time=0.6); self.wait(3.0)
        self.remove(dot, val, pl)
        self.clear_but(cap)

        # ---------- 3. Huffman ----------
        cap = self.swap(cap, self.caption("Compressing: short codes for frequent things", "telematics box statuses · Huffman code"))
        est = [("driving", 0.55, "1"), ("stuck in traffic", 0.20, "00"), ("unloading", 0.12, "010"),
               ("refueling", 0.07, "0111"), ("at the shop", 0.04, "01101"), ("alert", 0.02, "01100")]
        rows = VGroup()
        for i, (nm, pr, code) in enumerate(est):
            y = 1.7 - i * 0.68
            name = T(nm, 0.38, INK).move_to([-4.6, y, 0]).align_to([-6.3, 0, 0], LEFT)
            bar = Rectangle(width=max(0.03, 3.4 * pr), height=0.3, fill_color=MEAN, fill_opacity=0.85, stroke_width=0).move_to([-3.2, y, 0], aligned_edge=LEFT)
            pct = T(f"{int(round(pr * 100))}%", 0.3, MUTED).next_to(bar, RIGHT, buff=0.12)
            bits = VGroup(*[VGroup(RoundedRectangle(width=0.34, height=0.34, corner_radius=0.05, fill_color=POINT, fill_opacity=0.9, stroke_width=0), T(b, 0.3, BG, BOLD)) for b in code]).arrange(RIGHT, buff=0.06)
            bits.move_to([0.3, y, 0], aligned_edge=LEFT)
            rows.add(VGroup(name, bar, pct, bits))
        hdr = VGroup(T("frequency", 0.3, MUTED).move_to([-3.2, 2.35, 0], aligned_edge=LEFT), T("Huffman code", 0.3, POINT).move_to([0.3, 2.35, 0], aligned_edge=LEFT))
        self.play(FadeIn(hdr), run_time=0.3)
        for r in rows:
            self.mark("codes"); self.play(FadeIn(r[:3], shift=0.1 * RIGHT), FadeIn(r[3], shift=0.1 * LEFT), run_time=0.42)
        sm = VGroup(T("fixed code: 3 bits", 0.4, MUTED), T("Huffman: 1.89 bits", 0.48, POINT, BOLD), T("entropy: 1.87 (the minimum)", 0.4, MEAN, BOLD)).arrange(DOWN, aligned_edge=LEFT, buff=0.2).move_to([2.95, 0.0, 0], aligned_edge=LEFT)
        self.mark("sum"); self.play(FadeIn(sm, lag_ratio=0.3), run_time=1.0); self.wait(3.4)
        self.clear_but(cap)

        # ---------- 4. cross-entropy and KL ----------
        cap = self.swap(cap, self.caption("Cross-entropy: paying with the wrong code", "winter (data) measured with the whole-year frequencies (model)"))
        tipos = [("no incidents", 0.78, 0.90, 0.152), ("delay", 0.12, 0.06, 4.059), ("breakdown", 0.08, 0.035, 4.837), ("accident", 0.02, 0.005, 7.644)]
        grp = VGroup()
        for i, (nm, pv, qv, s) in enumerate(tipos):
            y = 1.75 - i * 0.82
            name = T(nm, 0.36, INK).move_to([-6.3, y, 0], aligned_edge=LEFT)
            bp = Rectangle(width=max(0.04, 5 * pv), height=0.22, fill_color=POINT, fill_opacity=0.9, stroke_width=0).move_to([-3.6, y + 0.13, 0], aligned_edge=LEFT)
            bq = Rectangle(width=max(0.04, 5 * qv), height=0.22, fill_color=MEAN, fill_opacity=0.9, stroke_width=0).move_to([-3.6, y - 0.13, 0], aligned_edge=LEFT)
            cs = T(f"{fmt(s, 2)} bits", 0.36, WARN if s > 7 else INK, BOLD).move_to([3.0, y, 0], aligned_edge=RIGHT)
            grp.add(VGroup(name, bp, bq, cs))
        lg = VGroup(T("data", 0.3, POINT, BOLD), T("model", 0.3, MEAN, BOLD), T("cost each time", 0.3, MUTED)).arrange(RIGHT, buff=0.5).move_to([-1.0, 2.45, 0])
        self.play(FadeIn(lg), LaggedStart(*[FadeIn(g, shift=0.1 * RIGHT) for g in grp], lag_ratio=0.2), run_time=1.4)
        sc = 2.4
        b1 = Rectangle(width=sc * 1.051, height=0.32, fill_color=POINT, fill_opacity=0.75, stroke_width=0).move_to([-3.6, -1.85, 0], aligned_edge=LEFT)
        b2 = Rectangle(width=sc * 1.051, height=0.32, fill_color=POINT, fill_opacity=0.75, stroke_width=0).move_to([-3.6, -2.45, 0], aligned_edge=LEFT)
        b3 = Rectangle(width=sc * 0.094, height=0.32, fill_color=WARN, fill_opacity=1, stroke_width=0).next_to(b2, RIGHT, buff=0)
        l1 = T("entropy 1.05", 0.34, INK).next_to(b1, LEFT, buff=0.2)
        l2 = T("cross-entropy 1.15", 0.34, INK).next_to(b2, LEFT, buff=0.2)
        self.mark("kl"); self.play(FadeIn(b1), FadeIn(l1), FadeIn(b2), FadeIn(l2), run_time=0.6)
        self.play(GrowFromEdge(b3, LEFT), run_time=0.6)
        kl = VGroup(T("KL = +0.094 bits per report", 0.42, WARN, BOLD), T("reversed: 0.074 · not symmetric", 0.34, MUTED)).arrange(DOWN, aligned_edge=LEFT, buff=0.15).move_to([4.2, -2.1, 0])
        self.play(FadeIn(kl, shift=0.1 * UP), run_time=0.6); self.wait(3.4)
        self.clear_but(cap)

        # ---------- 5. log loss ----------
        cap = self.swap(cap, self.caption("The loss of classifiers", "penalty = −ln(probability the model gave to what happened)"))
        Pl = Panel(0, 1, 0, 5.2, (-6.0, 1.4, -2.9, 1.9), xt=(0, 0.25, 0.5, 0.75, 1), xf=lambda v: f"{int(v * 100)}%", xl="probability given to what happened",
                   yt=(0, 1, 2, 3, 4, 5), yl="penalty")
        c1 = Pl.curve(lambda x: min(5.2, -math.log(x)), 0.0056, 1, WARN, 4, 300)
        c2 = Pl.curve(lambda x: (1 - x) ** 2, 0, 1, SAMPLE, 4, 100)
        self.play(FadeIn(Pl.g), Create(c1), Create(c2), run_time=1.0)
        lab1 = T("log loss", 0.32, WARN, BOLD).move_to(Pl.P(0.42, 1.55))
        lab2 = T("squared error", 0.32, SAMPLE, BOLD).move_to(Pl.P(0.22, 0.22))
        self.play(FadeIn(lab1), FadeIn(lab2), run_time=0.4)
        q = ValueTracker(0.9)
        d1 = always_redraw(lambda: Dot(Pl.P(q.get_value(), min(5.2, -math.log(q.get_value()))), radius=0.11, color=WARN))
        d2 = always_redraw(lambda: Dot(Pl.P(q.get_value(), (1 - q.get_value()) ** 2), radius=0.1, color=SAMPLE))
        rd = always_redraw(lambda: VGroup(T(f"said {int(round(q.get_value() * 100))}%", 0.4, INK),
                                          T(f"log loss {fmt(-math.log(q.get_value()), 2)}", 0.5, WARN, BOLD),
                                          T(f"squared {fmt((1 - q.get_value()) ** 2, 2)}", 0.42, SAMPLE, BOLD)).arrange(DOWN, aligned_edge=LEFT, buff=0.15).move_to([4.5, 0.9, 0]))
        self.add(d1, d2, rd)
        self.mark("conf"); self.play(q.animate.set_value(0.01), run_time=2.6, rate_func=smooth)
        msg = VGroup(T("confident and wrong: very expensive", 0.36, INK, BOLD), T("minimizing it = maximum likelihood", 0.34, MUTED),
                     T("and it rewards honest probabilities", 0.34, MUTED)).arrange(DOWN, aligned_edge=LEFT, buff=0.15).move_to([4.4, -1.4, 0])
        self.play(FadeIn(msg, shift=0.1 * UP), run_time=0.7); self.wait(3.3)
        self.remove(d1, d2, rd)
        self.clear_but(cap)

        # ---------- 6. tree ----------
        cap = self.swap(cap, self.caption("Trees: the question that clarifies the most", "1,000 trucks, 251 breakdowns · gain = entropy before − after"))
        W = 11.0

        def bar(x, y, w, n, a):
            fa = a / n
            return VGroup(Rectangle(width=w * fa, height=0.5, fill_color=POINT, fill_opacity=0.9, stroke_width=0).move_to([x, y, 0], aligned_edge=LEFT),
                          Rectangle(width=w * (1 - fa), height=0.5, fill_color=MUTED, fill_opacity=0.35, stroke_width=0).move_to([x + w * fa, y, 0], aligned_edge=LEFT))
        top = bar(-5.5, 1.6, W, 1000, 251)
        tl = T("all: entropy 0.81 bits", 0.36, MEAN, BOLD).next_to(top, DOWN, buff=0.15).align_to(top, LEFT)
        self.play(FadeIn(top), FadeIn(tl), run_time=0.6)
        qs = T("engine outside 80–100 °C?", 0.42, INK, BOLD).move_to([0, 0.55, 0])
        self.mark("split"); self.play(FadeIn(qs), run_time=0.5)
        ws, wn = (W - 0.3) * 0.202, (W - 0.3) * 0.798
        bs = bar(-5.5, -0.45, ws, 202, 160); bn = bar(-5.5 + ws + 0.3, -0.45, wn, 798, 91)
        ls = VGroup(T("yes: 79% breakdowns", 0.3, INK), T("entropy 0.74", 0.3, MEAN, BOLD)).arrange(DOWN, aligned_edge=LEFT, buff=0.08).next_to(bs, DOWN, buff=0.15).align_to(bs, LEFT)
        ln = VGroup(T("no: 11% breakdowns", 0.3, INK), T("entropy 0.51", 0.3, MEAN, BOLD)).arrange(DOWN, aligned_edge=RIGHT, buff=0.08).next_to(bn, DOWN, buff=0.15).align_to(bn, RIGHT)
        self.play(TransformFromCopy(top, VGroup(bs, bn)), run_time=0.9)
        self.play(FadeIn(ls), FadeIn(ln), run_time=0.5)
        gain = VGroup(T("gain: 0.813 − 0.558 = 0.255 bits", 0.48, GOOD, BOLD), T("\"more than 400,000 km?\" gains only 0.04 · correlation with temperature: 0.007", 0.32, MUTED)).arrange(DOWN, buff=0.18).move_to([0, -2.6, 0])
        self.mark("gain"); self.play(FadeIn(gain, shift=0.1 * UP), run_time=0.7); self.wait(3.6)
        self.clear_but(cap)

        # ---------- Closing ----------
        self.mark("legend")
        self.play(FadeOut(cap), run_time=0.6)
        e1 = Text("The average surprise measures uncertainty", font=FONT, weight=BOLD, color=INK).scale(0.66)
        e2 = VGroup(T("surprise: log₂(1/p) · entropy: its average, the minimum number of bits", 0.42, MUTED),
                    T("cross-entropy = entropy + KL: the price of a wrong model", 0.42, MUTED),
                    T("log loss: punishes false confidence and rewards honesty", 0.42, MUTED),
                    T("information gain: the question that clarifies the most", 0.42, MUTED)).arrange(DOWN, buff=0.18)
        VGroup(e1, e2).arrange(DOWN, buff=0.45)
        self.mark("end_text"); self.play(Write(e1), run_time=1.2); self.play(FadeIn(e2, shift=0.2 * UP, lag_ratio=0.15), run_time=1.0)
        self.wait(4.2)
        self.mark("fade"); self.play(FadeOut(VGroup(e1, e2)), run_time=1)
        self.marks["total"] = [round(self.renderer.time, 3)]
        json.dump(self.marks, open("marks.json", "w"), indent=1)
