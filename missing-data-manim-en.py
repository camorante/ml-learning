import json
import numpy as np
from manim import *
from scipy.stats import beta as Beta

# Dark palette of the visual ML series
BG = "#0E131B"; INK = "#E4EAF3"; MUTED = "#9BA7B9"; LINE = "#2B3545"
MEAN = "#6EA6FF"; SAMPLE = "#AE9FF3"; POINT = "#FF9A55"; GOOD = "#4CC38A"; WARN = "#F08A5D"
FONT = "Inter"
config.background_color = BG


class Faltantes(Scene):
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

    def construct(self):
        self.marks = {}
        title = Text("Missing data", font=FONT, weight=BOLD, color=INK).scale(1.3)
        sub = Text("the question is not how many are missing, but why", font=FONT, color=MUTED).scale(0.5)
        VGroup(title, sub).arrange(DOWN, buff=0.35)
        self.mark("title"); self.play(Write(title), run_time=1.2)
        self.mark("subtitle"); self.play(FadeIn(sub, shift=0.2 * UP)); self.wait(1.2)
        self.mark("title_out"); self.play(FadeOut(title), FadeOut(sub))
        rng = np.random.default_rng(3)

        # ---------- 1. gaps ----------
        cap = self.swap(None, self.caption("2,000 deliveries, 600 waits not logged", "each square is a delivery; the empty ones, those the driver didn't log"))
        cols, rows, cs = 40, 15, 0.27
        miss = rng.random(cols * rows) < 0.3
        sq = VGroup(*[Square(cs * 0.8, stroke_width=1.5 if miss[i] else 0, stroke_color=MUTED, fill_color=SAMPLE, fill_opacity=0 if miss[i] else 0.85).move_to([-cols * cs / 2 + cs / 2 + (i % cols) * cs, 1.6 - (i // cols) * cs, 0]) for i in range(cols * rows)])
        self.mark("grid"); self.play(FadeIn(sq, lag_ratio=0.001), run_time=1.4); self.wait(1.4)
        self.play(FadeOut(sq), run_time=0.6)

        # ---------- 2. three reasons ----------
        cap = self.swap(cap, self.caption("Three reasons a value goes missing", "and each one is handled differently"))
        def card(t1, t2, t3, col, x):
            box = RoundedRectangle(width=3.9, height=2.8, corner_radius=0.15, stroke_color=col, stroke_width=3, fill_color=BG, fill_opacity=1).move_to([x, 0, 0])
            a = Text(t1, font=FONT, weight=BOLD, color=col).scale(0.5); b = Text(t2, font=FONT, color=INK).scale(0.34); c = Text(t3, font=FONT, color=MUTED).scale(0.32)
            VGroup(a, b, c).arrange(DOWN, buff=0.25).move_to(box)
            return VGroup(box, a, b, c)
        c1 = card("By chance", "the app fails", "dropping rows adds no bias", GOOD, -4.3)
        c2 = card("By something visible", "stores don't get logged", "fill by customer type", MEAN, 0)
        c3 = card("By the value itself", "short waits don't get logged", "you need outside information", WARN, 4.3)
        for i, c in enumerate([c1, c2, c3]):
            self.mark("card"); self.play(FadeIn(c, shift=0.2 * UP), run_time=0.7); self.wait(0.5)
        self.wait(1.4); self.play(FadeOut(VGroup(c1, c2, c3)), run_time=0.6)

        # ---------- 3. mean ----------
        cap = self.swap(cap, self.caption("Filling with the average flattens the data", "600 deliveries with exactly the same wait"))
        Xh = lambda v: -6.0 + v / 150 * 12.0; base = -2.6
        vals = np.clip(np.r_[np.exp(rng.normal(np.log(38), 0.5, 1400))], 0, 149); edges = np.arange(0, 151, 5); cnt, _ = np.histogram(vals, edges); sc = 3.6 / 420
        bars = VGroup(*[Rectangle(width=12 * 5 / 150 * 0.85, height=max(0.001, c * sc), stroke_width=0, fill_color=SAMPLE, fill_opacity=0.85).move_to([Xh(e + 2.5), base + c * sc / 2, 0]) for e, c in zip(edges[:-1], cnt) if c > 0])
        k = int(np.mean(vals) // 5); spike = Rectangle(width=12 * 5 / 150 * 0.85, height=600 * sc, stroke_width=0, fill_color=POINT, fill_opacity=1).move_to([Xh(k * 5 + 2.5), base + cnt[k] * sc + 600 * sc / 2, 0])
        self.mark("hist"); self.play(*[GrowFromEdge(b, DOWN) for b in bars], run_time=1.0)
        self.mark("spike"); self.play(GrowFromEdge(spike, DOWN), run_time=0.9)
        lb = Text("less variation, weaker relationships, no long waits", font=FONT, color=WARN).scale(0.38).move_to([2.5, 1.8, 0])
        self.play(FadeIn(lb), run_time=0.5); self.wait(1.6)
        self.play(FadeOut(VGroup(bars, spike, lb)), run_time=0.6)

        # ---------- 4. gaps over time ----------
        cap = self.swap(cap, self.caption("Gaps over time", "what happened while there was no signal?"))
        Xt = lambda m: -5.5 + m / 240 * 11.0; Yt = lambda v: -2.4 + (v - 40) / 50 * 4.0
        lv = lambda m: 62 - 0.11 * m + (30 * min(1, max(0, (m - 100) / 10)) if m >= 100 else 0)
        gap = Rectangle(width=Xt(120) - Xt(80), height=4.4, stroke_width=0, fill_color=MUTED, fill_opacity=0.15).move_to([(Xt(80) + Xt(120)) / 2, -0.3, 0])
        a1 = VMobject(stroke_color=MEAN, stroke_width=5).set_points_as_corners([[Xt(m), Yt(lv(m)), 0] for m in range(0, 81, 5)])
        a2 = VMobject(stroke_color=MEAN, stroke_width=5).set_points_as_corners([[Xt(m), Yt(lv(m)), 0] for m in range(120, 241, 5)])
        lin = Line([Xt(80), Yt(lv(80)), 0], [Xt(120), Yt(lv(120)), 0], color=POINT, stroke_width=5)
        real = DashedVMobject(VMobject(stroke_color=GOOD, stroke_width=4).set_points_as_corners([[Xt(m), Yt(lv(m)), 0] for m in np.arange(80, 121, 1)]), num_dashes=14)
        self.mark("ts"); self.play(FadeIn(gap), Create(a1), Create(a2), run_time=1.0)
        l1 = Text("straight line: the tank \"fills up\" while driving", font=FONT, color=POINT).scale(0.38).move_to([0, 2.0, 0])
        self.mark("lin"); self.play(Create(lin), FadeIn(l1), run_time=0.8); self.wait(0.8)
        l2 = Text("it actually refueled: better to leave the gap flagged", font=FONT, weight=BOLD, color=GOOD).scale(0.4).move_to([0, 2.0, 0])
        self.mark("real"); self.play(Create(real), FadeOut(lin), ReplacementTransform(l1, l2), run_time=1.0); self.wait(1.6)
        self.play(FadeOut(VGroup(gap, a1, a2, real, l2)), run_time=0.6)

        # ---------- 5. the gap is a warning ----------
        cap = self.swap(cap, self.caption("Being missing is information too", "the sensor stops sending when the electrical system fails"))
        def vbar(x, v, lab, col):
            h = v * 10
            return VGroup(Rectangle(width=1.6, height=h, stroke_width=0, fill_color=col, fill_opacity=1).move_to([x, -2.4 + h / 2, 0]),
                          Text(f"{v * 100:.0f}%", font=FONT, weight=BOLD, color=INK).scale(0.5).move_to([x, -2.4 + h + 0.35, 0]),
                          Text(lab, font=FONT, color=MUTED).scale(0.34).move_to([x, -2.75, 0]))
        b1 = vbar(-2.0, 0.069, "has temperature", SAMPLE); b2 = vbar(2.0, 0.30, "temperature missing", WARN)
        tt = Text("trips with a breakdown in the following month", font=FONT, color=INK).scale(0.4).move_to([0, 2.05, 0])
        self.mark("bars"); self.play(FadeIn(tt), GrowFromEdge(b1[0], DOWN), FadeIn(b1[1:]), run_time=0.8); self.play(GrowFromEdge(b2[0], DOWN), FadeIn(b2[1:]), run_time=0.8)
        ind = Text("add a \"was missing\" column: the model uses it", font=FONT, weight=BOLD, color=GOOD).scale(0.4).move_to([0, 1.55, 0])
        self.mark("ind"); self.play(FadeIn(ind), run_time=0.6); self.wait(1.8)
        self.play(*[FadeOut(m) for m in self.mobjects if m is not cap], run_time=0.6)

        # ---------- Closing ----------
        self.mark("legend")
        self.play(FadeOut(cap), run_time=0.6)
        e1 = Text("Before filling in, ask why it is missing.", font=FONT, weight=BOLD, color=INK).scale(0.7)
        e2 = Text("And always record which values you made up.", font=FONT, color=MUTED).scale(0.48)
        VGroup(e1, e2).arrange(DOWN, buff=0.4)
        self.mark("end_text"); self.play(Write(e1), run_time=1.4); self.play(FadeIn(e2, shift=0.2 * UP), run_time=0.8)
        self.wait(2.0)
        self.mark("fade"); self.play(FadeOut(VGroup(e1, e2)), run_time=1)
        self.marks["total"] = [round(self.renderer.time, 3)]
        json.dump(self.marks, open("marks.json", "w"), indent=1)
