import json
import numpy as np
from manim import *

# Dark palette of the visual ML series
BG = "#0E131B"; INK = "#E4EAF3"; MUTED = "#9BA7B9"; LINE = "#2B3545"
MEAN = "#6EA6FF"; SAMPLE = "#AE9FF3"; POINT = "#FF9A55"; GOOD = "#4CC38A"; WARN = "#F08A5D"
FONT = "Inter"
config.background_color = BG


class Variables(Scene):
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
        title = Text("Preparing features", font=FONT, weight=BOLD, color=INK).scale(1.2)
        sub = Text("the model only sees columns of numbers", font=FONT, color=MUTED).scale(0.5)
        VGroup(title, sub).arrange(DOWN, buff=0.35)
        self.mark("title"); self.play(Write(title), run_time=1.2)
        self.mark("subtitle"); self.play(FadeIn(sub, shift=0.2 * UP)); self.wait(1.2)
        self.mark("title_out"); self.play(FadeOut(title), FadeOut(sub))

        # ---------- 1. kilograms vs kilometers ----------
        cap = self.swap(None, self.caption("Kilograms vs kilometers", "differences are added as they are: the biggest number wins"))
        def rng_bar(y, w, lab, val, col):
            r = Rectangle(width=w, height=0.55, stroke_width=0, fill_color=col, fill_opacity=1).move_to([-5.5 + w / 2, y, 0])
            return VGroup(Text(lab, font=FONT, color=MUTED).scale(0.38).next_to(r, UP, buff=0.12).align_to(r, LEFT), r,
                          Text(val, font=FONT, weight=BOLD, color=INK).scale(0.4).next_to(r, RIGHT, buff=0.2))
        b1 = rng_bar(0.9, 0.25, "distance", "up to 600 km", MEAN); b2 = rng_bar(-0.7, 10.0, "load", "up to 24,000 kg", POINT)
        self.mark("bars"); self.play(FadeIn(b1[0]), GrowFromEdge(b1[1], LEFT), FadeIn(b1[2]), run_time=0.6)
        self.play(FadeIn(b2[0]), GrowFromEdge(b2[1], LEFT), FadeIn(b2[2]), run_time=1.0); self.wait(0.8)
        n1 = rng_bar(0.9, 5.0, "distance, scaled", "−1 … 3", MEAN); n2 = rng_bar(-0.7, 5.0, "load, scaled", "−1 … 2", POINT)
        sc = Text("scaling: each column on its own scale", font=FONT, weight=BOLD, color=GOOD).scale(0.45).move_to([0, -2.3, 0])
        self.mark("scale"); self.play(Transform(b1, n1), Transform(b2, n2), FadeIn(sc), run_time=1.0); self.wait(1.6)
        self.play(FadeOut(VGroup(b1, b2, sc)), run_time=0.6)

        # ---------- 2. categories ----------
        cap = self.swap(cap, self.caption("Categories are not numbers", "numbering them invents an order that does not exist"))
        tipos = ["refrigerated", "van", "rigid", "semi-trailer"]
        rows = VGroup(*[VGroup(Text(t, font=FONT, color=INK).scale(0.42).move_to([-4.0, 0.9 - i * 0.6, 0], aligned_edge=LEFT),
                               Text(str(i + 1), font=FONT, weight=BOLD, color=WARN).scale(0.5).move_to([-1.3, 0.9 - i * 0.6, 0])) for i, t in enumerate(tipos)])
        self.mark("nums"); self.play(FadeIn(rows, lag_ratio=0.15), run_time=1.0)
        cross = Cross(VGroup(*[r[1] for r in rows]), stroke_color=WARN, stroke_width=5)
        self.play(Create(cross), run_time=0.5)
        grid = VGroup()
        for i in range(4):
            for j in range(4):
                sq = Square(0.5, stroke_color=LINE, stroke_width=2, fill_color=GOOD if i == j else BG, fill_opacity=1 if i == j else 0)
                grid.add(VGroup(sq, Text("1" if i == j else "0", font=FONT, weight=BOLD, color=BG if i == j else MUTED).scale(0.36)).move_to([1.2 + j * 0.6, 0.9 - i * 0.6, 0]))
        lab = Text("one column per type", font=FONT, weight=BOLD, color=GOOD).scale(0.42).move_to([2.1, 1.6, 0])
        self.mark("onehot"); self.play(FadeIn(grid, lag_ratio=0.03), FadeIn(lab), run_time=1.0); self.wait(1.8)
        self.play(FadeOut(VGroup(rows, cross, grid, lab)), run_time=0.6)

        # ---------- 3. customers ----------
        cap = self.swap(cap, self.caption("A thousand customers: their average, carefully", "a single trip says nothing about the customer"))
        ax = NumberLine(x_range=[5, 35, 5], length=10, color=MUTED, include_numbers=False).move_to([0, -0.8, 0])
        ticks = VGroup(*[Text(f"{v} L", font=FONT, color=MUTED).scale(0.32).next_to(ax.n2p(v), DOWN, buff=0.2) for v in range(5, 36, 5)])
        g = DashedLine(ax.n2p(25) + 1.8 * UP, ax.n2p(25), color=MUTED); gl = Text("fleet average", font=FONT, color=MUTED).scale(0.32).next_to(g, UP, buff=0.1)
        raw = Circle(0.2, color=POINT, stroke_width=5).move_to(ax.n2p(8.1) + 0.6 * UP); rl = Text("1 trip: 8.1 L", font=FONT, color=POINT).scale(0.38).next_to(raw, UP, buff=0.15)
        self.mark("cli"); self.play(Create(ax), FadeIn(ticks), Create(g), FadeIn(gl), run_time=0.8); self.play(FadeIn(raw), FadeIn(rl), run_time=0.5); self.wait(0.5)
        adj = Dot(ax.n2p(23.46) + 0.6 * UP, radius=0.17, color=MEAN); al = Text("adjusted: 23.5 L", font=FONT, weight=BOLD, color=MEAN).scale(0.38).next_to(adj, UP, buff=0.2).shift(0.9 * LEFT)
        arr = Arrow(raw.get_center(), adj.get_center(), buff=0.25, color=INK, stroke_width=4)
        self.mark("shrink"); self.play(GrowArrow(arr), FadeIn(adj), FadeIn(al), run_time=0.9)
        nt = Text("and computed without the trip itself", font=FONT, weight=BOLD, color=GOOD).scale(0.42).move_to([0, 2.0, 0])
        self.play(FadeIn(nt), run_time=0.5); self.wait(1.6)
        self.play(FadeOut(VGroup(ax, ticks, g, gl, raw, rl, adj, al, arr, nt)), run_time=0.6)

        # ---------- 4. time of day ----------
        cap = self.swap(cap, self.caption("Time of day is a circle", "11 PM and 1 AM are two hours apart, not twenty-two"))
        line = NumberLine(x_range=[0, 24, 3], length=11, color=MUTED, include_numbers=False).move_to([0, -0.3, 0])
        lt = VGroup(*[Text(f"{h}", font=FONT, color=MUTED).scale(0.32).next_to(line.n2p(h), DOWN, buff=0.2) for h in range(0, 25, 3)])
        p23 = Dot(line.n2p(23), radius=0.15, color=POINT); p1 = Dot(line.n2p(1), radius=0.15, color=POINT)
        br = DoubleArrow(line.n2p(1) + 0.6 * UP, line.n2p(23) + 0.6 * UP, buff=0, color=WARN, stroke_width=3, tip_length=0.2)
        bl = Text("22 apart", font=FONT, color=WARN).scale(0.38).next_to(br, UP, buff=0.1)
        self.mark("line"); self.play(Create(line), FadeIn(lt), FadeIn(p23), FadeIn(p1), run_time=0.8); self.play(GrowFromCenter(br), FadeIn(bl), run_time=0.6); self.wait(0.5)
        circ = Circle(1.6, color=MUTED, stroke_width=3).move_to([0, -0.4, 0])
        pos = lambda h: circ.get_center() + 1.6 * np.array([np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24), 0])
        ct = VGroup(*[Text(f"{h}", font=FONT, color=MUTED).scale(0.32).move_to(circ.get_center() + 2.0 * np.array([np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24), 0])) for h in range(0, 24, 3)])
        self.mark("circle"); self.play(FadeOut(br), FadeOut(bl), ReplacementTransform(line, circ), ReplacementTransform(lt, ct), p23.animate.move_to(pos(23)), p1.animate.move_to(pos(1)), run_time=1.4)
        cl = Text("on a clock, side by side", font=FONT, weight=BOLD, color=GOOD).scale(0.42).move_to([3.8, 0.6, 0])
        self.play(FadeIn(cl), run_time=0.5); self.wait(1.4)
        self.play(FadeOut(VGroup(circ, ct, p23, p1, cl)), run_time=0.6)

        # ---------- 5. windows ----------
        cap = self.swap(cap, self.caption("The truck's past, without peeking at the future", "average of its trips over the last 30 days"))
        rng = np.random.default_rng(4)
        xs = np.sort(rng.uniform(-5.5, 5.5, 34)); ys = rng.normal(0, 0.5, 34) - 0.3
        dots = VGroup(*[Dot([x, y, 0], radius=0.08, color=SAMPLE) for x, y in zip(xs, ys)])
        hoy = Line([1.5, 1.4, 0], [1.5, -2.0, 0], color=INK, stroke_width=4); hl = Text("today", font=FONT, weight=BOLD, color=INK).scale(0.4).next_to(hoy, UP, buff=0.1)
        self.mark("win"); self.play(FadeIn(dots, lag_ratio=0.03), Create(hoy), FadeIn(hl), run_time=1.0)
        good = Rectangle(width=3.0, height=3.2, stroke_width=0, fill_color=GOOD, fill_opacity=0.18).move_to([0.0, -0.3, 0])
        gl = Text("looking back only", font=FONT, weight=BOLD, color=GOOD).scale(0.4).move_to([-2.6, 1.6, 0])
        self.play(FadeIn(good), FadeIn(gl), run_time=0.6); self.wait(0.6)
        bad = Rectangle(width=3.0, height=3.2, stroke_width=0, fill_color=WARN, fill_opacity=0.18).move_to([1.5, -0.3, 0])
        bl2 = Text("peeking at the future: a trap", font=FONT, weight=BOLD, color=WARN).scale(0.4).move_to([4.4, 1.6, 0])
        self.mark("leak"); self.play(Transform(good, bad), FadeOut(gl), FadeIn(bl2), run_time=0.8); self.wait(1.6)
        self.play(*[FadeOut(m) for m in self.mobjects if m is not cap], run_time=0.6)

        # ---------- Closing ----------
        self.mark("legend")
        self.play(FadeOut(cap), run_time=0.6)
        e1 = Text("Everything is learned from the training data only.", font=FONT, weight=BOLD, color=INK).scale(0.66)
        e2 = Text("And applied the same way on the day it is used.", font=FONT, color=MUTED).scale(0.48)
        VGroup(e1, e2).arrange(DOWN, buff=0.4)
        self.mark("end_text"); self.play(Write(e1), run_time=1.4); self.play(FadeIn(e2, shift=0.2 * UP), run_time=0.8)
        self.wait(2.0)
        self.mark("fade"); self.play(FadeOut(VGroup(e1, e2)), run_time=1)
        self.marks["total"] = [round(self.renderer.time, 3)]
        json.dump(self.marks, open("marks.json", "w"), indent=1)
