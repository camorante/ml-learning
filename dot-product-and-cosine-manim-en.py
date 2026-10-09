import json
import numpy as np
from manim import *

# Dark palette of the series
BG = "#0E131B"; INK = "#E4EAF3"; MUTED = "#9BA7B9"; LINE = "#2B3545"
MEAN = "#6EA6FF"; SAMPLE = "#AE9FF3"; POINT = "#FF9A55"; GOOD = "#4CC38A"; WARN = "#F08A5D"
FONT = "Inter"
config.background_color = BG


def T(s, sc=0.45, col=INK, w=None):
    return Text(s, font=FONT, color=col, weight=w or NORMAL).scale(sc)


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
        title = Text("Dot product and cosine similarity", font=FONT, weight=BOLD, color=INK).scale(0.95)
        sub = Text("multiply in pairs and add up", font=FONT, color=MUTED).scale(0.5)
        VGroup(title, sub).arrange(DOWN, buff=0.35)
        self.mark("title"); self.play(Write(title), run_time=1.3)
        self.mark("subtitle"); self.play(FadeIn(sub, shift=0.2 * UP)); self.wait(1.3)
        self.mark("title_out"); self.play(FadeOut(title), FadeOut(sub))

        # ---------- 1. trip cost ----------
        cap = self.swap(None, self.caption("The cost of a trip", "kilometers per kind of road  ·  price of each kilometer"))
        rows = [("city", 40, "0.60", 24, POINT), ("rural road", 120, "0.40", 48, SAMPLE), ("highway", 200, "0.50", 100, MEAN)]
        grp = VGroup()
        for i, (n, k, p, c, col) in enumerate(rows):
            y = 1.3 - i * 1.05
            lab = T(n, 0.5, MUTED).move_to([-5.0, y, 0])
            r = Rectangle(width=k / 200 * 3.6, height=float(p) / 0.6 * 0.8, stroke_color=col, stroke_width=3, fill_color=col, fill_opacity=0.3)
            r.align_to([-3.6, 0, 0], LEFT).set_y(y)
            eq = T(f"{k} km × ${p} = ${c}", 0.5).move_to([2.9, y, 0]).align_to([1.0, 0, 0], LEFT)
            grp.add(VGroup(lab, r, eq))
        for g in grp:
            self.mark("row"); self.play(FadeIn(g[0]), GrowFromEdge(g[1], LEFT), FadeIn(g[2], shift=0.2 * LEFT), run_time=0.7); self.wait(0.25)
        bar = VGroup(); x = -3.6
        for (n, k, p, c, col) in rows:
            w = c / 172 * 6.0
            bar.add(Rectangle(width=w, height=0.55, stroke_width=0, fill_color=col, fill_opacity=0.85).move_to([x + w / 2, -1.95, 0])); x += w
        tot = T("= $172", 0.7, INK, BOLD).next_to(bar, RIGHT, buff=0.3)
        tl = T("total", 0.5, MUTED).move_to([-5.0, -1.95, 0])
        self.mark("sum"); self.play(FadeIn(tl), LaggedStart(*[GrowFromEdge(b, LEFT) for b in bar], lag_ratio=0.5), run_time=1.0); self.play(FadeIn(tot, scale=1.2), run_time=0.5)
        nm = T("that is a dot product:  (40, 120, 200) · (0.60, 0.40, 0.50)", 0.42, GOOD).move_to([0, -3.0, 0])
        self.play(FadeIn(nm), run_time=0.6); self.wait(2.6)
        self.clear_but(cap)

        # ---------- 2. two arrows ----------
        cap = self.swap(cap, self.caption("The sign tells you where they point", "truck (blue) and wind (orange)"))
        O = np.array([-2.6, -0.9, 0]); hd = np.arctan2(45, 60); S = 0.05
        truck = Arrow(O, O + S * 75 * np.array([np.cos(hd), np.sin(hd), 0]), buff=0, color=MEAN, stroke_width=8, max_tip_length_to_length_ratio=0.12)
        th = ValueTracker(0.0)
        def wind():
            a = hd + th.get_value() * DEGREES
            return Arrow(O, O + S * 50 * np.array([np.cos(a), np.sin(a), 0]), buff=0, color=POINT, stroke_width=8, max_tip_length_to_length_ratio=0.18)
        wv = always_redraw(wind)
        def shadow():
            al = 50 * np.cos(th.get_value() * DEGREES)
            u = np.array([np.cos(hd), np.sin(hd), 0])
            return Line(O, O + S * al * u + 1e-4 * u, stroke_width=14, color=GOOD if al >= 0 else WARN, stroke_opacity=0.6)
        sh = always_redraw(shadow)
        def readout():
            t = th.get_value(); d = 75 * 50 * np.cos(t * DEGREES)
            col = GOOD if d > 150 else WARN if d < -150 else INK
            s = f"{abs(d):,.0f}"
            v = VGroup(T(f"angle {t:.0f}°", 0.55, MUTED), T(("−" if d < -0.5 else "") + s, 1.1, col, BOLD),
                       T("helping" if d > 150 else "working against" if d < -150 else "from the side: zero", 0.5, col)).arrange(DOWN, buff=0.25)
            return v.move_to([3.8, -0.4, 0])
        ro = always_redraw(readout)
        self.mark("arrows"); self.play(GrowArrow(truck), FadeIn(wv), FadeIn(sh), FadeIn(ro), run_time=0.9); self.wait(0.8)
        self.mark("rot"); self.play(th.animate.set_value(90), run_time=2.2, rate_func=smooth); self.wait(0.8)
        self.mark("rot"); self.play(th.animate.set_value(180), run_time=2.2, rate_func=smooth); self.wait(1.4)
        self.clear_but(cap)

        # ---------- 3. the shadow ----------
        cap = self.swap(cap, self.caption("The shadow: how much closer to the warehouse", "trip · warehouse direction (arrow of length 1)"))
        O = np.array([-4.8, -2.3, 0]); al = 18 * DEGREES; u = np.array([np.cos(al), np.sin(al), 0]); S = 0.075
        road = Line(O - 0.6 * u, O + 9.8 * u, color=MUTED, stroke_width=3, stroke_opacity=0.5)
        wh = Square(0.5, color=INK, stroke_width=3).move_to(O + 9.9 * u)
        whl = T("warehouse", 0.42).next_to(wh, DOWN, buff=0.15)
        a = al + 40 * DEGREES; E = O + S * 50 * np.array([np.cos(a), np.sin(a), 0])
        trip = Arrow(O, E, buff=0, color=POINT, stroke_width=8, max_tip_length_to_length_ratio=0.1)
        P = O + S * 50 * np.cos(40 * DEGREES) * u
        drop = DashedLine(E, P, color=INK, stroke_width=2)
        shl = Line(O, P, color=GOOD, stroke_width=16, stroke_opacity=0.7)
        self.mark("road"); self.play(Create(road), FadeIn(wh), FadeIn(whl), run_time=0.8)
        self.play(GrowArrow(trip), FadeIn(T("50 km", 0.5, POINT, BOLD).next_to(E, LEFT, buff=0.2)), run_time=0.8); self.wait(0.4)
        self.mark("shadow"); self.play(Create(drop), run_time=0.6); self.play(Create(shl), run_time=0.9)
        res = T("38 km toward the warehouse", 0.55, GOOD, BOLD).move_to(O + 2.1 * u + np.array([0.6, -0.7, 0]))
        self.play(FadeIn(res, shift=0.2 * UP), run_time=0.6)
        nt = T("50 × cos 40° ≈ 38", 0.5, MUTED).move_to([2.6, -2.9, 0])
        self.play(FadeIn(nt), run_time=0.5); self.wait(2.0)
        self.clear_but(cap)

        # ---------- 4. cosine ----------
        cap = self.swap(cap, self.caption("Cosine similarity: shape, not size", "hours per week in the city (→) and on the highway (↑)"))
        ax = Axes(x_range=[0, 60, 10], y_range=[0, 60, 10], x_length=4.9, y_length=4.9, tips=False,
                  axis_config={"color": MUTED, "stroke_width": 2, "include_ticks": False}).move_to([-3.4, -0.85, 0])
        pts = {"A": (12, 28), "B": (24, 56), "C": (22, 18)}
        cols = {"A": POINT, "B": MEAN, "C": SAMPLE}
        arr = {k: Arrow(ax.c2p(0, 0), ax.c2p(*v), buff=0, color=cols[k], stroke_width=(10 if k == "A" else 5), max_tip_length_to_length_ratio=(0.16 if k == "A" else 0.1)) for k, v in pts.items()}
        arr["A"].set_z_index(3)
        labs = {k: T(k, 0.6, cols[k], BOLD).next_to(ax.c2p(*v), (LEFT if k == "A" else UR), buff=0.15) for k, v in pts.items()}
        self.play(Create(ax), run_time=0.6)
        self.mark("cosA"); self.play(GrowArrow(arr["A"]), FadeIn(labs["A"]), GrowArrow(arr["C"]), FadeIn(labs["C"]), run_time=0.9)
        dl = DashedLine(ax.c2p(*pts["A"]), ax.c2p(*pts["C"]), color=SAMPLE, stroke_width=4)
        r1 = VGroup(T("A and C", 0.5, MUTED), T("distance 14.1 h", 0.55, SAMPLE, BOLD), T("cosine 0.887 (27.5°)", 0.55, INK)).arrange(DOWN, aligned_edge=LEFT, buff=0.18).move_to([3.1, 1.0, 0])
        self.play(Create(dl), FadeIn(r1), run_time=0.8); self.wait(1.2)
        self.mark("cosB"); self.play(GrowArrow(arr["B"]), FadeIn(labs["B"]), run_time=0.9)
        r2 = VGroup(T("A and B (B = twice A)", 0.5, MUTED), T("distance 30.5 h", 0.55, INK), T("cosine 1 (0°)", 0.55, MEAN, BOLD)).arrange(DOWN, aligned_edge=LEFT, buff=0.18).move_to([3.1, -1.3, 0])
        self.play(FadeIn(r2), run_time=0.7)
        nt = T("same split of time → same direction", 0.45, GOOD).move_to([2.9, -2.9, 0])
        self.play(FadeIn(nt), run_time=0.5); self.wait(2.8)
        self.clear_but(cap)

        # ---------- 5. linear model ----------
        cap = self.swap(cap, self.caption("A linear model is a dot product", "score = weights · data + bias"))
        rng = np.random.default_rng(4)
        av = np.c_[rng.normal(6.6, 1.3, 22), rng.normal(6.4, 1.5, 22)]
        sa = np.c_[rng.normal(3.6, 1.3, 22), rng.normal(3.6, 1.5, 22)]
        ax = Axes(x_range=[0, 10, 2], y_range=[0, 10, 2], x_length=5.0, y_length=5.0, tips=False,
                  axis_config={"color": MUTED, "stroke_width": 2, "include_ticks": False}).move_to([-3.2, -0.85, 0])
        dots = VGroup(*[Dot(ax.c2p(*np.clip(p, 0.3, 9.7)), radius=0.07, color=POINT) for p in av],
                      *[Dot(ax.c2p(*np.clip(p, 0.3, 9.7)), radius=0.07, color=SAMPLE) for p in sa])
        xl = T("vibration →", 0.38, MUTED).next_to(ax, DOWN, buff=0.12); yl = T("hours since inspection ↑", 0.38, MUTED).next_to(ax, UP, buff=0.1).align_to(ax, LEFT)
        self.mark("dots"); self.play(Create(ax), FadeIn(dots, lag_ratio=0.02), FadeIn(xl), FadeIn(yl), run_time=1.0)
        # boundary 1·x + 0.8·y − 9 = 0
        fr = Line(ax.c2p(9.0 - 0.8 * 10 + 0.0, 10), ax.c2p(9.0, 0), color=INK, stroke_width=4)
        F = np.array([5.224, 4.720]); wv = np.array([1, 0.8]) / np.hypot(1, 0.8)
        warr = Arrow(ax.c2p(*F), ax.c2p(*(F + 1.8 * wv)), buff=0, color=MEAN, stroke_width=7, max_tip_length_to_length_ratio=0.25)
        wl = T("weights", 0.45, MEAN, BOLD).next_to(warr.get_end(), RIGHT, buff=0.1)
        self.mark("line"); self.play(Create(fr), run_time=0.8); self.play(GrowArrow(warr), FadeIn(wl), run_time=0.6)
        info = VGroup(T("1 × vibration + 0.8 × hours − 9", 0.48, INK),
                      T("> 0  →  to inspection", 0.5, POINT, BOLD),
                      T("the weights arrow forms", 0.45, MUTED), T("a right angle with the boundary", 0.45, MUTED)).arrange(DOWN, aligned_edge=LEFT, buff=0.2).move_to([3.3, 0.1, 0])
        self.play(FadeIn(info, shift=0.2 * LEFT), run_time=0.8)
        nt = VGroup(T("this is how regression, logistic regression,", 0.42, GOOD), T("SVMs… and a neuron score", 0.42, GOOD)).arrange(DOWN, aligned_edge=LEFT, buff=0.15).next_to(info, DOWN, buff=0.5).align_to(info, LEFT)
        self.mark("models"); self.play(FadeIn(nt), run_time=0.6); self.wait(2.6)
        self.clear_but(cap)

        # ---------- 6. finding look-alikes ----------
        cap = self.swap(cap, self.caption("Finding look-alikes: embeddings and cosine", "each text, an arrow (brakes · electrics · engine)"))
        q = T("new: “Noise when pressing the brake”", 0.5, POINT, BOLD).move_to([0, 1.55, 0])
        past = [("Squeal when braking downhill", 0.98), ("Worn brake pads", 0.95), ("Metallic noise in the engine", 0.42),
                ("Black smoke when accelerating", 0.31), ("Won't start: dead battery", 0.13)]
        self.mark("query"); self.play(FadeIn(q, shift=0.2 * DOWN), run_time=0.6)
        rows = VGroup()
        for i, (t, c) in enumerate(past):
            y = 0.75 - i * 0.68
            lab = T(t, 0.42, INK if i else GOOD).move_to([-2.6, y, 0]).align_to([-6.2, 0, 0], LEFT)
            bg = Rectangle(width=4.0, height=0.3, stroke_width=0, fill_color=LINE, fill_opacity=1).move_to([1.6, y, 0])
            fg = Rectangle(width=4.0 * c, height=0.3, stroke_width=0, fill_color=GOOD if i == 0 else SAMPLE, fill_opacity=1).align_to(bg, LEFT).set_y(y)
            nv = T(f"{c:.2f}", 0.42, GOOD if i == 0 else MUTED).next_to(bg, RIGHT, buff=0.25)
            rows.add(VGroup(lab, bg, fg, nv))
        self.mark("bars"); self.play(LaggedStart(*[AnimationGroup(FadeIn(r[0]), FadeIn(r[1]), GrowFromEdge(r[2], LEFT), FadeIn(r[3])) for r in rows], lag_ratio=0.25), run_time=1.8)
        nt = T("almost no words in common, and the most similar", 0.45, GOOD).move_to([0, -2.85, 0])
        self.play(FadeIn(nt), run_time=0.5); self.wait(2.6)
        self.clear_but(cap)

        # ---------- Closing ----------
        self.mark("legend")
        self.play(FadeOut(cap), run_time=0.6)
        e1 = Text("Multiply in pairs and add up.", font=FONT, weight=BOLD, color=INK).scale(0.72)
        e2 = Text("The sign tells you where they point; divided by the lengths, how similar they are.", font=FONT, color=MUTED).scale(0.4)
        VGroup(e1, e2).arrange(DOWN, buff=0.4)
        self.mark("end_text"); self.play(Write(e1), run_time=1.4); self.play(FadeIn(e2, shift=0.2 * UP), run_time=0.8)
        self.wait(2.4)
        self.mark("fade"); self.play(FadeOut(VGroup(e1, e2)), run_time=1)
        self.marks["total"] = [round(self.renderer.time, 3)]
        json.dump(self.marks, open("marks.json", "w"), indent=1)
