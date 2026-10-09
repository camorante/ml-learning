import json
import numpy as np
from manim import *

# Dark palette of the series
BG = "#0E131B"; INK = "#E4EAF3"; MUTED = "#9BA7B9"; LINE = "#2B3545"
MEAN = "#6EA6FF"; SAMPLE = "#AE9FF3"; POINT = "#FF9A55"; GOOD = "#4CC38A"; WARN = "#F08A5D"
FONT = "Inter"
config.background_color = BG

# The 12 trucks of the toy: hours, stops, km
TR = np.array([[7.5, 13.5, 180], [8, 13, 250], [6, 13.5, 160], [7, 10.5, 300], [8, 6, 370], [10, 7, 470],
               [8, 8.5, 330], [5.5, 5, 210], [10, 2.5, 660], [12, 1, 750], [9, 2.5, 610], [4, 9.5, 130]])


def T(s, sc=0.42, col=INK, w=NORMAL):
    return Text(s, font=FONT, color=col, weight=w).scale(sc)


class Vectores(Scene):
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
        title = Text("Vectors and distances", font=FONT, weight=BOLD, color=INK).scale(1.2)
        sub = Text("measuring how similar two trucks are", font=FONT, color=MUTED).scale(0.5)
        VGroup(title, sub).arrange(DOWN, buff=0.35)
        self.mark("title"); self.play(Write(title), run_time=1.2)
        self.mark("subtitle"); self.play(FadeIn(sub, shift=0.2 * UP)); self.wait(1.2)
        self.mark("title_out"); self.play(FadeOut(title), FadeOut(sub))

        # ---------- 1. a truck is an arrow ----------
        cap = self.swap(None, self.caption("A truck is a list of numbers", "stops per day and working hours per day"))
        O = np.array([-5.6, -3.3, 0]); u = 0.36
        P = lambda x, y: O + np.array([x * u, y * u, 0])
        ax = VGroup(Line(P(0, 0), P(15.5, 0), color=MUTED, stroke_width=2), Line(P(0, 0), P(0, 14), color=MUTED, stroke_width=2))
        axl = VGroup(T("stops", 0.34, MUTED).next_to(P(15.5, 0), DOWN, buff=0.15).shift(0.5 * LEFT),
                     T("hours", 0.34, MUTED).next_to(P(0, 14), RIGHT, buff=0.15))
        dots = VGroup(*[Dot(P(r[1], r[0]), radius=0.08, color=POINT) for r in TR])
        self.play(Create(ax), FadeIn(axl), run_time=0.7)
        self.mark("dots"); self.play(LaggedStart(*[FadeIn(d, scale=2) for d in dots], lag_ratio=0.08), run_time=1.2)
        ref = P(13, 8)
        hx = DashedLine(P(0, 0), P(13, 0), color=SAMPLE, stroke_width=4); hy = DashedLine(P(13, 0), ref, color=SAMPLE, stroke_width=4)
        arr = Arrow(P(0, 0), ref, buff=0, color=MEAN, stroke_width=6, max_tip_length_to_length_ratio=0.06)
        lx = T("13", 0.4, SAMPLE, BOLD).next_to(P(6.5, 0), UP, buff=0.12); ly = T("8", 0.4, SAMPLE, BOLD).next_to(P(13, 4), RIGHT, buff=0.12)
        self.mark("arrow"); self.play(GrowArrow(arr), Create(hx), Create(hy), dots[1].animate.set_color(MEAN).scale(1.6), run_time=1.2)
        self.play(FadeIn(lx), FadeIn(ly), run_time=0.4)
        vec = T("truck 2 = (13, 8)", 0.6, MEAN, BOLD).move_to([3.6, 1.0, 0])
        vs = T("a point, and also an arrow from zero", 0.36, MUTED).next_to(vec, DOWN, buff=0.25)
        self.play(FadeIn(vec, shift=0.2 * UP), FadeIn(vs), run_time=0.7); self.wait(0.6)
        nl = Line(ref, P(13.5, 7.5), color=GOOD, stroke_width=6)
        near = T("most similar: truck 1, at 0.71", 0.44, GOOD, BOLD).move_to([3.6, -0.6, 0])
        self.mark("near"); self.play(Create(nl), dots[0].animate.set_color(GOOD).scale(1.6), FadeIn(near), run_time=0.8); self.wait(2.6)
        self.clear_but(cap)

        # ---------- 2. adding arrows ----------
        cap = self.swap(cap, self.caption("Adding arrows", "Monday + Tuesday = the work of both days"))
        O = np.array([-5.4, -3.3, 0]); u = 0.5
        P = lambda x, y: O + np.array([x * u, y * u, 0])
        ax = VGroup(Line(P(0, 0), P(11, 0), color=LINE, stroke_width=2), Line(P(0, 0), P(0, 10.2), color=LINE, stroke_width=2))
        self.play(Create(ax), run_time=0.4)
        a1 = Arrow(P(0, 0), P(6, 4), buff=0, color=POINT, stroke_width=6, max_tip_length_to_length_ratio=0.1)
        a2 = Arrow(P(6, 4), P(9, 9), buff=0, color=SAMPLE, stroke_width=6, max_tip_length_to_length_ratio=0.1)
        a3 = Arrow(P(0, 0), P(9, 9), buff=0, color=MEAN, stroke_width=8, max_tip_length_to_length_ratio=0.07)
        t1 = T("Monday (6, 4)", 0.4, POINT).next_to(P(6, 4), RIGHT, buff=0.2)
        t2 = T("Tuesday (3, 5)", 0.4, SAMPLE).next_to(P(8.3, 7.4), RIGHT, buff=0.45)
        self.mark("sum"); self.play(GrowArrow(a1), FadeIn(t1), run_time=0.7)
        self.mark("sum"); self.play(GrowArrow(a2), FadeIn(t2), run_time=0.7)
        eq = T("(6, 4) + (3, 5) = (9, 9)", 0.6, MEAN, BOLD).move_to([3.4, 0.4, 0])
        eqs = T("stops with stops, hours with hours", 0.36, MUTED).next_to(eq, DOWN, buff=0.25)
        self.mark("sum"); self.play(GrowArrow(a3), FadeIn(eq), FadeIn(eqs), run_time=0.9); self.wait(1.0)
        a4 = Arrow(P(0, 0), P(12, 8), buff=0, color=GOOD, stroke_width=5, max_tip_length_to_length_ratio=0.06)
        e2 = T("2 × (6, 4) = (12, 8): same direction, twice as long", 0.36, GOOD).next_to(eqs, DOWN, buff=0.45)
        self.mark("double"); self.play(GrowArrow(a4), FadeIn(e2), run_time=0.8); self.wait(2.4)
        self.clear_but(cap)

        # ---------- 3. Pythagoras ----------
        cap = self.swap(cap, self.caption("The length of an arrow: Pythagoras", "3 blocks east and 4 north"))
        O = np.array([-5.0, -3.2, 0]); u = 1.05
        P = lambda x, y: O + np.array([x * u, y * u, 0])
        grid = VGroup(*[Line(P(i, -0.3), P(i, 4.6), color=LINE, stroke_width=1.5) for i in range(6)],
                      *[Line(P(-0.3, j), P(5.3, j), color=LINE, stroke_width=1.5) for j in range(5)])
        self.play(Create(grid), run_time=0.6)
        l1 = Line(P(0, 0), P(3, 0), color=POINT, stroke_width=8); l2 = Line(P(3, 0), P(3, 4), color=SAMPLE, stroke_width=8)
        hyp = Arrow(P(0, 0), P(3, 4), buff=0, color=MEAN, stroke_width=7, max_tip_length_to_length_ratio=0.06)
        n3 = T("3", 0.6, POINT, BOLD).next_to(P(1.5, 0), DOWN, buff=0.15); n4 = T("4", 0.6, SAMPLE, BOLD).next_to(P(3, 2), RIGHT, buff=0.15)
        self.mark("legs"); self.play(Create(l1), FadeIn(n3), run_time=0.6); self.play(Create(l2), FadeIn(n4), run_time=0.6)
        n5 = T("5", 0.7, MEAN, BOLD).move_to(P(1.2, 2.4))
        self.mark("pyth"); self.play(GrowArrow(hyp), FadeIn(n5), run_time=0.8)
        f1 = T("√(3² + 4²) = √(9 + 16) = √25 = 5", 0.56, INK, BOLD).move_to([3.0, 0.6, 0])
        f2 = T("the length (or norm) of the vector (3, 4)", 0.38, MUTED).next_to(f1, DOWN, buff=0.3)
        self.play(FadeIn(f1, shift=0.2 * UP), FadeIn(f2), run_time=0.7); self.wait(2.8)
        self.clear_but(cap)

        # ---------- 4. three distances ----------
        cap = self.swap(cap, self.caption("Distance is the length of the difference", "and there are several ways to measure it"))
        O = np.array([-5.8, -3.1, 0]); u = 0.9
        P = lambda x, y: O + np.array([x * u, y * u, 0])
        grid = VGroup(*[Line(P(i, -0.3), P(i, 4.6), color=LINE, stroke_width=1.5) for i in range(6)],
                      *[Line(P(-0.3, j), P(5.3, j), color=LINE, stroke_width=1.5) for j in range(5)])
        A = Dot(P(0.5, 0.5), radius=0.13, color=POINT); B = Dot(P(4.5, 3.5), radius=0.13, color=SAMPLE)
        lA = T("A", 0.45, POINT, BOLD).next_to(A, DL, buff=0.08); lB = T("B", 0.45, SAMPLE, BOLD).next_to(B, UR, buff=0.08)
        self.play(Create(grid), FadeIn(A, lA, B, lB), run_time=0.7)
        rows = [("straight line", "√(4² + 3²) = 5", Line(P(0.5, 0.5), P(4.5, 3.5))),
                ("by blocks", "4 + 3 = 7", VMobject().set_points_as_corners([P(0.5, 0.5), P(4.5, 0.5), P(4.5, 3.5)])),
                ("by the longest side", "the larger of 4 and 3 = 4", Line(P(0.5, 0.5), P(4.5, 0.5)))]
        prev = None; txts = VGroup()
        for k, (name, val, path) in enumerate(rows):
            path.set_stroke(GOOD, 8)
            row = VGroup(T(name, 0.42, INK, BOLD), T(val, 0.42, GOOD)).arrange(RIGHT, buff=0.35).move_to([2.6, 1.6 - 0.75 * k, 0]).align_to([-0.6, 0, 0], LEFT)
            self.mark("path")
            if prev is None:
                self.play(Create(path), FadeIn(row), run_time=0.8)
            else:
                self.play(ReplacementTransform(prev, path), FadeIn(row), run_time=0.8)
            prev = path; txts.add(row); self.wait(0.7)
        # "circles" of radius 1 of each type
        cx = [0.6, 2.9, 5.2]; cy = -2.3; r = 0.75
        shapes = VGroup(Circle(radius=r).move_to([cx[0], cy, 0]),
                        Polygon([cx[1] - r, cy, 0], [cx[1], cy + r, 0], [cx[1] + r, cy, 0], [cx[1], cy - r, 0]),
                        Square(side_length=2 * r).move_to([cx[2], cy, 0]))
        shapes.set_stroke(MEAN, 4).set_fill(MEAN, 0.18)
        sl = VGroup(*[T(s, 0.32, MUTED).next_to(shapes[i], DOWN, buff=0.15) for i, s in enumerate(["circle", "diamond", "square"])])
        st = T("all the points at distance 1:", 0.36, INK).move_to([2.9, -1.2, 0])
        self.mark("shapes"); self.play(FadeIn(st), LaggedStart(*[DrawBorderThenFill(s) for s in shapes], lag_ratio=0.3), FadeIn(sl), run_time=1.4)
        self.wait(2.4)
        self.clear_but(cap)

        # ---------- 5. units rule ----------
        cap = self.swap(cap, self.caption("Units rule", "km (hundreds) versus stops (units)"))
        km, pa = TR[:, 2], TR[:, 1]
        raw = [np.array([-6.0 + k * 0.0085, -2.6 + p * 0.0085, 0]) for k, p in zip(km, pa)]
        zk, zp = (km - km.mean()) / km.std(), (pa - pa.mean()) / pa.std()
        sc = [np.array([-1.6 + a * 1.45, -0.4 + b * 1.45, 0]) for a, b in zip(zk, zp)]
        dots = VGroup(*[Dot(q, radius=0.1, color=POINT) for q in raw]); dots[1].set_color(MEAN).scale(1.5)
        base = Line([-6.2, -2.6, 0], [0.8, -2.6, 0], color=LINE, stroke_width=2)
        bl = T("on the same ruler, 16 stops measure the same as 16 km: almost nothing", 0.34, MUTED).next_to(base, DOWN, buff=0.3).align_to(base, LEFT)
        self.mark("raw"); kml = T("km →", 0.34, MUTED).next_to(base, RIGHT, buff=0.15)
        self.play(Create(base), FadeIn(dots), FadeIn(bl), FadeIn(kml), run_time=0.9)
        g1 = Line(raw[1], raw[7], color=GOOD, stroke_width=6)
        r1 = VGroup(T("not scaled", 0.46, INK, BOLD), T("most similar to truck 2:", 0.38, MUTED),
                    T("truck 8, with 5 stops", 0.44, WARN, BOLD), T("(truck 2 makes 13)", 0.36, MUTED)).arrange(DOWN, buff=0.18, aligned_edge=LEFT).move_to([4.4, 0.8, 0])
        self.play(Create(g1), dots[7].animate.set_color(GOOD).scale(1.5), FadeIn(r1), run_time=0.9); self.wait(2.2)
        r2 = VGroup(T("scaled", 0.46, INK, BOLD), T("each number in deviations:", 0.38, MUTED),
                    T("truck 1, with 13.5 stops", 0.44, GOOD, BOLD), T("another urban delivery truck", 0.36, MUTED)).arrange(DOWN, buff=0.18, aligned_edge=LEFT).move_to([4.4, 0.8, 0])
        self.mark("scale")
        self.play(*[dots[i].animate.move_to(sc[i]) for i in range(12) if i != 7], FadeOut(g1), FadeOut(base), FadeOut(bl), FadeOut(kml), dots[7].animate.move_to(sc[7]).set_color(POINT).scale(1 / 1.5), FadeOut(r1), run_time=1.5)
        g2 = Line(sc[1], sc[0], color=GOOD, stroke_width=6)
        self.play(Create(g2), dots[0].animate.set_color(GOOD).scale(1.5), FadeIn(r2), run_time=0.8); self.wait(2.6)
        self.clear_but(cap)

        # ---------- 6. many dimensions ----------
        cap = self.swap(cap, self.caption("With many numbers, everything is equally far away", "distances between 120 random made-up trucks"))
        rng = np.random.default_rng(1)
        def hist(d):
            X = rng.random((120, d)); G = X @ X.T; n2 = np.diag(G)
            D = np.sqrt(np.maximum(n2[:, None] + n2[None, :] - 2 * G, 0))[np.triu_indices(120, 1)]
            h, _ = np.histogram(D / D.mean(), bins=40, range=(0, 2))
            M = np.sqrt(np.maximum(n2[:, None] + n2[None, :] - 2 * G, 0)); np.fill_diagonal(M, np.nan)
            ratio = np.nanmean(np.nanmin(M[:30], 1) / np.nanmax(M[:30], 1))
            return h / h.max(), ratio
        h2, r2v = hist(2); h1k, r1kv = hist(1000)
        x0, W, H0, Hh = -5.5, 8.0, -2.9, 4.2
        def bars(h):
            return VGroup(*[Rectangle(width=W / 40 * 0.82, height=max(0.01, v * Hh), stroke_width=0, fill_color=MEAN, fill_opacity=0.85)
                            .move_to([x0 + (i + 0.5) * W / 40, H0 + max(0.01, v * Hh) / 2, 0]) for i, v in enumerate(h)])
        axis = Line([x0, H0, 0], [x0 + W, H0, 0], color=MUTED, stroke_width=2)
        ticks = VGroup(*[T(s, 0.32, MUTED).move_to([x0 + f * W, H0 - 0.3, 0]) for f, s in ((0, "0"), (0.5, "1 = the average"), (1, "2"))])
        b = bars(h2)
        lab = T("2 numbers", 0.6, INK, BOLD).move_to([4.6, 1.2, 0])
        rat = VGroup(T("nearest / farthest", 0.36, MUTED), T(f"{round(r2v * 100)}%", 0.8, GOOD, BOLD)).arrange(DOWN, buff=0.2).move_to([4.6, -0.3, 0])
        self.mark("hist"); self.play(Create(axis), FadeIn(ticks), FadeIn(b, lag_ratio=0.02), FadeIn(lab), FadeIn(rat), run_time=1.0); self.wait(1.2)
        lab2 = T("1,000 numbers", 0.6, INK, BOLD).move_to(lab)
        rat2 = VGroup(T("nearest / farthest", 0.36, MUTED), T(f"{round(r1kv * 100)}%", 0.8, WARN, BOLD)).arrange(DOWN, buff=0.2).move_to(rat)
        self.mark("dims"); self.play(Transform(b, bars(h1k)), Transform(lab, lab2), Transform(rat, rat2), run_time=1.6)
        nt = T("\"the most similar\" almost stops meaning anything", 0.4, WARN, BOLD).move_to([0, -3.6, 0])
        self.play(FadeIn(nt), run_time=0.5); self.wait(2.4)
        self.play(*[FadeOut(m) for m in self.mobjects if m is not cap], run_time=0.6)

        # ---------- Ending ----------
        self.mark("legend")
        self.play(FadeOut(cap), run_time=0.6)
        e1 = Text("Similar means close.", font=FONT, weight=BOLD, color=INK).scale(0.7)
        e2 = Text("Scale the numbers before measuring, and be wary of many dimensions.", font=FONT, color=MUTED).scale(0.42)
        VGroup(e1, e2).arrange(DOWN, buff=0.4)
        self.mark("end_text"); self.play(Write(e1), run_time=1.4); self.play(FadeIn(e2, shift=0.2 * UP), run_time=0.8)
        self.wait(2.4)
        self.mark("fade"); self.play(FadeOut(VGroup(e1, e2)), run_time=1)
        self.marks["total"] = [round(self.renderer.time, 3)]
        json.dump(self.marks, open("marks.json", "w"), indent=1)
