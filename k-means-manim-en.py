import json
import numpy as np
from manim import *
from sklearn.cluster import KMeans

# Dark palette of the visual ML series
BG = "#0E131B"; INK = "#E4EAF3"; MUTED = "#9BA7B9"; LINE = "#2B3545"
MEAN = "#6EA6FF"; SAMPLE = "#AE9FF3"; POINT = "#FF9A55"; GOOD = "#4CC38A"; WARN = "#F08A5D"
COLS = [POINT, SAMPLE, MEAN, GOOD]
FONT = "Inter"
config.background_color = BG


def blobs(rng, centers, n, s):
    return np.vstack([np.c_[rng.normal(cx, s, n), rng.normal(cy, s, n)] for cx, cy in centers])


def lloyd_steps(P, C, iters=10):
    hist = []
    for _ in range(iters):
        L = ((P[:, None] - C[None]) ** 2).sum(2).argmin(1)
        hist.append((C.copy(), L.copy()))
        C2 = np.array([P[L == k].mean(0) if np.any(L == k) else C[k] for k in range(len(C))])
        if np.allclose(C2, C):
            break
        C = C2
    return hist


class KMedias(Scene):
    def caption(self, text, sub=None):
        t = Text(text, font=FONT, weight=SEMIBOLD, color=INK).scale(0.62)
        grp = VGroup(t)
        if sub:
            grp.add(Text(sub, font=FONT, color=MUTED).scale(0.42)); grp.arrange(DOWN, buff=0.18)
        grp.to_edge(UP, buff=0.4)
        band = Rectangle(width=config.frame_width + 1, height=1.55, stroke_width=0, fill_color=BG, fill_opacity=1).move_to([0, config.frame_height / 2 - 0.72, 0])
        return VGroup(band, grp)

    def mark(self, name):
        self.marks.setdefault(name, []).append(round(self.renderer.time, 3))

    def swap(self, old, new):
        self.mark("caption")
        if old is None:
            self.play(FadeIn(new, shift=0.2 * DOWN), run_time=0.8)
        else:
            self.play(FadeOut(old, shift=0.2 * UP), FadeIn(new, shift=0.2 * DOWN), run_time=0.8)
        return new

    def xmark(self, p, col, s=0.16, hollow=False):
        ring = Circle(radius=s + 0.06, color=col, stroke_width=4 if not hollow else 2.5, fill_color=BG, fill_opacity=0 if hollow else 1)
        if hollow:
            ring = DashedVMobject(ring, num_dashes=10)
            return VGroup(ring).move_to(p)
        x = VGroup(Line([-s * 0.65, -s * 0.65, 0], [s * 0.65, s * 0.65, 0], color=col, stroke_width=5),
                   Line([-s * 0.65, s * 0.65, 0], [s * 0.65, -s * 0.65, 0], color=col, stroke_width=5))
        return VGroup(ring, x).move_to(p)

    def construct(self):
        self.marks = {}
        title = Text("K-means", font=FONT, weight=BOLD, color=INK).scale(1.3)
        sub = Text("grouping data without anyone telling you the groups", font=FONT, color=MUTED).scale(0.5)
        VGroup(title, sub).arrange(DOWN, buff=0.35)
        self.mark("title"); self.play(Write(title), run_time=1.4)
        self.mark("subtitle"); self.play(FadeIn(sub, shift=0.2 * UP)); self.wait(1.2)
        self.mark("title_out"); self.play(FadeOut(title), FadeOut(sub))

        # ---------- 1. Assign and move ----------
        cap = self.swap(None, self.caption("Assign and move, over and over", "each point to the nearest center; each center to the middle of its points"))
        rng = np.random.default_rng(2)
        P = blobs(rng, [(2.5, 6.8), (7.6, 7.2), (5.2, 2.6)], 45, 0.9)
        ax = Axes(x_range=[0, 10], y_range=[0, 10], x_length=6.6, y_length=5.3, axis_config={"stroke_opacity": 0}, tips=False).move_to([-2.2, -0.85, 0])
        dots = VGroup(*[Dot(ax.c2p(*p), radius=0.06, color=MUTED) for p in P])
        self.mark("points"); self.play(LaggedStart(*[GrowFromCenter(d) for d in dots], lag_ratio=0.01), run_time=1.2)
        C0 = np.array([[4.5, 9.0], [5.0, 9.5], [5.5, 9.0]])
        hist = lloyd_steps(P, C0)
        cents = VGroup(*[self.xmark(ax.c2p(*c), COLS[k]) for k, c in enumerate(C0)])
        self.mark("centers"); self.play(LaggedStart(*[GrowFromCenter(c) for c in cents], lag_ratio=0.2), run_time=0.8)
        def jtext(J, phase):
            return VGroup(Text(phase, font=FONT, weight=SEMIBOLD, color=INK).scale(0.5),
                          Text(f"inertia: {J:.0f}", font=FONT, color=MUTED).scale(0.45)).arrange(DOWN, buff=0.2, aligned_edge=LEFT).move_to([4.4, -0.6, 0])
        info = None
        for it, (C, L) in enumerate(hist):
            J = ((P - C[L]) ** 2).sum()
            ni = jtext(J, f"round {it + 1}: assign")
            self.mark("assign")
            anims = [d.animate.set_color(COLS[l]) for d, l in zip(dots, L)]
            self.play(*anims, FadeIn(ni) if info is None else Transform(info, ni), run_time=0.6 if it < 3 else 0.4)
            if info is None: info = ni
            if it + 1 < len(hist):
                C2 = hist[it + 1][0]
                self.mark("move")
                self.play(*[cents[k].animate.move_to(ax.c2p(*C2[k])) for k in range(3)],
                          Transform(info, jtext(((P - C2[L]) ** 2).sum(), f"round {it + 1}: move")), run_time=0.7 if it < 3 else 0.5)
            self.wait(0.25)
        done = Text("nothing changes: done", font=FONT, weight=SEMIBOLD, color=GOOD).scale(0.45).next_to(info, DOWN, buff=0.35, aligned_edge=LEFT)
        self.mark("done"); self.play(FadeIn(done), run_time=0.6); self.wait(1.4)
        self.play(FadeOut(VGroup(dots, cents, info, done)), run_time=0.6)

        # ---------- 2. The elbow ----------
        cap = self.swap(cap, self.caption("How many groups? Look for the elbow", "more groups always lower the inertia, but at some point they barely do"))
        Js = [KMeans(k, n_init=10, random_state=0).fit(P).inertia_ for k in range(1, 9)]
        ax2 = Axes(x_range=[0.5, 8.5, 1], y_range=[0, Js[0] * 1.05, Js[0] / 2], x_length=9, y_length=4.4, tips=False,
                   axis_config={"color": LINE, "stroke_width": 2, "include_ticks": False}).move_to([0, -1.0, 0])
        xl = Text("K (number of groups)", font=FONT, color=MUTED).scale(0.34).next_to(ax2.x_axis, DOWN, buff=0.15).align_to(ax2.x_axis, RIGHT)
        yl = Text("inertia", font=FONT, color=MUTED).scale(0.34).next_to(ax2.y_axis, UP, buff=0.1)
        knum = VGroup(*[Text(str(k), font=FONT, color=MUTED).scale(0.32).next_to(ax2.c2p(k, 0), DOWN, buff=0.12) for k in range(1, 9)])
        curve = ax2.plot_line_graph(list(range(1, 9)), Js, line_color=MEAN, stroke_width=5, vertex_dot_radius=0.07, vertex_dot_style={"fill_color": MEAN})
        self.mark("elbow"); self.play(Create(ax2), FadeIn(xl), FadeIn(yl), FadeIn(knum), run_time=0.6)
        self.play(Create(curve), run_time=2.0)
        ring = Circle(radius=0.22, color=POINT, stroke_width=5).move_to(ax2.c2p(3, Js[2]))
        lab = Text("the elbow: K = 3", font=FONT, weight=SEMIBOLD, color=POINT).scale(0.48).next_to(ring, UR, buff=0.15)
        self.mark("elbow_k"); self.play(Create(ring), FadeIn(lab), run_time=0.8); self.wait(1.6)
        self.play(FadeOut(VGroup(ax2, xl, yl, knum, curve, ring, lab)), run_time=0.6)

        # ---------- 3. Starts ----------
        cap = self.swap(cap, self.caption("It depends on where you start", "a bad start gets stuck; k-means++ spreads the centers apart"))
        rng3 = np.random.default_rng(33)
        Q = blobs(rng3, [(2, 7.6), (7.8, 7.6), (2.2, 2.4), (8, 2.4)], 35, 0.55)
        ax3 = Axes(x_range=[0, 10], y_range=[0, 10], x_length=5.4, y_length=5.0, axis_config={"stroke_opacity": 0}, tips=False)
        def panel(C0, center, title, good):
            a = ax3.copy().move_to(center)
            hist = lloyd_steps(Q, np.array(C0, float), 30); C, L = hist[-1]
            J = ((Q - C[L]) ** 2).sum()
            g = VGroup(*[Dot(a.c2p(*q), radius=0.05, color=COLS[l]) for q, l in zip(Q, L)])
            starts = VGroup(*[self.xmark(a.c2p(*c), COLS[k], 0.12, hollow=True) for k, c in enumerate(C0)])
            ends = VGroup(*[self.xmark(a.c2p(*c), COLS[k], 0.13) for k, c in enumerate(C)])
            t = VGroup(Text(title, font=FONT, weight=SEMIBOLD, color=INK).scale(0.44),
                       Text(f"inertia {J:.0f}", font=FONT, color=GOOD if good else WARN).scale(0.42)).arrange(DOWN, buff=0.12).next_to(a, DOWN, buff=0.05)
            return g, starts, ends, t
        bad = panel([[1.6, 7.9], [2.4, 7.2], [7.6, 7.4], [5.0, 2.4]], [-3.4, -0.55, 0], "random start", False)
        good = panel([[2.1, 7.5], [7.9, 7.7], [2.1, 2.5], [8.1, 2.3]], [3.4, -0.55, 0], "k-means++", True)
        for side, (g, s, e, t) in (("bad", bad), ("good", good)):
            self.mark(side)
            self.play(FadeIn(g), FadeIn(s), run_time=0.7)
            self.play(LaggedStart(*[GrowFromCenter(x) for x in e], lag_ratio=0.15), FadeIn(t), run_time=0.9)
            self.wait(1.0)
        self.wait(1.0)
        self.play(FadeOut(VGroup(*bad, *good)), run_time=0.6)

        # ---------- 4. Shapes ----------
        cap = self.swap(cap, self.caption("It only sees round groups", "with curved shapes, it cuts in the wrong place"))
        rng4 = np.random.default_rng(4)
        t = np.pi * rng4.random(110)
        M = np.vstack([np.c_[3 + 2.8 * np.cos(t), 4.3 + 2.8 * np.sin(t)], np.c_[5.8 - 2.8 * np.cos(t), 5.7 - 2.8 * np.sin(t)]]) + 0.22 * rng4.standard_normal((220, 2))
        truth = np.r_[np.zeros(110), np.ones(110)].astype(int)
        L = KMeans(2, n_init=10, random_state=0).fit(M).labels_
        ax4 = Axes(x_range=[-0.5, 9.5], y_range=[0.5, 9.5], x_length=6.6, y_length=5.2, axis_config={"stroke_opacity": 0}, tips=False).move_to([0, -0.85, 0])
        md = VGroup(*[Dot(ax4.c2p(*m), radius=0.06, color=COLS[k]) for m, k in zip(M, truth)])
        l1 = Text("what we see: two moons", font=FONT, color=INK).scale(0.42).move_to([0, -3.75, 0])
        self.mark("moons"); self.play(LaggedStart(*[GrowFromCenter(d) for d in md], lag_ratio=0.005), FadeIn(l1), run_time=1.2); self.wait(0.8)
        l2 = Text("what K-means finds: two halves", font=FONT, weight=SEMIBOLD, color=WARN).scale(0.42).move_to([0, -3.75, 0])
        self.mark("kmoons"); self.play(*[d.animate.set_color(COLS[2 + l]) for d, l in zip(md, L)], Transform(l1, l2), run_time=1.2)
        self.wait(1.8)
        self.play(FadeOut(VGroup(md, l1)), run_time=0.6)

        # ---------- Closing ----------
        self.mark("legend")
        self.play(FadeOut(cap), run_time=0.6)
        e1 = Text("Assign, move, repeat.", font=FONT, weight=BOLD, color=INK).scale(0.85)
        e2 = Text("Choose K carefully, start well, and scale your data.", font=FONT, color=MUTED).scale(0.48)
        VGroup(e1, e2).arrange(DOWN, buff=0.4)
        self.mark("end_text"); self.play(Write(e1), run_time=1.3); self.play(FadeIn(e2, shift=0.2 * UP), run_time=0.8)
        self.wait(2.0)
        self.mark("fade"); self.play(FadeOut(VGroup(e1, e2)), run_time=1)
        self.marks["total"] = [round(self.renderer.time, 3)]
        json.dump(self.marks, open("marks.json", "w"), indent=1)
