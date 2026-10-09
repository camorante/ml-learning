import json
import numpy as np
from manim import *
from sklearn.neighbors import KNeighborsClassifier

# Dark palette of the visual ML series
BG = "#0E131B"; INK = "#E4EAF3"; MUTED = "#9BA7B9"; LINE = "#2B3545"
MEAN = "#6EA6FF"; SAMPLE = "#AE9FF3"; POINT = "#FF9A55"; GOOD = "#4CC38A"; WARN = "#F08A5D"
FONT = "Inter"
config.background_color = BG

def hexrgb(h):
    h = h.lstrip("#"); return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], float)
RB, RP, RS = hexrgb(BG), hexrgb(POINT), hexrgb(SAMPLE)

def bnd(x):
    return 5 + 1.8 * np.sin(x / 1.5)

def gen(rng, n, noise=0.12):
    P = rng.uniform(0.3, 9.7, (n, 2)); c = (P[:, 1] > bnd(P[:, 0])).astype(int)
    f = rng.random(n) < noise; c[f] = 1 - c[f]; return P, c

def prob_image(fn, res=150):
    g = (np.arange(res) + 0.5) / res * 10
    X, Y = np.meshgrid(g, g[::-1])
    p = fn(np.c_[X.ravel(), Y.ravel()]).reshape(res, res)
    a = (0.12 + 0.6 * np.abs(p - 0.5) * 2)[..., None]
    col = np.where((p > 0.5)[..., None], RP, RS)
    return (RB * (1 - a) + col * a).astype(np.uint8)


class Vecinos(Scene):
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

    def dot(self, p, c, r=0.08):
        return Dot(p, radius=r, color=POINT if c else SAMPLE).set_stroke(BG, 3, background=True)

    def diamond(self, p, s=0.17):
        return Square(side_length=s * 1.4, fill_color=INK, fill_opacity=1, stroke_color=BG, stroke_width=3).rotate(PI / 4).move_to(p)

    def construct(self):
        self.marks = {}
        title = Text("k-nearest neighbors", font=FONT, weight=BOLD, color=INK).scale(1.05)
        sub = Text("birds of a feather flock together", font=FONT, color=MUTED).scale(0.5)
        VGroup(title, sub).arrange(DOWN, buff=0.35)
        self.mark("title"); self.play(Write(title), run_time=1.4)
        self.mark("subtitle"); self.play(FadeIn(sub, shift=0.2 * UP)); self.wait(1.2)
        self.mark("title_out"); self.play(FadeOut(title), FadeOut(sub))

        # ---------- 1. They vote ----------
        cap = self.swap(None, self.caption("The nearest neighbors vote", "the new point copies whatever the majority says"))
        rng = np.random.default_rng(3)
        P, c = gen(rng, 80, 0.1)
        ax = Axes(x_range=[0, 10], y_range=[0, 10], x_length=5.6, y_length=5.6, axis_config={"stroke_opacity": 0}, tips=False).move_to([-2.6, -0.85, 0])
        frame = Polygon(ax.c2p(0, 0), ax.c2p(10, 0), ax.c2p(10, 10), ax.c2p(0, 10), stroke_color=LINE, stroke_width=2)
        dots = VGroup(*[self.dot(ax.c2p(*p), k) for p, k in zip(P, c)])
        self.mark("points"); self.play(Create(frame), LaggedStart(*[GrowFromCenter(d) for d in dots], lag_ratio=0.02), run_time=1.4)
        q = np.array([3.2, 5.8])
        dq = self.diamond(ax.c2p(*q))
        self.mark("query"); self.play(GrowFromCenter(dq), run_time=0.6)
        order = np.argsort(((P - q) ** 2).sum(1))
        unit = ax.c2p(1, 0)[0] - ax.c2p(0, 0)[0]
        circ = lines = rings = res = None
        for k in (1, 3, 7):
            nn = order[:k]; r = np.sqrt(((P[nn[-1]] - q) ** 2).sum()) + 0.12
            nc = Circle(radius=r * unit, color=MEAN, stroke_width=2.5, fill_color=MEAN, fill_opacity=0.12).move_to(ax.c2p(*q))
            nl = VGroup(*[Line(ax.c2p(*q), ax.c2p(*P[i]), color=INK, stroke_width=2, stroke_opacity=0.6) for i in nn])
            nr = VGroup(*[Circle(radius=0.14, color=INK, stroke_width=2.5).move_to(ax.c2p(*P[i])) for i in nn])
            o = int(c[nn].sum()); v = k - o
            win = "orange" if o > v else "purple"
            nres = VGroup(Text(f"k = {k}", font=FONT, weight=BOLD, color=INK).scale(0.6),
                          Text(f"{o} orange · {v} purple", font=FONT, color=MUTED).scale(0.45),
                          Text(f"→ {win}", font=FONT, weight=SEMIBOLD, color=POINT if o > v else SAMPLE).scale(0.55)).arrange(DOWN, buff=0.25, aligned_edge=LEFT).move_to([3.4, -0.6, 0])
            self.mark("k")
            if circ is None:
                self.play(GrowFromCenter(nc), Create(nl), FadeIn(nr), FadeIn(nres), run_time=1.0)
            else:
                self.play(Transform(circ, nc), Transform(lines, nl), Transform(rings, nr), Transform(res, nres), run_time=1.0)
                nc, nl, nr, nres = circ, lines, rings, res
            circ, lines, rings, res = nc, nl, nr, nres
            self.bring_to_front(dq); self.wait(1.1)
        self.play(FadeOut(VGroup(frame, dots, dq, circ, lines, rings, res)), run_time=0.6)

        # ---------- 2. How many neighbors ----------
        cap = self.swap(cap, self.caption("How many neighbors?", "small k is jumpy; large k is blurry"))
        rng2 = np.random.default_rng(17)
        P, c = gen(rng2, 160); PT, cT = gen(rng2, 1500)
        cur = None; lab = None
        curve = None
        for k in (1, 15, 101):
            m = KNeighborsClassifier(k).fit(P, c)
            im = ImageMobject(prob_image(lambda Q: m.predict_proba(Q)[:, 1])); im.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
            im.height = 5.3; im.move_to([-2.2, -0.75, 0])
            x0, y0 = -2.2 - 2.65, -0.75 - 2.65
            g = Group(im, *[self.dot([x0 + p[0] / 10 * 5.3, y0 + p[1] / 10 * 5.3, 0], kk, 0.055) for p, kk in zip(P, c)])
            acc = (m.predict(PT) == cT).mean()
            note = {1: "copies the noise", 15: "follows the shape", 101: "no longer sees the curve"}[k]
            nl = VGroup(Text(f"k = {k}", font=FONT, weight=BOLD, color=INK).scale(0.6),
                        Text(note, font=FONT, color=MUTED).scale(0.45),
                        Text(f"new data: {acc:.0%}", font=FONT, weight=SEMIBOLD, color=GOOD if k == 15 else WARN).scale(0.5)).arrange(DOWN, buff=0.25, aligned_edge=LEFT).move_to([3.6, -0.6, 0])
            self.mark("kmap")
            if cur is None:
                bc = DashedVMobject(VMobject().set_points_smoothly([[x0 + xx / 10 * 5.3, y0 + bnd(xx) / 10 * 5.3, 0] for xx in np.linspace(0, 10, 40)]), num_dashes=30).set_stroke(GOOD, 4)
                self.play(FadeIn(g), FadeIn(nl), run_time=0.9); curve = bc; self.play(Create(curve), run_time=0.6)
            else:
                self.play(FadeOut(cur), FadeIn(g), FadeTransform(lab, nl), run_time=0.9)
            self.bring_to_front(curve)
            cur, lab = g, nl
            self.wait(1.4)
        self.play(FadeOut(Group(cur, lab, curve)), run_time=0.6)

        # ---------- 3. Scale ----------
        cap = self.swap(cap, self.caption("Watch the units", "without scaling, kilometers crush temperature"))
        T = [("A", 4000, 98, 1), ("B", 6000, 88, 0), ("C", 9000, 101, 1), ("D", 12000, 90, 0), ("E", 15000, 99, 1), ("F", 18000, 87, 0), ("G", 21000, 92, 0), ("H", 25000, 100, 1)]
        Q = (14000, 97)
        ax3 = Axes(x_range=[2000, 27000, 5000], y_range=[85, 103, 4], x_length=9.5, y_length=4.3, tips=False,
                   axis_config={"color": LINE, "stroke_width": 2, "include_ticks": False}).move_to([0, -0.8, 0])
        xl = Text("km since service", font=FONT, color=MUTED).scale(0.32).next_to(ax3.x_axis, DOWN, buff=0.12).align_to(ax3.x_axis, RIGHT)
        yl = Text("engine temperature", font=FONT, color=MUTED).scale(0.32).next_to(ax3.y_axis, UP, buff=0.1).shift(RIGHT * 1.0)
        tr = VGroup(*[VGroup(self.dot(ax3.c2p(x, y), cc, 0.13), Text(n, font=FONT, color=INK).scale(0.36).next_to(ax3.c2p(x, y), RIGHT, buff=0.18)) for n, x, y, cc in T])
        dq = self.diamond(ax3.c2p(*Q), 0.2)
        self.mark("trucks"); self.play(Create(ax3), FadeIn(xl), FadeIn(yl), LaggedStart(*[FadeIn(t, scale=0.6) for t in tr], lag_ratio=0.1), GrowFromCenter(dq), run_time=1.6)
        def nn3(scaled):
            lo, hi = np.array([4000, 87.]), np.array([25000, 101.])
            X = np.array([[x, y] for _, x, y, _ in T], float); q = np.array(Q, float)
            if scaled: X = (X - lo) / (hi - lo); q = (q - lo) / (hi - lo)
            d = np.sqrt(((X - q) ** 2).sum(1)); return np.argsort(d)[:3], np.sort(d)[2]
        idx, R = nn3(False)
        zone = Rectangle(width=ax3.c2p(Q[0] + R, 0)[0] - ax3.c2p(Q[0] - R, 0)[0], height=ax3.c2p(0, 103)[1] - ax3.c2p(0, 85)[1],
                         stroke_color=MEAN, stroke_width=2.5, fill_color=MEAN, fill_opacity=0.12).move_to([ax3.c2p(Q[0], 0)[0], ax3.c2p(0, 94)[1], 0])
        rings = VGroup(*[Circle(radius=0.22, color=INK, stroke_width=3).move_to(ax3.c2p(T[i][1], T[i][2])) for i in idx])
        verdict = Text("neighbors D, E, F → 1 of 3 broke down: \"no risk\"", font=FONT, color=SAMPLE).scale(0.42).move_to([0, -3.75, 0])
        self.mark("raw"); self.play(FadeIn(zone), Create(rings), FadeIn(verdict), run_time=1.0); self.bring_to_front(tr, dq)
        self.wait(1.8)
        idx2, R2 = nn3(True)
        rx = R2 * (ax3.c2p(25000, 0)[0] - ax3.c2p(4000, 0)[0]); ry = R2 * (ax3.c2p(0, 101)[1] - ax3.c2p(0, 87)[1])
        zone2 = Ellipse(width=2 * rx, height=2 * ry, stroke_color=MEAN, stroke_width=2.5, fill_color=MEAN, fill_opacity=0.12).move_to(ax3.c2p(*Q))
        rings2 = VGroup(*[Circle(radius=0.22, color=INK, stroke_width=3).move_to(ax3.c2p(T[i][1], T[i][2])) for i in idx2])
        verdict2 = Text("scaled 0 to 1: neighbors A, C, E → 3 of 3: \"at risk\"", font=FONT, weight=SEMIBOLD, color=POINT).scale(0.42).move_to([0, -3.75, 0])
        self.mark("scaled"); self.play(Transform(zone, zone2), Transform(rings, rings2), Transform(verdict, verdict2), run_time=1.3); self.bring_to_front(tr, dq)
        self.wait(2.0)
        self.play(FadeOut(VGroup(ax3, xl, yl, tr, dq, zone, rings, verdict)), run_time=0.6)

        # ---------- 4. Dimension ----------
        cap = self.swap(cap, self.caption("With many features, everyone is far away", "the curse of dimensionality"))
        rng4 = np.random.default_rng(31)
        base = Line([-5, -3.0, 0], [5, -3.0, 0], color=LINE, stroke_width=2)
        l0 = Text("0", font=FONT, color=MUTED).scale(0.32).next_to(base.get_left(), DOWN, buff=0.12)
        l1 = Text("the largest distance", font=FONT, color=MUTED).scale(0.32).next_to(base.get_right(), DOWN, buff=0.12).shift(LEFT * 0.8)
        def hist(d):
            q = rng4.random(d); D = np.sqrt(((rng4.random((500, d)) - q) ** 2).sum(1)); D = D / D.max()
            h, _ = np.histogram(D, bins=40, range=(0, 1)); h = h / h.max()
            bars = VGroup(*[Rectangle(width=10 / 40 - 0.04, height=max(0.001, 4.2 * v), stroke_width=0, fill_color=MEAN, fill_opacity=1).move_to([-5 + (i + 0.5) * 10 / 40, -3.0 + 2.1 * max(0.001, v), 0]) for i, v in enumerate(h)])
            mn = D.min()
            ln = Line([-5 + mn * 10, -3.0, 0], [-5 + mn * 10, 1.4, 0], color=POINT, stroke_width=4)
            lab = Text(f"{d} feature{'s' if d > 1 else ''}: the nearest is at {mn:.0%} of the farthest", font=FONT, color=INK).scale(0.4).move_to([0, 1.75, 0])
            return VGroup(bars, ln, lab)
        cur = None
        for d in (2, 10, 100, 500):
            h = hist(d)
            self.mark("dim")
            if cur is None: self.play(Create(base), FadeIn(l0), FadeIn(l1), FadeIn(h), run_time=1.0)
            else: self.play(Transform(cur, h), run_time=1.0); h = cur
            cur = h; self.wait(1.0)
        self.wait(0.6)
        self.play(FadeOut(VGroup(cur, base, l0, l1)), run_time=0.6)

        # ---------- Closing ----------
        self.mark("legend")
        self.play(FadeOut(cap), run_time=0.6)
        e1 = Text("Find the ones that look alike", font=FONT, weight=BOLD, color=INK).scale(0.75)
        e2 = Text("and copy what happened to them.", font=FONT, weight=BOLD, color=INK).scale(0.75)
        e3 = Text("Scale your data, choose k well, and don't use too many features.", font=FONT, color=MUTED).scale(0.45)
        VGroup(e1, e2, e3).arrange(DOWN, buff=0.35)
        self.mark("end_text"); self.play(Write(e1), run_time=1.1); self.play(Write(e2), run_time=1.0); self.play(FadeIn(e3, shift=0.2 * UP), run_time=0.8)
        self.wait(2.0)
        self.mark("fade"); self.play(FadeOut(VGroup(e1, e2, e3)), run_time=1)
        self.marks["total"] = [round(self.renderer.time, 3)]
        json.dump(self.marks, open("marks.json", "w"), indent=1)
