import json
import numpy as np
from manim import *
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier

# Dark palette of the Visual ML series
BG = "#0E131B"; INK = "#E4EAF3"; MUTED = "#9BA7B9"; LINE = "#2B3545"
MEAN = "#6EA6FF"; SAMPLE = "#AE9FF3"; POINT = "#FF9A55"; GOOD = "#4CC38A"; WARN = "#F08A5D"
FONT = "Inter"
config.background_color = BG

def hexrgb(h):
    h = h.lstrip("#"); return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], float)

RB, RP, RS = hexrgb(BG), hexrgb(POINT), hexrgb(SAMPLE)

def truth(P):
    return (((P[:, 0] - 5) ** 2 / 9 + (P[:, 1] - 5) ** 2 / 5) < 1.6).astype(int)

def gen(rng, n, noise=0.1):
    P = rng.uniform(0.3, 9.7, (n, 2)); c = truth(P)
    f = rng.random(n) < noise; c[f] = 1 - c[f]; return P, c

def prob_image(prob_fn, res=160):
    """RGB image of the probability map (row 0 = top)."""
    g = (np.arange(res) + 0.5) / res * 10
    X, Y = np.meshgrid(g, g[::-1])
    p = prob_fn(np.c_[X.ravel(), Y.ravel()]).reshape(res, res)
    a = (0.12 + 0.72 * np.abs(p - 0.5) * 2)[..., None]
    col = np.where((p > 0.5)[..., None], RP, RS)
    img = RB * (1 - a) + col * a
    return img.astype(np.uint8)


class Ensambles(Scene):
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

    def map_mob(self, prob_fn, center, size, P=None, c=None, r=0.06, res=160):
        im = ImageMobject(prob_image(prob_fn, res)); im.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
        im.height = size; im.move_to(center)
        frame = Square(side_length=size, stroke_color=LINE, stroke_width=2).move_to(center)
        g = Group(im, frame)
        if P is not None:
            x0, y0 = center[0] - size / 2, center[1] - size / 2
            g.add(*[self.dot([x0 + p[0] / 10 * size, y0 + p[1] / 10 * size, 0], k, r) for p, k in zip(P, c)])
        return g

    def construct(self):
        self.marks = {}
        rng = np.random.default_rng(11)
        title = Text("Random forest and boosting", font=FONT, weight=BOLD, color=INK).scale(1.0)
        sub = Text("many trees think better than one", font=FONT, color=MUTED).scale(0.5)
        VGroup(title, sub).arrange(DOWN, buff=0.35)
        self.mark("title"); self.play(Write(title), run_time=1.4)
        self.mark("subtitle"); self.play(FadeIn(sub, shift=0.2 * UP)); self.wait(1.2)
        self.mark("title_out"); self.play(FadeOut(title), FadeOut(sub))

        # ---------- 1. A tree is jumpy ----------
        cap = self.swap(None, self.caption("A single tree is jumpy", "another sample of the same data, another tree"))
        cur = None
        for k in range(3):
            P, c = gen(rng, 70)
            t = DecisionTreeClassifier(random_state=0).fit(P, c)
            m = self.map_mob(lambda Q: t.predict_proba(Q)[:, 1], [0, -0.75, 0], 5.6, P, c)
            self.mark("nervous")
            if cur is None: self.play(FadeIn(m), run_time=0.9)
            else: self.play(FadeOut(cur), FadeIn(m), run_time=0.7)
            cur = m; self.wait(0.9)
        self.play(FadeOut(cur), run_time=0.5)

        # ---------- 2. Bootstrap ----------
        cap = self.swap(cap, self.caption("Each tree gets a different sample", "drawing with replacement: some repeat, others are left out"))
        xs = np.linspace(-5.4, 5.4, 10)
        balls = VGroup(*[VGroup(Circle(radius=0.36, color=MUTED, stroke_width=3, fill_color=BG, fill_opacity=1),
                                Text(str(i + 1), font=FONT, color=INK).scale(0.42)).move_to([x, 1.2, 0]) for i, x in enumerate(xs)])
        lab_a = Text("data", font=FONT, color=MUTED).scale(0.36).next_to(balls, LEFT, buff=0.25)
        self.mark("balls"); self.play(LaggedStart(*[GrowFromCenter(b) for b in balls], lag_ratio=0.08), FadeIn(lab_a), run_time=1.2)
        draws = [2, 6, 2, 9, 0, 3, 6, 9, 6, 4]
        bag = VGroup()
        for k, d in enumerate(draws):
            cp = balls[d].copy(); cp[0].set_stroke(MEAN); cp[0].set_fill(MEAN, 0.25)
            tgt = [xs[k], -0.6, 0]
            self.mark("draw")
            self.play(cp.animate.move_to(tgt), Indicate(balls[d], color=MEAN, scale_factor=1.15), run_time=0.42)
            bag.add(cp)
        lab_b = Text("sample", font=FONT, color=MEAN).scale(0.36).next_to(bag, LEFT, buff=0.25)
        out = [i for i in range(10) if i not in draws]
        self.mark("oob")
        self.play(FadeIn(lab_b), *[balls[i].animate.set_opacity(0.3) for i in out], run_time=0.8)
        oobt = Text(f"{len(out)} of 10 were left out: they are used to test the tree", font=FONT, color=INK).scale(0.42).move_to([0, -2.1, 0])
        oobt2 = Text("on average, ≈ 37% is always left out", font=FONT, color=GOOD).scale(0.4).move_to([0, -2.75, 0])
        self.play(FadeIn(oobt), run_time=0.6); self.play(FadeIn(oobt2), run_time=0.6)
        self.wait(1.4)
        self.play(FadeOut(VGroup(balls, bag, lab_a, lab_b, oobt, oobt2)), run_time=0.6)

        # ---------- 3. A forest that votes ----------
        cap = self.swap(cap, self.caption("Random forest: many trees that vote", "their quirks don't line up, so they cancel out"))
        P, c = gen(np.random.default_rng(11), 120)
        PT, cT = gen(np.random.default_rng(99), 1500)
        rf = RandomForestClassifier(n_estimators=100, max_features=1, min_samples_leaf=2, random_state=3).fit(P, c)
        ests = rf.estimators_[:9]
        smalls = Group()
        for i, e in enumerate(ests):
            cx = -5.2 + (i % 3) * 1.7; cy = 0.55 - (i // 3) * 1.7
            smalls.add(self.map_mob(lambda Q, e=e: e.predict_proba(Q)[:, 1], [cx, cy, 0], 1.55, res=80))
        self.mark("trees"); self.play(LaggedStart(*[FadeIn(s, scale=0.8) for s in smalls], lag_ratio=0.12), run_time=1.6)
        self.wait(0.6)
        big = self.map_mob(lambda Q: rf.predict_proba(Q)[:, 1], [3.4, -0.55, 0], 4.6, P, c, 0.055)
        arrow = Arrow([-0.15, -0.55, 0], [0.95, -0.55, 0], color=MUTED, buff=0)
        vote = Text("vote", font=FONT, color=MUTED).scale(0.34).next_to(arrow, UP, buff=0.08)
        self.mark("vote"); self.play(GrowArrow(arrow), FadeIn(vote), FadeIn(big), run_time=1.2)
        one = DecisionTreeClassifier(random_state=0).fit(P, c)
        a1 = (one.predict(PT) == cT).mean(); aF = (rf.predict(PT) == cT).mean()
        res = VGroup(Text(f"new data: one tree gets {a1:.0%} right", font=FONT, color=INK).scale(0.36), Text(f"the forest of 100 trees, {aF:.0%}", font=FONT, weight=SEMIBOLD, color=GOOD).scale(0.4)).arrange(DOWN, buff=0.12).move_to([3.4, -3.45, 0])
        self.mark("acc"); self.play(FadeIn(res, shift=0.15 * UP), run_time=0.8)
        self.wait(2.0)
        self.play(FadeOut(Group(smalls, big, arrow, vote, res)), run_time=0.6)

        # ---------- 4. Boosting ----------
        cap = self.swap(cap, self.caption("Boosting: each tree fixes the one before it", "the next tree only looks at the errors that are left"))
        r2 = np.random.default_rng(14)
        X = np.linspace(0.4, 9.6, 16) + r2.uniform(-0.15, 0.15, 16)
        g1 = lambda x: 3 + 2 * np.sin(x / 1.4) + 0.35 * x
        Y = g1(X) + 0.5 * r2.standard_normal(16)
        ax = Axes(x_range=[0, 10], y_range=[0, 10], x_length=11, y_length=4.6, axis_config={"stroke_opacity": 0}, tips=False).move_to([0, -1.05, 0])
        base = Line(ax.c2p(0, 0), ax.c2p(10, 0), color=LINE, stroke_width=2)
        pts = VGroup(*[self.dot(ax.c2p(x, y), 1, 0.09) for x, y in zip(X, Y)])
        self.mark("bpoints"); self.play(Create(base), LaggedStart(*[GrowFromCenter(p) for p in pts], lag_ratio=0.05), run_time=1.0)
        lr = 0.5; F = np.full(16, Y.mean()); trees = []
        xs_ = np.linspace(0, 10, 600)
        def model(x):
            v = np.full_like(x, Y.mean(), dtype=float)
            for t in trees: v += lr * t.predict(x[:, None])
            return v
        def curve():
            return VMobject(color=MEAN, stroke_width=5).set_points_as_corners([ax.c2p(x, y) for x, y in zip(xs_, model(xs_))])
        def resid():
            m = model(X)
            return VGroup(*[Line(ax.c2p(x, y), ax.c2p(x, f), color=SAMPLE, stroke_width=3, stroke_opacity=0.8) for x, y, f in zip(X, Y, m)])
        def info(k):
            e = np.sqrt(np.mean((Y - model(X)) ** 2))
            return Text(f"trees: {k}   ·   typical error: {e:.2f}", font=FONT, color=INK).scale(0.4).move_to([0, -3.75, 0])
        cv, rs, inf = curve(), resid(), info(0)
        self.mark("mean"); self.play(Create(cv), FadeIn(rs), FadeIn(inf), run_time=1.0); self.bring_to_front(pts)
        self.wait(0.6)
        for k in range(1, 9):
            R = Y - model(X)
            trees.append(DecisionTreeRegressor(max_depth=1).fit(X[:, None], R))
            ncv, nrs, ninf = curve(), resid(), info(k)
            self.mark("boost")
            self.play(Transform(cv, ncv), Transform(rs, nrs), Transform(inf, ninf), run_time=0.7 if k < 4 else 0.45)
            self.bring_to_front(pts); self.wait(0.35 if k < 4 else 0.15)
        for _ in range(30):
            R = Y - model(X); trees.append(DecisionTreeRegressor(max_depth=1).fit(X[:, None], R))
        self.mark("boost_many"); self.play(Transform(cv, curve()), Transform(rs, resid()), Transform(inf, info(len(trees))), run_time=1.2)
        self.bring_to_front(pts); self.wait(1.4)
        self.play(FadeOut(VGroup(cv, rs, inf, pts, base)), run_time=0.6)

        # ---------- 5. Stop in time ----------
        cap = self.swap(cap, self.caption("But you have to stop in time", "with too many trees, it chases the noise"))
        r3 = np.random.default_rng(4)
        Xa = r3.uniform(0.2, 9.8, 40); Ya = g1(Xa) + 0.8 * r3.standard_normal(40)
        Xb = r3.uniform(0.2, 9.8, 600); Yb = g1(Xb) + 0.8 * r3.standard_normal(600)
        Fa = np.full(40, Ya.mean()); Fb = np.full(600, Ya.mean()); tr = [np.mean((Fa - Ya) ** 2)]; te = [np.mean((Fb - Yb) ** 2)]
        for _ in range(200):
            t = DecisionTreeRegressor(max_depth=2, min_samples_leaf=3).fit(Xa[:, None], Ya - Fa)
            Fa += 0.2 * t.predict(Xa[:, None]); Fb += 0.2 * t.predict(Xb[:, None]); tr.append(np.mean((Fa - Ya) ** 2)); te.append(np.mean((Fb - Yb) ** 2))
        best = int(np.argmin(te))
        ax3 = Axes(x_range=[0, 200, 50], y_range=[0, 2.2, 0.5], x_length=9, y_length=4.4, tips=False,
                   axis_config={"color": LINE, "stroke_width": 2, "include_ticks": False}).move_to([0, -1.0, 0])
        xl = Text("trees", font=FONT, color=MUTED).scale(0.34).next_to(ax3.x_axis, DOWN, buff=0.15).align_to(ax3.x_axis, RIGHT)
        yl = Text("error", font=FONT, color=MUTED).scale(0.34).next_to(ax3.y_axis, UP, buff=0.1)
        ltr = ax3.plot_line_graph(np.arange(201), tr, add_vertex_dots=False, line_color=POINT, stroke_width=5)
        lte = ax3.plot_line_graph(np.arange(201), te, add_vertex_dots=False, line_color=INK, stroke_width=5)
        t_tr = Text("training data", font=FONT, color=POINT).scale(0.34).next_to(ax3.c2p(200, tr[-1]), UP, buff=0.12).shift(LEFT * 1.0)
        t_te = Text("new data", font=FONT, color=INK).scale(0.34).next_to(ax3.c2p(200, te[-1]), UP, buff=0.12).shift(LEFT * 0.8)
        self.mark("curves"); self.play(Create(ax3), FadeIn(xl), FadeIn(yl), run_time=0.6)
        self.play(Create(ltr), Create(lte), run_time=2.4, rate_func=linear)
        self.play(FadeIn(t_tr), FadeIn(t_te), run_time=0.5)
        bd = Dot(ax3.c2p(best, te[best]), radius=0.11, color=GOOD)
        bl = DashedLine(ax3.c2p(best, 0), ax3.c2p(best, 2.1), color=GOOD, stroke_width=3)
        bt = Text(f"early stopping: {best} trees", font=FONT, weight=SEMIBOLD, color=GOOD).scale(0.4).next_to(ax3.c2p(best, 2.1), RIGHT, buff=0.15)
        self.mark("stop"); self.play(Create(bl), GrowFromCenter(bd), FadeIn(bt), run_time=0.9)
        self.wait(2.0)
        self.play(FadeOut(VGroup(ax3, xl, yl, ltr, lte, t_tr, t_te, bd, bl, bt)), run_time=0.6)

        # ---------- Closing ----------
        self.mark("legend")
        self.play(FadeOut(cap), run_time=0.6)
        l1 = Text("Forest: many trees that vote at the same time.", font=FONT, weight=BOLD, color=INK).scale(0.6)
        l2 = Text("Boosting: small trees that correct each other in a row.", font=FONT, weight=BOLD, color=INK).scale(0.6)
        l3 = Text("Together, the kings of tabular data.", font=FONT, color=MUTED).scale(0.48)
        VGroup(l1, l2, l3).arrange(DOWN, buff=0.4)
        self.mark("end_text"); self.play(Write(l1), run_time=1.2); self.play(Write(l2), run_time=1.2); self.play(FadeIn(l3, shift=0.2 * UP), run_time=0.8)
        self.wait(2.0)
        self.mark("fade"); self.play(FadeOut(VGroup(l1, l2, l3)), run_time=1)
        self.marks["total"] = [round(self.renderer.time, 3)]
        self.marks["accs"] = [float(a1), float(aF), best]
        json.dump(self.marks, open("marks.json", "w"), indent=1)
