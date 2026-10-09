import json
import numpy as np
from manim import *

# Dark palette of the visual ML series
BG = "#0E131B"; INK = "#E4EAF3"; MUTED = "#9BA7B9"; LINE = "#2B3545"
MEAN = "#6EA6FF"; SAMPLE = "#AE9FF3"; POINT = "#FF9A55"; GOOD = "#4CC38A"; WARN = "#F08A5D"
FONT = "Inter"
config.background_color = BG


class Gradiente(Scene):
    def caption(self, text, sub=None):
        t = Text(text, font=FONT, weight=SEMIBOLD, color=INK).scale(0.62)
        grp = VGroup(t)
        if sub:
            grp.add(Text(sub, font=FONT, color=MUTED).scale(0.42)); grp.arrange(DOWN, buff=0.18)
        grp.to_edge(UP, buff=0.4)
        band = Rectangle(width=config.frame_width + 1, height=1.55, stroke_width=0, fill_color=BG, fill_opacity=1)
        band.move_to([0, config.frame_height / 2 - 0.72, 0])
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

    def ball(self, p, r=0.13):
        return Dot(p, radius=r, color=POINT).set_stroke(BG, 4, background=True)

    def contours(self, a, b, ang, center=ORIGIN, levels=(0.15, 0.5, 1.1, 2.0, 3.2, 4.8, 6.8, 9.5)):
        g = VGroup()
        for i, c in enumerate(levels):
            e = Ellipse(width=2 * np.sqrt(2 * c / a), height=2 * np.sqrt(2 * c / b), color=MEAN,
                        stroke_width=2, stroke_opacity=0.65 - 0.06 * i, fill_color=MEAN, fill_opacity=0.10)
            e.rotate(ang).move_to(center)
            g.add(e)
        return g

    def construct(self):
        self.marks = {}
        title = Text("Gradient descent", font=FONT, weight=BOLD, color=INK).scale(1.1)
        sub = Text("how almost all of machine learning learns", font=FONT, color=MUTED).scale(0.5)
        VGroup(title, sub).arrange(DOWN, buff=0.35)
        self.mark("title"); self.play(Write(title), run_time=1.4)
        self.mark("subtitle"); self.play(FadeIn(sub, shift=0.2 * UP)); self.wait(1.2)
        self.mark("title_out"); self.play(FadeOut(title), FadeOut(sub))

        # ---------- 1. Mountain in the fog ----------
        cap = self.swap(None, self.caption("Walking down a mountain in the fog", "you only feel the slope under your feet"))
        f = lambda x: 0.11 * (x - 1.2) ** 2 - 2.6
        df = lambda x: 0.22 * (x - 1.2)
        ax = Axes(x_range=[-6.5, 6.5], y_range=[-3, 2.5], x_length=13, y_length=5.5,
                  axis_config={"stroke_opacity": 0}, tips=False).shift(0.6 * DOWN)
        hill = ax.plot(f, x_range=[-6.3, 6.3], color=MEAN, stroke_width=5)
        fill = ax.get_area(hill, x_range=[-6.3, 6.3], bounded_graph=ax.plot(lambda x: -3.2), color=MEAN, opacity=0.15)
        fog = Rectangle(width=8, height=5.6, stroke_width=0, fill_color=BG, fill_opacity=0.82).move_to([3.6, -1.0, 0])
        self.mark("hill"); self.play(Create(hill), FadeIn(fill), run_time=1.2)
        self.play(FadeIn(fog), run_time=0.6)
        x = -5.6
        b = self.ball(ax.c2p(x, f(x)))
        self.mark("ball"); self.play(GrowFromCenter(b), run_time=0.5)
        for k in range(7):
            m = df(x)
            tan = Line(ax.c2p(x - 1, f(x) - m), ax.c2p(x + 1, f(x) + m), color=SAMPLE, stroke_width=4)
            xn = x - 2.2 * m
            self.mark("step1")
            self.play(Create(tan), run_time=0.25)
            self.play(b.animate.move_to(ax.c2p(xn, f(xn))), FadeOut(tan), fog.animate.shift(RIGHT * 0.35), run_time=0.55)
            x = xn
        self.mark("bottom"); self.play(Flash(b, color=GOOD, line_length=0.25, flash_radius=0.3), run_time=0.7)
        self.wait(0.6)
        self.play(FadeOut(VGroup(hill, fill, fog, b)), run_time=0.6)

        # ---------- 2. Step size ----------
        cap = self.swap(cap, self.caption("The step size changes everything", "too small · just right · too big"))
        panels = VGroup()
        balls, paths, labels = [], [], []
        etas = [0.08, 0.6, 2.15]
        names = ["slow", "gets there fast", "blows up!"]
        cols = [MUTED, GOOD, WARN]
        for i, eta in enumerate(etas):
            a = Axes(x_range=[-4.5, 4.5], y_range=[-0.3, 10], x_length=3.9, y_length=3.6, axis_config={"stroke_opacity": 0}, tips=False)
            a.move_to([-4.4 + 4.4 * i, -0.7, 0])
            c = a.plot(lambda x: 0.5 * x * x, x_range=[-4.4, 4.4], color=MEAN, stroke_width=4)
            lab = Text(names[i], font=FONT, color=cols[i]).scale(0.42).next_to(a, DOWN, buff=0.15)
            panels.add(VGroup(a, c)); labels.append(lab)
            xs = [-3.4]
            for _ in range(9):
                xs.append(xs[-1] - eta * xs[-1])
            balls.append((a, xs))
        self.mark("panels"); self.play(*[Create(p[1]) for p in panels], *[FadeIn(l) for l in labels], run_time=1.0)
        dots = [self.ball(a.c2p(xs[0], 0.5 * xs[0] ** 2), 0.11) for a, xs in balls]
        self.play(*[GrowFromCenter(d) for d in dots], run_time=0.4)
        trails = [VGroup() for _ in balls]
        for k in range(1, 8):
            anims = []
            for j, (a, xs) in enumerate(balls):
                x0, x1 = xs[k - 1], xs[k]
                y1 = min(0.5 * x1 * x1, 10.5)
                xc = np.clip(x1, -4.6, 4.6)
                seg = Line(a.c2p(np.clip(x0, -4.6, 4.6), min(0.5 * x0 * x0, 10.5)), a.c2p(xc, y1), color=POINT, stroke_width=2.5, stroke_opacity=0.7)
                trails[j].add(seg)
                anims += [Create(seg), dots[j].animate.move_to(a.c2p(xc, y1))]
            self.mark("step2")
            self.play(*anims, run_time=0.42)
        self.wait(1.0)
        self.play(*[FadeOut(m) for m in [*panels, *labels, *dots, *trails]], run_time=0.6)

        # ---------- 3. Long valley ----------
        cap = self.swap(cap, self.caption("Long valleys: zigzag", "the features are on very different scales"))
        A_ = (0.08, 1.6); ang = 0.5; c0 = np.array([0.4, -0.7, 0])
        R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
        H = R @ np.diag(A_) @ R.T
        cont = self.contours(A_[0], A_[1], ang, c0)
        self.mark("valley"); self.play(LaggedStart(*[Create(e) for e in cont], lag_ratio=0.08), run_time=1.3); self.bring_to_front(cap)

        def path(Hm, eta, start, n, mom=0.0):
            p = np.array(start, float); v = np.zeros(2); P = [p.copy()]
            for _ in range(n):
                g = Hm @ p
                v = mom * v - eta * g; p = p + v; P.append(p.copy())
            return [c0 + np.array([q[0], q[1], 0]) for q in P]

        P1 = path(H, 1.05, [-4.6, 1.4], 40)
        zz = VMobject(color=POINT, stroke_width=3.5).set_points_as_corners(P1)
        bz = self.ball(P1[0])
        self.play(GrowFromCenter(bz), run_time=0.3)
        self.mark("zigzag"); self.play(Create(zz), MoveAlongPath(bz, zz), run_time=3.2, rate_func=linear)
        self.wait(0.6)

        cap = self.swap(cap, self.caption("Same scale: a direct path", "standardizing the data rounds out the valley"))
        round_c = self.contours(0.8, 0.8, 0, c0)
        P2 = path(np.eye(2) * 0.8, 0.55, [-4.6, 1.4], 14)
        st = VMobject(color=POINT, stroke_width=3.5).set_points_as_corners(P2)
        self.mark("round"); self.play(Transform(cont, round_c), FadeOut(zz), bz.animate.move_to(P2[0]), run_time=1.2); self.bring_to_front(cap)
        self.play(Create(st), MoveAlongPath(bz, st), run_time=1.4, rate_func=linear)
        self.mark("arrive"); self.play(Flash(bz, color=GOOD, line_length=0.25, flash_radius=0.3), run_time=0.6)
        self.wait(0.6)

        cap = self.swap(cap, self.caption("With momentum, the bounces cancel out", "the ball remembers where it was heading"))
        long_c = self.contours(A_[0], A_[1], ang, c0)
        P3 = path(H, 0.25, [-4.6, 1.4], 70, mom=0.8)
        mm = VMobject(color=POINT, stroke_width=3.5).set_points_smoothly(P3)
        self.mark("momentum"); self.play(Transform(cont, long_c), FadeOut(st), bz.animate.move_to(P3[0]), run_time=1.0); self.bring_to_front(cap)
        self.play(Create(mm), MoveAlongPath(bz, mm), run_time=2.6, rate_func=linear)
        self.wait(0.8)
        self.play(FadeOut(VGroup(cont, mm, bz)), run_time=0.6)

        # ---------- 4. False holes ----------
        cap = self.swap(cap, self.caption("False holes", "each ball falls into the nearest hole"))
        g = lambda x: 0.035 * x ** 2 - 1.9 * np.exp(-((x - 2.6) ** 2) / 1.3) - 1.0 * np.exp(-((x + 2.8) ** 2) / 1.0) + 0.3
        dg = lambda x, h=1e-4: (g(x + h) - g(x - h)) / (2 * h)
        ax2 = Axes(x_range=[-6.5, 6.5], y_range=[-2.2, 2], x_length=13, y_length=4.4, axis_config={"stroke_opacity": 0}, tips=False).shift(0.9 * DOWN)
        cur = ax2.plot(g, x_range=[-6.3, 6.3], color=MEAN, stroke_width=5)
        self.mark("holes"); self.play(Create(cur), run_time=1.0)
        starts = [-5.4, -1.2, 5.6]
        hb = [self.ball(ax2.c2p(s, g(s)) + 0.12 * UP) for s in starts]
        self.play(*[GrowFromCenter(h) for h in hb], run_time=0.4)
        xs = list(starts)
        for k in range(10):
            xs = [x - 0.9 * dg(x) for x in xs]
            self.mark("step4") if k % 3 == 0 else None
            self.play(*[h.animate.move_to(ax2.c2p(x, g(x)) + 0.12 * UP) for h, x in zip(hb, xs)], run_time=0.22)
        deep = Text("the deepest", font=FONT, color=GOOD).scale(0.4).next_to(ax2.c2p(2.6, g(2.6)), DOWN, buff=0.35)
        fake = Text("false hole", font=FONT, color=WARN).scale(0.4).next_to(ax2.c2p(-2.8, g(-2.8)), DOWN, buff=0.35)
        self.mark("labels4"); self.play(FadeIn(deep), FadeIn(fake), run_time=0.6)
        self.wait(1.2)
        self.play(FadeOut(VGroup(cur, *hb, deep, fake)), run_time=0.6)

        # ---------- 5. SGD ----------
        cap = self.swap(cap, self.caption("With lots of data: steps in a hurry", "each step looks at just a handful of random data points"))
        cont2 = self.contours(0.55, 0.9, 0.3, c0)
        self.mark("sgd"); self.play(LaggedStart(*[Create(e) for e in cont2], lag_ratio=0.08), run_time=1.0); self.bring_to_front(cap)
        H2 = np.array([[0.55, 0], [0, 0.9]]); R2 = np.array([[np.cos(0.3), -np.sin(0.3)], [np.sin(0.3), np.cos(0.3)]]); H2 = R2 @ H2 @ R2.T
        rng = np.random.default_rng(3)
        p = np.array([-4.8, 1.6]); full = [p.copy()]
        for _ in range(16):
            p = p - 0.45 * (H2 @ p); full.append(p.copy())
        p = np.array([-4.8, 1.6]); noisy = [p.copy()]
        for t in range(60):
            p = p - 0.16 * (H2 @ p + rng.normal(0, 0.9, 2)); noisy.append(p.copy())
        F = VMobject(color=GOOD, stroke_width=4.5).set_points_as_corners([c0 + np.r_[q, 0] for q in full])
        N = VMobject(color=SAMPLE, stroke_width=2.5).set_points_as_corners([c0 + np.r_[q, 0] for q in noisy])
        lf = Text("all the data: smooth but expensive", font=FONT, color=GOOD).scale(0.4).move_to([-3.6, -3.55, 0])
        ln = Text("random batches: wobbly, but cheap", font=FONT, color=SAMPLE).scale(0.4).move_to([3.4, -3.55, 0])
        lf = VGroup(BackgroundRectangle(lf, color=BG, fill_opacity=0.9, buff=0.12), lf)
        ln = VGroup(BackgroundRectangle(ln, color=BG, fill_opacity=0.9, buff=0.12), ln)
        self.add(F, N); self.bring_to_front(cap)
        self.mark("paths5"); self.play(Create(F), Create(N), FadeIn(lf), FadeIn(ln), run_time=3.2, rate_func=linear)
        self.wait(1.4)

        # ---------- Closing ----------
        self.mark("legend")
        self.play(FadeOut(VGroup(cap, cont2, F, N, lf, ln)), run_time=0.7)
        end1 = Text("Step by step, downhill.", font=FONT, weight=BOLD, color=INK).scale(0.9)
        end2 = Text("That's how almost all of machine learning learns.", font=FONT, color=MUTED).scale(0.5)
        VGroup(end1, end2).arrange(DOWN, buff=0.35)
        self.mark("end_text"); self.play(Write(end1), run_time=1.4); self.play(FadeIn(end2, shift=0.2 * UP), run_time=0.8)
        self.wait(2.0)
        self.mark("fade"); self.play(FadeOut(end1), FadeOut(end2), run_time=1)
        self.marks["total"] = [round(self.renderer.time, 3)]
        json.dump(self.marks, open("marks.json", "w"), indent=1)
