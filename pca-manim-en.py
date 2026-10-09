import json
import numpy as np
from manim import *

# Dark palette of the Visual ML series
BG = "#0E131B"; INK = "#E4EAF3"; MUTED = "#9BA7B9"; LINE = "#2B3545"
MEAN = "#6EA6FF"; SAMPLE = "#AE9FF3"; POINT = "#FF9A55"; GOOD = "#4CC38A"; WARN = "#F08A5D"
FONT = "Inter"
config.background_color = BG


class Componentes(Scene):
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

    def construct(self):
        self.marks = {}
        title = Text("PCA", font=FONT, weight=BOLD, color=INK).scale(1.4)
        sub = Text("look at the data from the angle that tells the most", font=FONT, color=MUTED).scale(0.5)
        VGroup(title, sub).arrange(DOWN, buff=0.35)
        self.mark("title"); self.play(Write(title), run_time=1.4)
        self.mark("subtitle"); self.play(FadeIn(sub, shift=0.2 * UP)); self.wait(1.2)
        self.mark("title_out"); self.play(FadeOut(title), FadeOut(sub))

        # ---------- 1. The best shadow ----------
        cap = self.swap(None, self.caption("Find the best shadow", "rotate the line until the shadows are as spread out as possible"))
        rng = np.random.default_rng(6)
        th = np.deg2rad(32)
        U = np.c_[1.25 * rng.standard_normal(60), 0.38 * rng.standard_normal(60)]
        P = U @ np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]])
        P -= P.mean(0)
        O = np.array([-1.6, -0.8, 0])
        pts = VGroup(*[Dot(O + [p[0], p[1], 0], radius=0.065, color=POINT) for p in P])
        self.mark("points"); self.play(LaggedStart(*[GrowFromCenter(d) for d in pts], lag_ratio=0.02), run_time=1.2)
        ang = ValueTracker(110.0)
        def unit():
            a = np.deg2rad(ang.get_value()); return np.array([np.cos(a), np.sin(a)])
        line = always_redraw(lambda: Line(O - 3.6 * np.r_[unit(), 0], O + 3.6 * np.r_[unit(), 0], color=MEAN, stroke_width=4))
        def proj():
            u = unit(); t = P @ u; Q = np.outer(t, u)
            segs = VGroup(*[Line(O + [p[0], p[1], 0], O + [q[0], q[1], 0], color=SAMPLE, stroke_width=1.6, stroke_opacity=0.7) for p, q in zip(P, Q)])
            sh = VGroup(*[Dot(O + [q[0], q[1], 0], radius=0.045, color=MEAN) for q in Q])
            return VGroup(segs, sh)
        pr = always_redraw(proj)
        tot = (P ** 2).sum()
        def bars():
            u = unit(); a = ((P @ u) ** 2).sum() / tot
            g = VGroup()
            for i, (lab, v, col) in enumerate([("lengthwise", a, MEAN), ("crosswise", 1 - a, SAMPLE)]):
                y = 0.3 - i * 0.8
                g.add(Text(lab, font=FONT, color=INK).scale(0.36).move_to([3.15, y + 0.28, 0], aligned_edge=LEFT))
                g.add(Rectangle(width=2.6, height=0.2, stroke_width=0, fill_color=LINE, fill_opacity=1).move_to([3.15, y, 0], aligned_edge=LEFT))
                g.add(Rectangle(width=max(0.01, 2.6 * v), height=0.2, stroke_width=0, fill_color=col, fill_opacity=1).move_to([3.15, y, 0], aligned_edge=LEFT))
                g.add(Text(f"{v:.0%}", font=FONT, color=INK).scale(0.36).move_to([6.05, y, 0], aligned_edge=LEFT))
            return g
        br = always_redraw(bars)
        self.mark("line"); self.play(Create(line), FadeIn(pr), FadeIn(br), run_time=0.9)
        self.mark("sweep"); self.play(ang.animate.set_value(190.0), run_time=3.0, rate_func=smooth)
        self.mark("best"); self.play(ang.animate.set_value(180 + 32.0), run_time=1.6, rate_func=smooth)
        a = ((P @ unit()) ** 2).sum() / tot
        lab = Text(f"component 1: keeps {a:.0%}", font=FONT, weight=SEMIBOLD, color=GOOD).scale(0.42).move_to([4.6, -1.3, 0])
        self.play(FadeIn(lab), run_time=0.6); self.wait(1.2)
        for m in (line, pr, br): m.clear_updaters()

        # ---------- 2. New axes ----------
        cap = self.swap(cap, self.caption("New axes", "component 2 is perpendicular; we rotate the data to look at it head-on"))
        u = unit(); v = np.array([-u[1], u[0]])
        pc2 = Line(O - 1.6 * np.r_[v, 0], O + 1.6 * np.r_[v, 0], color=SAMPLE, stroke_width=4)
        l1 = Text("1", font=FONT, weight=BOLD, color=MEAN).scale(0.42).move_to(O + 3.85 * np.r_[-u, 0])
        l2 = Text("2", font=FONT, weight=BOLD, color=SAMPLE).scale(0.42).move_to(O + 1.85 * np.r_[v, 0])
        self.mark("pc2"); self.play(FadeOut(pr), FadeOut(br), FadeOut(lab), Create(pc2), FadeIn(l1), FadeIn(l2), run_time=0.9)
        R = np.array([[-u[0], -u[1]], [v[0], v[1]]])     # component 1 becomes the horizontal axis
        Pn = P @ R.T
        newl = Line(O + [-3.6, 0, 0], O + [3.6, 0, 0], color=MEAN, stroke_width=4)
        new2 = Line(O + [0, -1.6, 0], O + [0, 1.6, 0], color=SAMPLE, stroke_width=4)
        self.mark("rotate")
        self.play(*[d.animate.move_to(O + [q[0], q[1], 0]) for d, q in zip(pts, Pn)], Transform(line, newl), Transform(pc2, new2),
                  l1.animate.move_to(O + [3.85, 0, 0]), l2.animate.move_to(O + [0, 1.85, 0]), run_time=1.8)
        note = Text("almost everything happens along axis 1:\none number per point is enough", font=FONT, color=INK, line_spacing=0.9).scale(0.34).move_to([4.45, -1.5, 0])
        self.play(FadeIn(note), run_time=0.6); self.wait(1.6)
        self.play(FadeOut(VGroup(pts, line, pc2, l1, l2, note)), run_time=0.6)

        # ---------- 3. Explained variance ----------
        cap = self.swap(cap, self.caption("Keep what matters", "6 measurements per truck, but almost everything fits in 2 components"))
        vals = [0.67, 0.25, 0.05, 0.02, 0.008, 0.002]
        ax = Axes(x_range=[0.4, 6.6, 1], y_range=[0, 1, 0.25], x_length=8, y_length=4.2, tips=False,
                  axis_config={"color": LINE, "stroke_width": 2, "include_ticks": False}).move_to([0, -1.0, 0])
        bl = VGroup(*[Rectangle(width=0.8, height=max(0.01, 4.2 * v), stroke_width=0, fill_color=MEAN if i < 2 else LINE, fill_opacity=1)
                      .move_to(ax.c2p(i + 1, 0), aligned_edge=DOWN) for i, v in enumerate(vals)])
        nums = VGroup(*[Text(f"{v:.0%}" if v >= 0.01 else "<1%", font=FONT, color=INK).scale(0.32).next_to(b, UP, buff=0.08) for b, v in zip(bl, vals)])
        xs = VGroup(*[Text(str(i + 1), font=FONT, color=MUTED).scale(0.32).next_to(ax.c2p(i + 1, 0), DOWN, buff=0.12) for i in range(6)])
        self.mark("scree"); self.play(Create(ax), FadeIn(xs), LaggedStart(*[GrowFromEdge(b, DOWN) for b in bl], lag_ratio=0.15), run_time=1.4)
        self.play(FadeIn(nums), run_time=0.5)
        cum = Text("components 1 and 2: 92% of the information", font=FONT, weight=SEMIBOLD, color=GOOD).scale(0.46).move_to([1.2, 1.2, 0])
        self.mark("cum"); self.play(FadeIn(cum), Indicate(bl[0], color=GOOD), Indicate(bl[1], color=GOOD), run_time=1.0); self.wait(1.4)
        self.play(FadeOut(VGroup(ax, bl, nums, xs, cum)), run_time=0.6)

        # ---------- 4. Compress and reconstruct ----------
        cap = self.swap(cap, self.caption("Compress and reconstruct", "a day of speeds: 24 numbers → just a few"))
        r2 = np.random.default_rng(8); H = np.arange(24)
        shapes = [lambda h: np.where((h >= 7) & (h <= 18), 38 + 8 * np.sin((h - 7) / 11 * np.pi), 0),
                  lambda h: np.where((h >= 21) | (h <= 5), 78, np.where((h >= 6) & (h <= 8), 40, 0)),
                  lambda h: np.where(((h >= 5) & (h <= 10)) | ((h >= 15) & (h <= 20)), 52, 0)]
        T = []
        for _ in range(150):
            w = np.maximum(0, r2.random(3) - 0.25); w = w / (w.sum() or 1); sh = int(round(r2.standard_normal() * 1.2))
            hh = (H + sh) % 24
            T.append(np.maximum(0, sum(w[k] * shapes[k](hh) for k in range(3)) + 4 * r2.standard_normal(24)))
        T = np.array(T); mu = T.mean(0); Zc = T - mu
        _, _, Vt = np.linalg.svd(Zc, full_matrices=False)
        idx = int(np.argmax((Zc @ Vt[:2].T) ** 2 @ np.ones(2))); z = Zc[idx]
        ax4 = Axes(x_range=[0, 23, 6], y_range=[-5, 95, 25], x_length=9.5, y_length=4.2, tips=False,
                   axis_config={"color": LINE, "stroke_width": 2, "include_ticks": False}).move_to([0, -1.0, 0])
        hl = VGroup(*[Text(f"{h} h", font=FONT, color=MUTED).scale(0.3).next_to(ax4.c2p(h, -5), DOWN, buff=0.1) for h in (0, 6, 12, 18, 23)])
        real = VMobject(color=POINT, stroke_width=4).set_points_as_corners([ax4.c2p(h, T[idx][h]) for h in H])
        rdots = VGroup(*[Dot(ax4.c2p(h, T[idx][h]), radius=0.05, color=POINT) for h in H])
        self.mark("profile"); self.play(Create(ax4), FadeIn(hl), Create(real), FadeIn(rdots), run_time=1.2)
        def rec(k):
            rcn = mu + (Vt[:k].T @ (Vt[:k] @ z) if k else 0)
            return VMobject(color=MEAN, stroke_width=5).set_points_as_corners([ax4.c2p(h, rcn[h]) for h in H])
        def rlab(k):
            rcn = mu + (Vt[:k].T @ (Vt[:k] @ z) if k else 0); e = np.sqrt(((rcn - T[idx]) ** 2).mean())
            return Text(f"{k} number{'s' if k != 1 else ''} · error {e:.1f} km/h", font=FONT, weight=SEMIBOLD, color=MEAN).scale(0.42).move_to([0, 1.25, 0])
        cur = rec(0); cl = rlab(0)
        self.mark("rec"); self.play(Create(cur), FadeIn(cl), run_time=0.8); self.wait(0.6)
        for k in (1, 2, 3):
            self.mark("rec"); self.play(Transform(cur, rec(k)), Transform(cl, rlab(k)), run_time=0.9); self.wait(0.7)
        self.wait(0.6)
        self.play(FadeOut(VGroup(ax4, hl, real, rdots, cur, cl)), run_time=0.6)

        # ---------- 5. Only straight lines ----------
        cap = self.swap(cap, self.caption("It only knows straight lines", "on a curve, it lumps together points that are far apart"))
        r3 = np.random.default_rng(23)
        tt = np.linspace(-0.15 * np.pi, 1.15 * np.pi, 70)
        A = np.c_[2.3 * np.cos(tt), 2.3 * np.sin(tt)] + 0.11 * r3.standard_normal((70, 2))
        m = A.mean(0); _, _, V = np.linalg.svd(A - m); w = V[0]
        O5 = np.array([0, -0.9, 0])
        ad = VGroup(*[Dot(O5 + [p[0] - m[0], p[1] - m[1], 0], radius=0.07, color=POINT) for p in A])
        ln = Line(O5 - 4.5 * np.r_[w, 0], O5 + 4.5 * np.r_[w, 0], color=MEAN, stroke_width=4)
        self.mark("arc"); self.play(LaggedStart(*[GrowFromCenter(d) for d in ad], lag_ratio=0.01), run_time=1.0); self.play(Create(ln), run_time=0.6)
        t = (A - m) @ w
        best = None
        for i in range(70):
            for j in range(i + 1, 70):
                if abs(t[i] - t[j]) < 0.06 and (best is None or j - i > best[0]): best = (j - i, i, j)
        _, i, j = best
        rings = VGroup(*[Circle(radius=0.2, color=INK, stroke_width=3).move_to(ad[k].get_center()) for k in (i, j)])
        drops = VGroup(*[DashedLine(ad[k].get_center(), O5 + t[k] * np.r_[w, 0], color=SAMPLE, stroke_width=3) for k in (i, j)])
        self.mark("collapse"); self.play(Create(rings), run_time=0.6); self.play(Create(drops), run_time=0.8)
        note = Text("far apart on the curve, together in the shadow", font=FONT, weight=SEMIBOLD, color=WARN).scale(0.44).move_to([0, -3.75, 0])
        self.play(FadeIn(note), run_time=0.6); self.wait(1.6)
        self.play(FadeOut(VGroup(ad, ln, rings, drops, note)), run_time=0.6)

        # ---------- Closing ----------
        self.mark("legend")
        self.play(FadeOut(cap), run_time=0.6)
        e1 = Text("Rotate, project and keep what matters.", font=FONT, weight=BOLD, color=INK).scale(0.7)
        e2 = Text("A few numbers that tell almost the whole story.", font=FONT, color=MUTED).scale(0.48)
        VGroup(e1, e2).arrange(DOWN, buff=0.4)
        self.mark("end_text"); self.play(Write(e1), run_time=1.4); self.play(FadeIn(e2, shift=0.2 * UP), run_time=0.8)
        self.wait(2.0)
        self.mark("fade"); self.play(FadeOut(VGroup(e1, e2)), run_time=1)
        self.marks["total"] = [round(self.renderer.time, 3)]
        json.dump(self.marks, open("marks.json", "w"), indent=1)
