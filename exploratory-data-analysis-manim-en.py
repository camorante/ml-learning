import json
import numpy as np
from manim import *
from scipy.stats import beta as Beta

# Dark palette of the visual ML series
BG = "#0E131B"; INK = "#E4EAF3"; MUTED = "#9BA7B9"; LINE = "#2B3545"
MEAN = "#6EA6FF"; SAMPLE = "#AE9FF3"; POINT = "#FF9A55"; GOOD = "#4CC38A"; WARN = "#F08A5D"
FONT = "Inter"
config.background_color = BG


class Exploratorio(Scene):
    def caption(self, text, sub=None):
        t = Text(text, font=FONT, weight=SEMIBOLD, color=INK).scale(0.62)
        if t.width > 13.2: t.scale_to_fit_width(13.2)
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
        title = Text("Exploratory data analysis", font=FONT, weight=BOLD, color=INK).scale(1.05)
        sub = Text("looking at the data before using it", font=FONT, color=MUTED).scale(0.5)
        VGroup(title, sub).arrange(DOWN, buff=0.35)
        self.mark("title"); self.play(Write(title), run_time=1.2)
        self.mark("subtitle"); self.play(FadeIn(sub, shift=0.2 * UP)); self.wait(1.2)
        self.mark("title_out"); self.play(FadeOut(title), FadeOut(sub))
        rng = np.random.default_rng(4)

        # ---------- 1. histogram ----------
        cap = self.swap(None, self.caption("A histogram exposes the errors", "engine temperature over 2,400 trips"))
        X = lambda v: -6.0 + (v + 50) / 160 * 12.0; Y0 = -2.6
        temps = np.r_[rng.normal(88, 4, 2340), np.full(60, -40.0)]
        edges = np.arange(-50, 111, 2.5); cnt, _ = np.histogram(temps, edges); sc = 4.0 / cnt.max()
        bars = VGroup(*[Rectangle(width=12 * 2.5 / 160 * 0.85, height=max(0.001, c * sc), stroke_width=0, fill_color=WARN if e < -10 else SAMPLE, fill_opacity=1 if e < -10 else 0.75).move_to([X(e + 1.25), Y0 + c * sc / 2, 0]) for e, c in zip(edges[:-1], cnt) if c > 0])
        axis = Line([-6.2, Y0, 0], [6.2, Y0, 0], color=LINE, stroke_width=2)
        ticks = VGroup(*[Text(f"{t} °C".replace("-", "−"), font=FONT, color=MUTED).scale(0.32).move_to([X(t), Y0 - 0.3, 0]) for t in (-40, 0, 40, 80)])
        self.mark("hist"); self.play(Create(axis), FadeIn(ticks), *[GrowFromEdge(b, DOWN) for b in bars], run_time=1.4); self.wait(0.6)
        lab = Text("60 trips at −40 °C: the sensor had no reading", font=FONT, weight=SEMIBOLD, color=WARN).scale(0.42).move_to([X(-40) + 3.2, Y0 + 1.6, 0])
        arr = Arrow(lab.get_left() + 0.1 * LEFT, [X(-40) + 0.1, Y0 + 0.7, 0], color=WARN, buff=0.1, stroke_width=4)
        self.mark("sentinel"); self.play(FadeIn(lab), GrowArrow(arr), run_time=0.8); self.wait(2.2)
        self.play(FadeOut(VGroup(bars, axis, ticks, lab, arr)), run_time=0.6)

        # ---------- 2. mean and median ----------
        cap = self.swap(cap, self.caption("The mean gets dragged along; the median doesn't", "trip distance"))
        Xd = lambda v: -6.0 + v / 950 * 12.0; Yd = -1.0
        d = np.exp(rng.normal(np.log(45), 0.6, 120)); d = d[d < 400]
        dots = VGroup(*[Dot([Xd(v), Yd + rng.uniform(-0.6, 0.6), 0], radius=0.05, color=SAMPLE).set_opacity(0.7) for v in d])
        ax2 = Line([-6.2, Yd - 0.9, 0], [6.2, Yd - 0.9, 0], color=LINE, stroke_width=2)
        tk2 = VGroup(*[Text(f"{t} km", font=FONT, color=MUTED).scale(0.32).move_to([Xd(t), Yd - 1.2, 0]) for t in (0, 200, 400, 600, 800)])
        def marker(v, col, txt_, up):
            y = Yd + (1.5 if up else 1.0)
            return VGroup(Line([Xd(v), Yd - 0.85, 0], [Xd(v), y - 0.15, 0], color=col, stroke_width=4), Text(txt_, font=FONT, weight=BOLD, color=col).scale(0.38).move_to([Xd(v) + 0.9, y, 0]))
        md, me = float(np.median(d)), float(np.mean(d))
        mmed = marker(md, GOOD, f"median {md:.0f}", False); mmean = marker(me, MEAN, f"mean {me:.0f}", True)
        self.mark("dots"); self.play(Create(ax2), FadeIn(tk2), FadeIn(dots, lag_ratio=0.01), run_time=1.2)
        self.play(FadeIn(mmed), FadeIn(mmean), run_time=0.7); self.wait(0.6)
        d2 = np.r_[d, np.full(6, 900.0)]
        far = VGroup(*[Dot([Xd(900), Yd - 0.5 + 0.2 * i, 0], radius=0.08, color=POINT) for i in range(6)])
        me2, md2 = float(np.mean(d2)), float(np.median(d2))
        self.mark("far"); self.play(FadeIn(far, shift=0.3 * LEFT), run_time=0.6)
        self.play(Transform(mmean, marker(me2, MEAN, f"mean {me2:.0f}", True)), Transform(mmed, marker(md2, GOOD, f"median {md2:.0f}", False)), run_time=1.0); self.wait(2.0)
        self.play(FadeOut(VGroup(dots, ax2, tk2, mmed, mmean, far)), run_time=0.6)

        # ---------- 3. correlation ----------
        cap = self.swap(cap, self.caption("One bad data point flips a correlation", "speed and fuel consumption of six trips"))
        Xv = lambda v: -5.5 + (v - 40) / 230 * 11.0; Yc = lambda c: -2.6 + (c - 18) / 13 * 4.0
        V = [50, 60, 70, 80, 90]; Cc = [20, 22, 23, 26, 29]
        pts = VGroup(*[Dot([Xv(v), Yc(c), 0], radius=0.12, color=SAMPLE) for v, c in zip(V, Cc)])
        F = Dot([Xv(95), Yc(21), 0], radius=0.13, color=POINT)
        def fitline(xs, ys, col):
            s, b = np.polyfit(xs, ys, 1)
            return Line([Xv(40), Yc(s * 40 + b), 0], [Xv(270), Yc(s * 270 + b), 0], color=col, stroke_width=4)
        ln = fitline(V, Cc, MEAN)
        rt = Text("r = 0.98", font=FONT, weight=BOLD, color=MEAN).scale(0.55).move_to([3.5, 1.4, 0])
        self.mark("pts"); self.play(FadeIn(pts, lag_ratio=0.15), run_time=0.8); self.play(Create(ln), FadeIn(rt), run_time=0.8); self.wait(0.8)
        self.play(FadeIn(F), run_time=0.4)
        F2 = Dot([Xv(255), Yc(21), 0], radius=0.13, color=POINT)
        e255 = Text("255 km/h: error code", font=FONT, color=POINT).scale(0.38).move_to([Xv(255) - 0.6, Yc(21) - 0.5, 0])
        rt2 = Text("r = −0.19", font=FONT, weight=BOLD, color=WARN).scale(0.55).move_to([3.5, 1.4, 0])
        self.mark("flip"); self.play(Transform(F, F2), Transform(ln, fitline(V + [255], Cc + [21], WARN)), Transform(rt, rt2), FadeIn(e255), run_time=1.4); self.wait(2.2)
        self.play(FadeOut(VGroup(pts, F, ln, rt, e255)), run_time=0.6)

        # ---------- 4. redundant columns ----------
        cap = self.swap(cap, self.caption("Columns that say the same thing", "distance from the odometer and from the GPS"))
        vals = [32, 58, 41, 120, 27, 76]
        def col(xc, name, noise, colr):
            g = VGroup(Text(name, font=FONT, weight=SEMIBOLD, color=INK).scale(0.4).move_to([xc, 1.6, 0]))
            for i, v in enumerate(vals):
                g.add(Rectangle(width=v / 120 * 3.2 * noise[i], height=0.32, stroke_width=0, fill_color=colr, fill_opacity=0.85).move_to([xc - 1.6 + v / 120 * 3.2 * noise[i] / 2, 1.0 - i * 0.5, 0]))
            return g
        c1 = col(-3.0, "distance (odometer)", [1] * 6, MEAN); c2 = col(2.6, "GPS km", [1.01, 0.99, 1.0, 1.005, 0.995, 1.0], SAMPLE)
        self.mark("cols"); self.play(FadeIn(c1, lag_ratio=0.1), run_time=0.8); self.play(FadeIn(c2, lag_ratio=0.1), run_time=0.8)
        rr = Text("correlation 1.00: one of the two is extra", font=FONT, weight=BOLD, color=WARN).scale(0.46).move_to([0, -2.4, 0])
        self.mark("redund"); self.play(FadeIn(rr, shift=0.2 * UP), c2.animate.set_opacity(0.25), run_time=0.8); self.wait(2.0)
        self.play(FadeOut(VGroup(c1, c2, rr)), run_time=0.6)

        # ---------- 5. Simpson ----------
        cap = self.swap(cap, self.caption("Looking at everything together can tell the opposite story", "speed and fuel consumption of three truck types"))
        Xs = lambda v: -5.5 + (v - 30) / 75 * 11.0; Ys = lambda c: -2.9 + (c - 15) / 42 * 4.6
        groups = [(74, 23.6, 0.26, SAMPLE, "light"), (62, 30.8, 0.19, MEAN, "medium"), (52, 42.5, 0.13, POINT, "heavy")]
        allpts = VGroup(); allx = []; ally = []; glines = VGroup()
        for vm, cm, sl, colr, nm in groups:
            v = rng.normal(vm, 8, 70); c = cm + sl * (v - vm) + rng.normal(0, 1.6, 70)
            allx += list(v); ally += list(c)
            allpts.add(*[Dot([Xs(a), Ys(b), 0], radius=0.045, color=colr).set_opacity(0.7) for a, b in zip(v, c)])
            glines.add(VGroup(Line([Xs(vm - 16), Ys(cm - sl * 16), 0], [Xs(vm + 16), Ys(cm + sl * 16), 0], color=colr, stroke_width=5),
                              Text(nm, font=FONT, weight=BOLD, color=colr).scale(0.36).move_to([Xs(vm + 16) + 0.75, Ys(cm + sl * 16), 0])))
        grey = allpts.copy().set_color(MUTED)
        s, b = np.polyfit(allx, ally, 1)
        gl = Line([Xs(32), Ys(s * 32 + b), 0], [Xs(100), Ys(s * 100 + b), 0], color=INK, stroke_width=5)
        t1 = Text("all together: faster, lower consumption", font=FONT, color=INK).scale(0.4).move_to([0, -3.35, 0])
        self.mark("simp"); self.play(FadeIn(grey, lag_ratio=0.003), run_time=1.0); self.play(Create(gl), FadeIn(t1), run_time=0.8); self.wait(1.4)
        t2 = Text("by type: faster, higher consumption", font=FONT, weight=BOLD, color=GOOD).scale(0.42).move_to([0, -3.35, 0])
        self.mark("split"); self.play(ReplacementTransform(grey, allpts), FadeOut(gl), run_time=1.0)
        self.play(*[Create(g[0]) for g in glines], *[FadeIn(g[1]) for g in glines], ReplacementTransform(t1, t2), run_time=1.0); self.wait(2.2)
        self.play(*[FadeOut(m) for m in self.mobjects if m is not cap], run_time=0.6)

        # ---------- Closing ----------
        self.mark("legend")
        self.play(FadeOut(cap), run_time=0.6)
        e1 = Text("Look before you use.", font=FONT, weight=BOLD, color=INK).scale(0.8)
        e2 = Text("Every column, every pair and every group.", font=FONT, color=MUTED).scale(0.48)
        VGroup(e1, e2).arrange(DOWN, buff=0.4)
        self.mark("end_text"); self.play(Write(e1), run_time=1.2); self.play(FadeIn(e2, shift=0.2 * UP), run_time=0.8)
        self.wait(2.0)
        self.mark("fade"); self.play(FadeOut(VGroup(e1, e2)), run_time=1)
        self.marks["total"] = [round(self.renderer.time, 3)]
        json.dump(self.marks, open("marks.json", "w"), indent=1)
