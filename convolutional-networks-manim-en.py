import json
import numpy as np
from manim import *

# The series dark palette
BG = "#0E131B"; INK = "#E4EAF3"; MUTED = "#9BA7B9"; LINE = "#2B3545"
MEAN = "#6EA6FF"; SAMPLE = "#AE9FF3"; POINT = "#FF9A55"; GOOD = "#4CC38A"; WARN = "#F08A5D"
FONT = "Inter"
config.background_color = BG
D = json.load(open("../../rn/r8-en/embed.json", encoding="utf-8"))   # the same data as the pages


def T(s, sc=0.45, col=INK, w=None):
    return Text(s, font=FONT, color=col, weight=w or NORMAL).scale(sc)


def fmt(v, d=1):
    s = f"{abs(v):.{d}f}"
    return ("−" if v < 0 and round(abs(v), d) != 0 else "") + s


def mapa(w, x):
    """The filter sum at each position, accumulated as in the toy."""
    out = []
    for i in range(len(x) - len(w) + 1):
        a = 0.0
        for k in range(len(w)):
            a += w[k] * x[i + k]
        out.append(a)
    return out


class Panel:
    """Simple 2D axes: maps data (x, y) to the scene inside a box."""

    def __init__(self, x0, x1, y0, y1, box, xt=(), xf=None, xl=None, yt=(), yf=None, yl=None):
        self.x0, self.x1, self.y0, self.y1 = x0, x1, y0, y1
        self.L, self.R, self.B, self.Tp = box
        g = VGroup(Line([self.L, self.B, 0], [self.R, self.B, 0], stroke_color=MUTED, stroke_width=2))
        for v in xt:
            g.add(T(xf(v) if xf else str(v), 0.28, MUTED).next_to(self.P(v, y0), DOWN, buff=0.1))
        for v in yt:
            g.add(Line(self.P(x0, v), self.P(x1, v), stroke_color=LINE, stroke_width=1.5))
            g.add(T(yf(v) if yf else str(v), 0.28, MUTED).next_to(self.P(x0, v), LEFT, buff=0.1))
        if xl:
            g.add(T(xl, 0.3, INK).next_to([self.R, self.B, 0], DOWN, buff=0.42).align_to([self.R, 0, 0], RIGHT))
        if yl:
            g.add(T(yl, 0.3, MUTED).next_to([self.L, self.Tp, 0], UP, buff=0.12).align_to([self.L, 0, 0], LEFT))
        self.g = g

    def P(self, x, y):
        return np.array([self.L + (x - self.x0) / (self.x1 - self.x0) * (self.R - self.L),
                         self.B + (y - self.y0) / (self.y1 - self.y0) * (self.Tp - self.B), 0])

    def poly(self, pts, col, w=4):
        q = [self.P(min(self.x1, max(self.x0, x)), min(self.y1, max(self.y0, y))) for x, y in pts]
        return VMobject(stroke_color=col, stroke_width=w).set_points_as_corners(q)


def senal_panel(box, yl="acceleration (m/s²)"):
    return Panel(-0.5, 63.5, -6, 3, box, xt=(0, 20, 40, 60), xf=lambda v: fmt(v / 10, 0) + " s",
                 yt=(-6, -3, 0, 3), yf=lambda v: fmt(v, 0), yl=yl)


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
        title = Text("Convolutional networks", font=FONT, weight=BOLD, color=INK).scale(0.95)
        sub = Text("a small filter that sweeps the whole signal", font=FONT, color=MUTED).scale(0.46)
        VGroup(title, sub).arrange(DOWN, buff=0.35)
        self.mark("title"); self.play(Write(title), run_time=1.4)
        self.mark("subtitle"); self.play(FadeIn(sub, shift=0.2 * UP)); self.wait(1.4)
        self.mark("title_out"); self.play(FadeOut(title), FadeOut(sub))

        # ---------- 1. the problem ----------
        cap = self.swap(None, self.caption("The problem: a regular network learns each instant apart",
                                           "trained with the braking always at the start"))
        M = D["mover"]; F = np.array(M["frenadas"]["brusca"]); fondo = np.array(M["fondo"]); z0, z1 = D["principio"]
        Ps = senal_panel((-6.0, 6.0, -0.5, 2.05))
        zona = Polygon(Ps.P(z0 - 0.5, -6), Ps.P(z1 + 0.5, -6), Ps.P(z1 + 0.5, 3), Ps.P(z0 - 0.5, 3), stroke_width=0, fill_color=GOOD, fill_opacity=0.16)
        zl = T("braking started here in training", 0.3, GOOD, BOLD).move_to(zona.get_top() + 0.22 * DOWN).align_to(zona, LEFT).shift(0.1 * RIGHT)
        s = ValueTracker(4)

        def curva():
            k = int(round(s.get_value()))
            x = fondo.copy(); x[k:k + len(F)] += F
            return Ps.poly(list(enumerate(x)), INK, 3.5)
        tr = always_redraw(curva)
        self.play(FadeIn(Ps.g), FadeIn(zona), FadeIn(zl), Create(tr), run_time=1.0)

        def barra(clave, nombre, col, y):
            def f():
                k = int(round(s.get_value())); p = M["p"][f"brusca|{clave}"][k]
                fondo_b = Rectangle(width=6.0, height=0.42, stroke_color=LINE, stroke_width=2).move_to([0.9, y, 0]).align_to([-2.1, 0, 0], LEFT)
                lleno = Rectangle(width=max(0.001, 6.0 * p), height=0.42, stroke_width=0, fill_color=col, fill_opacity=0.9).align_to(fondo_b, LEFT).align_to(fondo_b, DOWN)
                et = T(nombre, 0.34, col, BOLD).next_to(fondo_b, LEFT, buff=0.25)
                pv = T(f"{round(100 * p)}%", 0.36, INK, BOLD).next_to(fondo_b, RIGHT, buff=0.2)
                return VGroup(fondo_b, lleno, et, pv)
            return always_redraw(f)
        b1 = barra("densa_B", "regular (2,113)", MUTED, -2.1); b2 = barra("cnn_B", "convolutional (201)", POINT, -2.95)
        pt = T("probability of “harsh braking”", 0.32, MUTED).move_to([0.9, -1.45, 0])
        self.play(FadeIn(b1), FadeIn(b2), FadeIn(pt), run_time=0.6)
        self.mark("shift"); self.play(s.animate.set_value(22), run_time=2.0, rate_func=linear)
        self.mark("lost"); self.play(s.animate.set_value(44), run_time=2.4, rate_func=linear)
        msg = T("the regular one misses it outside where it learned; the convolutional one does not", 0.34, GOOD, BOLD).move_to([0, -3.65, 0])
        self.play(FadeIn(msg), run_time=0.5); self.wait(1.8)
        self.clear_but(cap)

        # ---------- 2. the sliding filter ----------
        cap = self.swap(cap, self.caption("A 5-weight filter that slides", "it multiplies 5 readings by its weights and adds"))
        x = D["trazas"]["brusca"]; w = D["presets"]["aprendido"]; th = D["umbral"]["aprendido"]; sm = mapa(w, x)
        Pa = senal_panel((-6.0, 6.0, 0.15, 2.45))
        Pm = Panel(-0.5, 63.5, -2, 11, (-6.0, 6.0, -3.35, -0.75), xt=(0, 20, 40, 60), xf=lambda v: fmt(v / 10, 0) + " s", yt=(0, 5, 10), yf=lambda v: fmt(v, 0), yl="filter sum")
        tr = Pa.poly(list(enumerate(x)), INK, 3.5)
        wl = T("weights: " + "  ".join(fmt(v, 2) for v in w), 0.3, POINT, BOLD).move_to([3.4, 2.75, 0])
        self.play(FadeIn(Pa.g), Create(tr), FadeIn(Pm.g), FadeIn(wl), run_time=1.0)
        q = ValueTracker(0)

        def ventana():
            i = int(round(q.get_value()))
            return Rectangle(width=Pa.P(i + 4.5, 0)[0] - Pa.P(i - 0.5, 0)[0], height=Pa.Tp - Pa.B, stroke_color=POINT, stroke_width=4, fill_color=POINT, fill_opacity=0.15).move_to([(Pa.P(i - 0.5, 0)[0] + Pa.P(i + 4.5, 0)[0]) / 2, (Pa.Tp + Pa.B) / 2, 0])

        def barras():
            i = int(round(q.get_value())); g = VGroup(); bw = (Pm.R - Pm.L) / 64 * 0.75
            for j in range(i + 1):
                v = sm[j]; y0, y1 = Pm.P(0, 0)[1], Pm.P(0, v)[1]
                g.add(Rectangle(width=bw, height=max(0.01, abs(y1 - y0)), stroke_width=0, fill_color=POINT if j == i else (WARN if v > th else MUTED),
                                fill_opacity=1 if j == i else 0.7).move_to([Pm.P(j + 2, 0)[0], (y0 + y1) / 2, 0]))
            return g
        vn, bs = always_redraw(ventana), always_redraw(barras)
        self.add(vn, bs)
        self.mark("slide"); self.play(q.animate.set_value(59), run_time=6.0, rate_func=linear)
        ul = DashedLine(Pm.P(-0.5, th), Pm.P(63.5, th), color=WARN, stroke_width=3)
        ut = T(f"threshold {fmt(th, 2)}", 0.3, WARN, BOLD).next_to(Pm.P(-0.5, th), UP, buff=0.06).align_to(Pm.P(-0.5, 0), LEFT)
        am = int(np.argmax(sm))
        mt = T(f"largest: {fmt(sm[am], 2)} at {fmt(am / 10, 1)} s → alert", 0.34, WARN, BOLD).move_to(Pm.P(48, 9.2))
        self.mark("alert"); self.play(Create(ul), FadeIn(ut), FadeIn(mt), run_time=0.8); self.wait(1.8)
        self.clear_but(cap)

        # ---------- 3. several filters and pooling ----------
        cap = self.swap(cap, self.caption("Several filters, and pooling", "each filter looks for a shape; pooling keeps the strongest"))
        F1 = D["f1"]; Pq = senal_panel((-5.0, 6.0, 1.15, 2.45), yl="")
        tr = Pq.poly(list(enumerate(x)), INK, 3)
        self.play(FadeIn(Pq.g), Create(tr), run_time=0.7)
        filas = VGroup(); cw = (Pq.R - Pq.L) / 64
        for f in range(4):
            y = 0.35 - f * 0.68; mp = [max(0.0, v + F1["b"][f]) for v in mapa(F1["w"][f], x)]
            row = VGroup(*[Rectangle(width=cw * 1.02, height=0.5, stroke_width=0, fill_color=POINT, fill_opacity=min(1.0, v / 4)).move_to([Pq.P(i + 2, 0)[0], y, 0]) for i, v in enumerate(mp)])
            caja = Rectangle(width=Pq.P(61.5, 0)[0] - Pq.P(1.5, 0)[0], height=0.5, stroke_color=LINE, stroke_width=1.5).move_to([(Pq.P(61.5, 0)[0] + Pq.P(1.5, 0)[0]) / 2, y, 0])
            gl = VGroup()
            for k, v in enumerate(F1["w"][f]):
                h = abs(v) * 0.42
                gl.add(Rectangle(width=0.13, height=max(0.01, h), stroke_width=0, fill_color=POINT if v >= 0 else MEAN, fill_opacity=1).move_to([-6.7 + 0.18 * k, y + (h / 2 if v >= 0 else -h / 2), 0]))
            filas.add(VGroup(caja, row, gl, T(f"filter {f + 1}", 0.26, MUTED).next_to(caja, LEFT, buff=0.1).shift(0.0 * UP)))
        for r in filas:
            r[3].move_to([-5.5, r[0].get_center()[1], 0])
        self.mark("maps"); self.play(LaggedStart(*[FadeIn(r) for r in filas], lag_ratio=0.25), run_time=1.6); self.wait(0.6)
        n3 = T("filter 3 lights up right over the braking", 0.32, POINT, BOLD).move_to([0.5, -2.55, 0])
        velos = VGroup(*[Rectangle(width=14.5, height=0.62, stroke_width=0, fill_color=BG, fill_opacity=0.72).move_to([0, filas[f][0].get_center()[1], 0]) for f in (0, 1, 3)])
        self.play(FadeIn(n3), FadeIn(velos), run_time=0.7)
        self.wait(1.0)
        # pooling of 2 over the map of filter 3
        mp = [max(0.0, v + F1["b"][2]) for v in mapa(F1["w"][2], x)]
        pool = [max(mp[2 * j], mp[2 * j + 1]) for j in range(30)]
        y = filas[2][0].get_center()[1]
        prow = VGroup(*[Rectangle(width=cw * 2.04, height=0.5, stroke_width=0, fill_color=POINT, fill_opacity=min(1.0, v / 4)).move_to([(Pq.P(2 * j + 2, 0)[0] + Pq.P(2 * j + 3, 0)[0]) / 2, y - 1.95, 0]) for j, v in enumerate(pool)])
        pl = T("pooling of 2: 60 → 30, the maximum of each pair", 0.32, INK, BOLD).move_to([0.5, -3.35, 0])
        self.mark("pool"); self.play(FadeOut(n3), TransformFromCopy(filas[2][1], prow), FadeIn(pl), run_time=1.4); self.wait(1.6)
        self.clear_but(cap)

        # ---------- 4. stacked layers ----------
        cap = self.swap(cap, self.caption("Stacked layers: each one sees a longer stretch", "filters of 5 and pooling of 2, one layer on top of another"))
        Pr = senal_panel((-6.0, 6.0, -3.2, -0.9), yl="")
        xr = fondo.copy(); xr[24:24 + len(F)] += F
        tr = Pr.poly(list(enumerate(xr)), INK, 3)
        self.play(FadeIn(Pr.g), Create(tr), run_time=0.7)
        prev = None; ytop = -0.2
        for n, (r, a) in enumerate([(5, 29), (14, 25), (32, 16)]):
            band = Rectangle(width=Pr.P(a + r - 0.5, 0)[0] - Pr.P(a - 0.5, 0)[0], height=Pr.Tp - Pr.B, stroke_color=POINT, stroke_width=3, fill_color=POINT, fill_opacity=0.18).move_to([(Pr.P(a - 0.5, 0)[0] + Pr.P(a + r - 0.5, 0)[0]) / 2, (Pr.Tp + Pr.B) / 2, 0])
            capas = VGroup(*[RoundedRectangle(width=1.7, height=0.42, corner_radius=0.1, stroke_color=POINT, stroke_width=2, fill_color=POINT, fill_opacity=0.25 + 0.2 * k).move_to([-4.8 + 0 * k, ytop + 0.55 * k, 0]) for k in range(n + 1)])
            for k, c in enumerate(capas):
                c.add(T(f"layer {k + 1}", 0.26, INK, BOLD).move_to(c))
            txt = T(f"{n + 1} layer{'s' if n else ''}: sees {r} readings, {fmt(r / 10, 1)} s", 0.4, POINT, BOLD).move_to([1.4, 0.6, 0])
            self.mark("rf")
            if prev is None:
                self.play(FadeIn(band), FadeIn(capas), FadeIn(txt), run_time=0.8)
            else:
                self.play(Transform(prev[0], band), Transform(prev[1], capas), Transform(prev[2], txt), run_time=0.8)
            if prev is None:
                prev = [band, capas, txt]
            self.wait(1.3)
        nt = T("the first ones see edges; the upper ones, the whole braking", 0.34, GOOD, BOLD).move_to([1.4, -0.2, 0])
        self.play(FadeIn(nt), run_time=0.5); self.wait(1.4)
        self.clear_but(cap)

        # ---------- 5. in an image ----------
        cap = self.swap(cap, self.caption("In an image: a 3 × 3 filter", "vertical edges on the lane line"))
        img = np.array([[1.0 if abs(c - (3 + 0.45 * r)) < 1.3 else 0.12 for c in range(16)] for r in range(16)])
        K = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        Y = np.array([[np.sum(K * img[r:r + 3, c:c + 3]) for c in range(14)] for r in range(14)]); ym = np.abs(Y).max()
        cs = 0.31; o1 = np.array([-5.6, 1.95, 0]); o2 = np.array([0.9, 1.95 - cs, 0])
        grid = VGroup(*[Square(cs, stroke_width=0.5, stroke_color=LINE, fill_color=interpolate_color(ManimColor(BG), ManimColor(INK), img[r, c]), fill_opacity=1).move_to(o1 + np.array([(c + 0.5) * cs, -(r + 0.5) * cs, 0])) for r in range(16) for c in range(16)])
        marco = Square(14 * cs, stroke_color=LINE, stroke_width=2).move_to(o2 + np.array([7 * cs, -7 * cs, 0]))
        l1 = T("16 × 16 image", 0.3, MUTED).next_to(grid, UP, buff=0.1).align_to(grid, LEFT)
        l2 = T("14 × 14 map", 0.3, MUTED).next_to(marco, UP, buff=0.1).align_to(marco, LEFT)
        self.play(FadeIn(grid), Create(marco), FadeIn(l1), FadeIn(l2), run_time=0.8)
        kk = ValueTracker(0)

        def ventana2():
            k = int(kk.get_value()); r, c = divmod(min(k, 195), 14)
            return Square(3 * cs, stroke_color=POINT, stroke_width=5).move_to(o1 + np.array([(c + 1.5) * cs, -(r + 1.5) * cs, 0]))

        def mapa2():
            k = int(kk.get_value()); g = VGroup()
            for j in range(min(k, 195) + 1):
                r, c = divmod(j, 14); v = Y[r, c] / ym
                g.add(Square(cs, stroke_width=0, fill_color=POINT if v >= 0 else MEAN, fill_opacity=abs(v)).move_to(o2 + np.array([(c + 0.5) * cs, -(r + 0.5) * cs, 0])))
            return g
        v2, m2 = always_redraw(ventana2), always_redraw(mapa2)
        self.add(m2, v2)
        self.mark("img"); self.play(kk.animate.set_value(195), run_time=4.5, rate_func=linear)
        n5 = T("orange: dark to light · blue: light to dark · 9 weights for the whole image", 0.3, INK).move_to([0, -3.55, 0])
        self.play(FadeIn(n5), FadeOut(v2), run_time=0.5); self.wait(1.6)
        self.clear_but(cap)

        # ---------- 6. the result ----------
        cap = self.swap(cap, self.caption("The result on 1,000 test windows", "harsh braking in the accelerometer"))
        grupos = [("numbers to learn", 201, 2113, lambda v: f"{v:,}", 2113),
                  ("accuracy, braking anywhere", 96.3, 88.8, lambda v: fmt(v, 1) + "%", 100),
                  ("accuracy, braking at the end (seen only at the start)", 96.7, 46.1, lambda v: fmt(v, 1) + "%", 100)]
        objs = VGroup()
        for gi, (nm, a, b, ff, top) in enumerate(grupos):
            y = 1.7 - gi * 1.75
            h = T(nm, 0.34, MUTED).move_to([-5.9, y + 0.42, 0], aligned_edge=LEFT)
            ba = Rectangle(width=7.5 * a / top, height=0.36, stroke_width=0, fill_color=POINT, fill_opacity=0.9).move_to([-3.4, y, 0], aligned_edge=LEFT)
            bb = Rectangle(width=7.5 * b / top, height=0.36, stroke_width=0, fill_color=MUTED, fill_opacity=0.9).move_to([-3.4, y - 0.46, 0], aligned_edge=LEFT)
            la = T("convolutional", 0.28, POINT, BOLD).next_to(ba, LEFT, buff=0.15).align_to([-3.55, 0, 0], RIGHT)
            lb = T("dense", 0.28, MUTED, BOLD).next_to(bb, LEFT, buff=0.15).align_to([-3.55, 0, 0], RIGHT)
            va = T(ff(a), 0.32, INK, BOLD).next_to(ba, RIGHT, buff=0.15); vb = T(ff(b), 0.32, INK, BOLD).next_to(bb, RIGHT, buff=0.15)
            objs.add(VGroup(h, ba, bb, la, lb, va, vb))
        for k, g in enumerate(objs):
            self.mark(["params", "acc", "shiftacc"][k])
            self.play(FadeIn(g[0]), GrowFromEdge(g[1], LEFT), GrowFromEdge(g[2], LEFT), FadeIn(g[3:]), run_time=0.9); self.wait(1.2)
        self.wait(1.3)
        self.clear_but(cap)

        # ---------- Closing ----------
        self.mark("legend")
        self.play(FadeOut(cap), run_time=0.6)
        e1 = Text("Convolutional networks", font=FONT, weight=BOLD, color=INK).scale(0.62)
        e2 = VGroup(T("a small filter, the same weights everywhere", 0.42, MUTED),
                    T("it recognizes the shape wherever it is", 0.42, MUTED),
                    T("pooling tolerates shifts; stacked layers see farther", 0.42, MUTED),
                    T("10 times fewer numbers, and it gets more right", 0.42, MUTED)).arrange(DOWN, buff=0.18)
        VGroup(e1, e2).arrange(DOWN, buff=0.45)
        self.mark("end_text"); self.play(Write(e1), run_time=1.0); self.play(FadeIn(e2, shift=0.2 * UP, lag_ratio=0.15), run_time=1.0)
        self.wait(3.3)
        self.mark("fade"); self.play(FadeOut(VGroup(e1, e2)), run_time=1)
        self.marks["total"] = [round(self.renderer.time, 3)]
        json.dump(self.marks, open("marks.json", "w"), indent=1)
