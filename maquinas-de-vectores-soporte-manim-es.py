import json
import numpy as np
from manim import *
from sklearn.svm import SVC

# Paleta oscura de la serie ML visual
BG = "#0E131B"; INK = "#E4EAF3"; MUTED = "#9BA7B9"; LINE = "#2B3545"
MEAN = "#6EA6FF"; SAMPLE = "#AE9FF3"; POINT = "#FF9A55"; GOOD = "#4CC38A"; WARN = "#F08A5D"
FONT = "Inter"
config.background_color = BG


def col(c):
    return POINT if c > 0 else SAMPLE


def gap_at(P, y, n):
    t = P @ n
    mn, mx = t[y > 0].min(), t[y < 0].max()
    return mn - mx, (mn + mx) / 2, (mn - mx) / 2


def street(O, n, m, h, edge_op=1.0, fill_op=0.18, mid_w=5, L=9):
    """Calle: centro en n·x = m, medio ancho h (coordenadas relativas a O)."""
    n = np.asarray(n, float); d = np.array([-n[1], n[0]])
    P = lambda t, s: O + np.r_[n * t + d * s, 0]
    g = VGroup()
    if h > 0:
        g.add(Polygon(P(m - h, -L), P(m - h, L), P(m + h, L), P(m + h, -L), stroke_width=0, fill_color=MEAN, fill_opacity=fill_op))
        for t in (m - h, m + h):
            g.add(DashedLine(P(t, -L), P(t, L), color=MEAN, stroke_width=2.4, stroke_opacity=edge_op, dash_length=0.12))
    g.add(Line(P(m, -L), P(m, L), color=MEAN if h > 0 else MUTED, stroke_width=mid_w))
    return g


class Calle(Scene):
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
        new.set_z_index(10)
        return new

    def construct(self):
        self.marks = {}
        title = Text("Máquinas de vectores soporte", font=FONT, weight=BOLD, color=INK).scale(1.0)
        sub = Text("la calle más ancha entre dos grupos", font=FONT, color=MUTED).scale(0.5)
        VGroup(title, sub).arrange(DOWN, buff=0.35)
        self.mark("title"); self.play(Write(title), run_time=1.4)
        self.mark("subtitle"); self.play(FadeIn(sub, shift=0.2 * UP)); self.wait(1.2)
        self.mark("title_out"); self.play(FadeOut(title), FadeOut(sub))

        # ---------- datos separables ----------
        rng = np.random.default_rng(4)
        th = np.deg2rad(35); u = np.array([np.cos(th), np.sin(th)]); v = np.array([-u[1], u[0]])
        while True:
            y = np.array([1 if i % 2 else -1 for i in range(28)])
            P = np.array([u * (c * 1.05 + 0.42 * rng.standard_normal()) + v * 1.15 * rng.standard_normal() for c in y])
            angs = np.arange(0, 180, 0.25)
            gaps = [max(gap_at(P, y, np.array([-np.sin(np.deg2rad(a)), np.cos(np.deg2rad(a))]) * s)[0] for s in (1, -1)) for a in angs]
            g = max(gaps)
            if 0.75 < g < 1.2 and np.abs(P).max() < 2.9:
                break
        best_a = angs[int(np.argmax(gaps))]
        P = P * 1.25; g = g * 1.25
        O = np.array([0, -0.75, 0])
        dots = VGroup(*[Dot(O + np.r_[p, 0], radius=0.09, color=col(c)) for p, c in zip(P, y)]).set_z_index(3)

        def orient(a):
            n = np.array([-np.sin(np.deg2rad(a)), np.cos(np.deg2rad(a))])
            r1, r2 = gap_at(P, y, n), gap_at(P, y, -n)
            return (n, *r1) if r1[0] >= r2[0] else (-n, *r2)

        # ---------- 1. muchas rayas ----------
        cap = self.swap(None, self.caption("Muchas rayas separan los dos grupos", "frenadas normales (violeta) y bruscas (naranja)"))
        self.mark("points"); self.play(LaggedStart(*[GrowFromCenter(d) for d in dots], lag_ratio=0.03), run_time=1.2)
        cands = []
        for d in (-22, 14, -9, 25, 4):
            n, gp, m, h = orient(best_a + d)
            if gp > 0:
                off = (np.random.default_rng(int(d + 50)).uniform(-0.8, 0.8)) * h
                cands.append(street(O, n, m + off, 0, mid_w=3.5))
        shown = None
        for s in cands:
            self.mark("cand")
            if shown is None:
                self.play(Create(s), run_time=0.5)
            else:
                self.play(ReplacementTransform(shown, s), run_time=0.45)
            shown = s; self.wait(0.25)
        q = Text("¿cuál elegir?", font=FONT, weight=SEMIBOLD, color=WARN).scale(0.5).move_to([4.9, 1.6, 0])
        self.play(FadeIn(q), run_time=0.5); self.wait(0.8)

        # ---------- 2. la calle más ancha ----------
        cap = self.swap(cap, self.caption("La SVM elige la calle más ancha", "gira y ensancha hasta tocar a los dos grupos"))
        self.play(FadeOut(q), FadeOut(shown), run_time=0.4)
        ang = ValueTracker(best_a + 30)
        def st():
            n, gp, m, h = orient(ang.get_value())
            return street(O, n, m, max(h, 0)).set_z_index(1)
        stv = always_redraw(st)
        def wlab():
            n, gp, m, h = orient(ang.get_value())
            r = max(gp, 0) / g
            grp = VGroup(Text("anchura", font=FONT, color=INK).scale(0.36))
            bar_bg = Rectangle(width=2.4, height=0.2, stroke_width=0, fill_color=LINE, fill_opacity=1)
            bar = Rectangle(width=max(0.01, 2.4 * r), height=0.2, stroke_width=0, fill_color=MEAN, fill_opacity=1).align_to(bar_bg, LEFT)
            grp.add(VGroup(bar_bg, bar)); grp.add(Text(f"{r:.0%}", font=FONT, color=INK).scale(0.36))
            grp.arrange(RIGHT, buff=0.2).move_to([4.6, 2.05, 0])
            return grp
        wl = always_redraw(wlab)
        self.mark("street"); self.play(FadeIn(stv), FadeIn(wl), run_time=0.7)
        self.mark("sweep"); self.play(ang.animate.set_value(best_a - 25), run_time=2.6, rate_func=smooth)
        self.mark("best"); self.play(ang.animate.set_value(best_a), run_time=1.6, rate_func=smooth)
        stv.clear_updaters(); wl.clear_updaters()
        n, gp, m, h = orient(best_a)
        t = P @ n
        sv = [i for i in range(len(P)) if (y[i] < 0 and abs(t[i] - (m - h)) < 1e-9) or (y[i] > 0 and abs(t[i] - (m + h)) < 1e-9)]
        lab = Text("la más ancha posible", font=FONT, weight=SEMIBOLD, color=GOOD).scale(0.44).move_to([4.6, 1.55, 0])
        self.play(FadeIn(lab), run_time=0.6); self.wait(1.0)

        # ---------- 3. vectores soporte ----------
        cap = self.swap(cap, self.caption("Solo mandan los puntos del borde", "los vectores soporte: el resto podría no existir"))
        rings = VGroup(*[Circle(radius=0.2, color=INK, stroke_width=3.5).move_to(dots[i].get_center()) for i in sv]).set_z_index(4)
        self.mark("sv"); self.play(FadeOut(lab), LaggedStart(*[Create(r) for r in rings], lag_ratio=0.3), run_time=1.0)
        others = VGroup(*[dots[i] for i in range(len(P)) if i not in sv])
        self.mark("hide"); self.play(others.animate.set_opacity(0.08), run_time=1.0)
        same = Text("la calle no se mueve", font=FONT, weight=SEMIBOLD, color=GOOD).scale(0.44).move_to([4.6, 1.55, 0])
        self.play(FadeIn(same), run_time=0.5); self.wait(1.2)
        self.mark("show"); self.play(others.animate.set_opacity(1), FadeOut(same), run_time=0.8)
        self.play(FadeOut(VGroup(stv, wl, rings, dots)), run_time=0.6)

        # ---------- 4. margen blando ----------
        cap = self.swap(cap, self.caption("Si no hay calle limpia: multa", "dejar pasar algunos puntos, pagando por cada uno"))
        r2 = np.random.default_rng(9)
        Y2 = np.array([1 if i % 2 else -1 for i in range(44)])
        Q = np.array([u * (c * 0.75 + 0.6 * r2.standard_normal()) + v * 1.2 * r2.standard_normal() for c in Y2])
        keep = np.abs(Q).max(1) < 2.9
        P2, y2 = Q[keep], Y2[keep]
        d2 = VGroup(*[Dot(O + np.r_[p, 0], radius=0.09, color=col(c)) for p, c in zip(P2, y2)]).set_z_index(3)
        self.mark("points2"); self.play(LaggedStart(*[GrowFromCenter(d) for d in d2], lag_ratio=0.02), run_time=1.0)

        def soft(C):
            s = SVC(kernel="linear", C=C).fit(P2, y2)
            w = s.coef_[0]; b = s.intercept_[0]; nw = np.linalg.norm(w)
            g = street(O, w / nw, -b / nw, 1 / nw).set_z_index(1)
            yf = y2 * s.decision_function(P2)
            rr = VGroup(*[Circle(radius=0.19, color=INK, stroke_width=3).move_to(d2[i].get_center()) for i in s.support_]).set_z_index(4)
            k = int((yf < 1 - 1e-6).sum())
            lab = Text("multa barata" if C < 0.5 else "multa media" if C < 5 else "multa cara", font=FONT, weight=SEMIBOLD, color=GOOD if C < 0.5 else MEAN if C < 5 else WARN).scale(0.46)
            lab2 = Text(f"{k} puntos dentro o al otro lado", font=FONT, color=INK).scale(0.36)
            info = VGroup(lab, lab2).arrange(DOWN, buff=0.15, aligned_edge=LEFT).move_to([4.7, 1.75, 0])
            return VGroup(g, rr, info)
        cur = soft(0.15)
        self.mark("soft"); self.play(FadeIn(cur), run_time=0.8); self.wait(1.4)
        for C in (1.0, 100.0):
            nxt = soft(C)
            self.mark("soft"); self.play(Transform(cur, nxt), run_time=1.2); self.wait(1.4)
        self.play(FadeOut(VGroup(cur, d2)), run_time=0.6)

        # ---------- 5. levantar ----------
        cap = self.swap(cap, self.caption("El truco: levantar los datos", "llegadas a tiempo (violeta) y muy pronto o muy tarde (naranja)"))
        r3 = np.random.default_rng(21)
        xs = np.r_[r3.uniform(-2.0, 2.0, 14), r3.uniform(2.9, 5.6, 7), -r3.uniform(2.9, 5.6, 7)]
        cs = np.r_[-np.ones(14), np.ones(14)]
        base = -2.9
        axis = Line([-6.2, base, 0], [6.2, base, 0], color=LINE, stroke_width=3)
        lift = ValueTracker(0.0)
        H = lambda x: x * x / 6.0
        dd = VGroup(*[Dot([x, base, 0], radius=0.1, color=col(c)) for x, c in zip(xs, cs)]).set_z_index(3)
        for dt, x in zip(dd, xs):
            dt.add_updater(lambda m, x=x: m.move_to([x, base + lift.get_value() * H(x), 0]))
        bowl = always_redraw(lambda: FunctionGraph(lambda x: base + lift.get_value() * H(x), x_range=[-6.1, 6.1], color=MUTED, stroke_width=2, stroke_opacity=min(1, lift.get_value() * 3)).set_z_index(0))
        self.mark("line1d"); self.play(Create(axis), LaggedStart(*[GrowFromCenter(d) for d in dd], lag_ratio=0.03), run_time=1.1)
        no = Text("ningún corte único los separa", font=FONT, weight=SEMIBOLD, color=WARN).scale(0.44).move_to([0, -3.55, 0])
        self.play(FadeIn(no), run_time=0.5); self.wait(0.9)
        self.add(bowl)
        self.mark("lift"); self.play(FadeOut(no), lift.animate.set_value(1.0), run_time=2.4, rate_func=smooth)
        cut = (H(2.0) + H(2.9)) / 2
        hl = Line([-6.2, base + cut, 0], [6.2, base + cut, 0], color=MEAN, stroke_width=5)
        cx = np.sqrt(cut * 6)
        drops = VGroup(*[DashedLine([s * cx, base + cut, 0], [s * cx, base, 0], color=MEAN, stroke_width=2.5) for s in (-1, 1)])
        self.mark("cut"); self.play(Create(hl), run_time=0.8)
        ok = Text("una raya recta arriba = dos cortes abajo", font=FONT, weight=SEMIBOLD, color=GOOD).scale(0.44).move_to([0, 2.1, 0])
        self.play(Create(drops), FadeIn(ok), run_time=0.9); self.wait(1.6)
        for dt in dd: dt.clear_updaters()
        bowl.clear_updaters()
        self.play(FadeOut(VGroup(axis, dd, bowl, hl, drops, ok)), run_time=0.6)

        # ---------- 6. kernel RBF ----------
        cap = self.swap(cap, self.caption("El kernel hace el truco por nosotros", "mide el parecido entre puntos y dibuja fronteras curvas"))
        r4 = np.random.default_rng(8)
        A = np.c_[r4.uniform(-4.6, 4.6, 130), r4.uniform(-2.7, 2.7, 130)]
        R = 1.8
        ya = np.where(np.hypot(A[:, 0] / 1.15, A[:, 1]) < R, -1, 1)
        O6 = np.array([0, -0.75, 0])
        da = VGroup(*[Dot(O6 + np.r_[p, 0], radius=0.075, color=col(c)) for p, c in zip(A, ya)]).set_z_index(3)
        self.mark("ring"); self.play(LaggedStart(*[GrowFromCenter(d) for d in da], lag_ratio=0.01), run_time=1.2)
        s = SVC(kernel="rbf", gamma=0.5, C=10).fit(A, ya)
        def bound(level):
            pts = []
            for a in np.linspace(0, 2 * np.pi, 241):
                dvec = np.array([np.cos(a), np.sin(a)]); lo, hi = 0.0, 4.5
                for _ in range(40):
                    mid = (lo + hi) / 2
                    if s.decision_function([dvec * mid])[0] < level: lo = mid
                    else: hi = mid
                pts.append(O6 + np.r_[dvec * lo, 0])
            return pts
        curve = VMobject(color=MEAN, stroke_width=5).set_points_smoothly(bound(0.0)).set_z_index(2)
        edges = VGroup(*[DashedVMobject(VMobject(color=MEAN, stroke_width=2.4).set_points_smoothly(bound(l)), num_dashes=70) for l in (-1, 1)]).set_z_index(2)
        self.mark("boundary"); self.play(Create(curve), run_time=1.6); self.play(Create(edges), run_time=1.0)
        svr = VGroup(*[Circle(radius=0.17, color=INK, stroke_width=2.6).move_to(da[i].get_center()) for i in s.support_]).set_z_index(4)
        self.mark("sv2"); self.play(LaggedStart(*[Create(c) for c in svr], lag_ratio=0.05), run_time=1.0); self.wait(1.4)
        self.play(FadeOut(VGroup(da, curve, edges, svr)), run_time=0.6)

        # ---------- Cierre ----------
        self.mark("legend")
        self.play(FadeOut(cap), run_time=0.6)
        e1 = Text("La calle más ancha se equivoca menos.", font=FONT, weight=BOLD, color=INK).scale(0.7)
        e2 = Text("Y solo hacen falta los puntos del borde.", font=FONT, color=MUTED).scale(0.48)
        VGroup(e1, e2).arrange(DOWN, buff=0.4)
        self.mark("end_text"); self.play(Write(e1), run_time=1.4); self.play(FadeIn(e2, shift=0.2 * UP), run_time=0.8)
        self.wait(2.0)
        self.mark("fade"); self.play(FadeOut(VGroup(e1, e2)), run_time=1)
        self.marks["total"] = [round(self.renderer.time, 3)]
        json.dump(self.marks, open("marks.json", "w"), indent=1)
