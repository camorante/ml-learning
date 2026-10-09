import json
import numpy as np
from manim import *
from sklearn.tree import DecisionTreeClassifier

# Paleta oscura de la serie ML visual
BG = "#0E131B"; INK = "#E4EAF3"; MUTED = "#9BA7B9"; LINE = "#2B3545"
MEAN = "#6EA6FF"; SAMPLE = "#AE9FF3"; POINT = "#FF9A55"; GOOD = "#4CC38A"; WARN = "#F08A5D"
FONT = "Inter"
config.background_color = BG


def leaf_boxes(clf, box):
    """Rectángulos de las hojas de un árbol de sklearn: (x0, x1, y0, y1, prob_clase_1)."""
    t = clf.tree_; out = []
    def rec(i, b):
        if t.children_left[i] == -1:
            v = t.value[i][0]; out.append((*b, v[1] / v.sum())); return
        j, s = t.feature[i], t.threshold[i]
        bl, br = list(b), list(b)
        if j == 0: bl[1] = s; br[0] = s
        else: bl[3] = s; br[2] = s
        rec(t.children_left[i], bl); rec(t.children_right[i], br)
    rec(0, list(box)); return out


class Arboles(Scene):
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

    def dot(self, p, c, r=0.11, hollow=False):
        if hollow:
            return Circle(radius=r, color=POINT if c else SAMPLE, stroke_width=4).move_to(p)
        return Dot(p, radius=r, color=POINT if c else SAMPLE).set_stroke(BG, 4, background=True)

    def boxes(self, ax, clf, box):
        g = VGroup()
        for x0, x1, y0, y1, p in leaf_boxes(clf, box):
            col = POINT if p > 0.5 else SAMPLE
            r = Polygon(ax.c2p(x0, y0), ax.c2p(x1, y0), ax.c2p(x1, y1), ax.c2p(x0, y1), stroke_color=INK, stroke_width=2, stroke_opacity=0.55,
                        fill_color=col, fill_opacity=0.06 + 0.2 * abs(p - 0.5) * 2)
            g.add(r)
        return g

    def node(self, text, pos, leaf=None):
        t = Text(text, font=FONT, color=INK).scale(0.36)
        if leaf is None:
            box = RoundedRectangle(corner_radius=0.12, width=t.width + 0.35, height=0.5, stroke_color=INK, stroke_width=2, fill_color=BG, fill_opacity=1)
        else:
            col = POINT if leaf else SAMPLE
            box = RoundedRectangle(corner_radius=0.25, width=t.width + 0.4, height=0.5, stroke_color=col, stroke_width=2.5, fill_color=col, fill_opacity=0.2)
        return VGroup(box, t).move_to(pos)

    def edge(self, a, b, lab):
        l = Line(a.get_bottom(), b.get_top(), color=MUTED, stroke_width=2)
        t = Text(lab, font=FONT, color=MUTED).scale(0.3).next_to(l.get_center(), LEFT if b.get_x() < a.get_x() else RIGHT, buff=0.08)
        return VGroup(l, t)

    def construct(self):
        self.marks = {}
        title = Text("Árboles de decisión", font=FONT, weight=BOLD, color=INK).scale(1.1)
        sub = Text("decidir con preguntas de sí o no", font=FONT, color=MUTED).scale(0.5)
        VGroup(title, sub).arrange(DOWN, buff=0.35)
        self.mark("title"); self.play(Write(title), run_time=1.4)
        self.mark("subtitle"); self.play(FadeIn(sub, shift=0.2 * UP)); self.wait(1.2)
        self.mark("title_out"); self.play(FadeOut(title), FadeOut(sub))

        # ---------- 1. Entregas ----------
        cap = self.swap(None, self.caption("Un árbol hace preguntas de sí o no", "¿llegará tarde la entrega?"))
        DEL = [(12, 0, 0), (18, 0, 0), (25, 0, 0), (30, 0, 1), (40, 0, 1), (8, 1, 0), (15, 1, 1), (22, 1, 1), (28, 1, 1), (35, 1, 1)]
        ax = Axes(x_range=[0, 45], y_range=[0, 2], x_length=5.2, y_length=4.2, axis_config={"stroke_opacity": 0}, tips=False).move_to([-2.9, -0.8, 0])
        frame = Polygon(ax.c2p(0, 0), ax.c2p(45, 0), ax.c2p(45, 2), ax.c2p(0, 2), stroke_color=LINE, stroke_width=2)
        lab_p = Text("hora pico", font=FONT, color=MUTED).scale(0.34).next_to(ax.c2p(0, 1.5), LEFT, buff=0.15)
        lab_n = Text("normal", font=FONT, color=MUTED).scale(0.34).next_to(ax.c2p(0, 0.5), LEFT, buff=0.15)
        lab_d = Text("distancia →", font=FONT, color=MUTED).scale(0.34).next_to(ax.c2p(45, 0), DOWN, buff=0.15).shift(LEFT * 0.6)
        dots = VGroup(*[self.dot(ax.c2p(d, h + 0.5), y, 0.12) for d, h, y in DEL])
        self.mark("points"); self.play(Create(frame), FadeIn(lab_p), FadeIn(lab_n), FadeIn(lab_d), LaggedStart(*[GrowFromCenter(d) for d in dots], lag_ratio=0.1), run_time=1.6)
        self.wait(0.4)
        n1 = self.node("¿distancia ≤ 26.5 km?", [3.5, 1.2, 0])
        self.mark("q1")
        cut1 = Line(ax.c2p(26.5, 0), ax.c2p(26.5, 2), color=INK, stroke_width=4)
        right1 = Polygon(ax.c2p(26.5, 0), ax.c2p(45, 0), ax.c2p(45, 2), ax.c2p(26.5, 2), stroke_width=0, fill_color=POINT, fill_opacity=0.3)
        lf_r = self.node("tarde 4/4", [5.5, 0.0, 0], leaf=1)
        self.play(FadeIn(n1, shift=0.2 * DOWN), Create(cut1), run_time=0.9)
        e1 = self.edge(n1, lf_r, "no")
        self.play(FadeIn(right1), Create(e1), FadeIn(lf_r), run_time=0.9); self.bring_to_front(dots)
        self.wait(0.6)
        n2 = self.node("¿fuera de hora pico?", [2.1, 0.0, 0])
        e2 = self.edge(n1, n2, "sí")
        cut2 = Line(ax.c2p(0, 1), ax.c2p(26.5, 1), color=INK, stroke_width=3.5)
        bl = Polygon(ax.c2p(0, 0), ax.c2p(26.5, 0), ax.c2p(26.5, 1), ax.c2p(0, 1), stroke_width=0, fill_color=SAMPLE, fill_opacity=0.3)
        lf_n = self.node("a tiempo 3/3", [1.15, -1.25, 0], leaf=0)
        self.mark("q2"); self.play(Create(e2), FadeIn(n2), Create(cut2), run_time=0.9)
        e3 = self.edge(n2, lf_n, "sí")
        self.play(FadeIn(bl), Create(e3), FadeIn(lf_n), run_time=0.8); self.bring_to_front(dots)
        self.wait(0.5)
        n3 = self.node("¿distancia ≤ 11.5 km?", [3.5, -1.25, 0])
        e4 = self.edge(n2, n3, "no")
        cut3 = Line(ax.c2p(11.5, 1), ax.c2p(11.5, 2), color=INK, stroke_width=3)
        tl = Polygon(ax.c2p(0, 1), ax.c2p(11.5, 1), ax.c2p(11.5, 2), ax.c2p(0, 2), stroke_width=0, fill_color=SAMPLE, fill_opacity=0.3)
        tm = Polygon(ax.c2p(11.5, 1), ax.c2p(26.5, 1), ax.c2p(26.5, 2), ax.c2p(11.5, 2), stroke_width=0, fill_color=POINT, fill_opacity=0.3)
        lf_a = self.node("a tiempo 1/1", [2.6, -2.5, 0], leaf=0)
        lf_b = self.node("tarde 2/2", [4.5, -2.5, 0], leaf=1)
        self.mark("q3"); self.play(Create(e4), FadeIn(n3), Create(cut3), run_time=0.9)
        self.play(FadeIn(tl), FadeIn(tm), Create(self.edge(n3, lf_a, "sí")), Create(self.edge(n3, lf_b, "no")), FadeIn(lf_a), FadeIn(lf_b), run_time=0.9); self.bring_to_front(dots)
        self.mark("done1"); self.wait(1.6)
        self.play(*[FadeOut(m) for m in self.mobjects if m is not cap], run_time=0.7)

        # ---------- 2. Pureza ----------
        cap = self.swap(cap, self.caption("¿Qué pregunta primero?", "la que deja los grupos más puros"))
        ds = sorted(DEL, key=lambda r: r[0])
        X = lambda d: -5.6 + (d - 4) / 38 * 11.2
        strip = VGroup(*[self.dot([X(d), 0.2, 0], y, 0.17) for d, h, y in ds])
        nums = VGroup(*[Text(str(d), font=FONT, color=MUTED).scale(0.32).move_to([X(d), -0.3, 0]) for d, h, y in ds])
        self.mark("strip"); self.play(LaggedStart(*[GrowFromCenter(d) for d in strip], lag_ratio=0.08), FadeIn(nums), run_time=1.2)

        def gini(L):
            if not L: return 0
            p = sum(r[2] for r in L) / len(L); return 1 - p * p - (1 - p) ** 2
        def score(t):
            L = [r for r in ds if r[0] <= t]; R = [r for r in ds if r[0] > t]
            return (len(L) * gini(L) + len(R) * gini(R)) / 10
        tt = ValueTracker(6.0)
        cut = always_redraw(lambda: Line([X(tt.get_value()), 1.1, 0], [X(tt.get_value()), -0.7, 0], color=INK, stroke_width=4))
        lab = always_redraw(lambda: Text(f"desorden {score(tt.get_value()):.3f}", font=FONT, weight=SEMIBOLD,
                                         color=GOOD if score(tt.get_value()) < 0.27 else INK).scale(0.55).move_to([0, -2.0, 0]))
        barbg = Rectangle(width=6, height=0.22, stroke_width=0, fill_color=LINE, fill_opacity=1).move_to([0, -2.7, 0])
        bar = always_redraw(lambda: Rectangle(width=max(0.01, 6 * score(tt.get_value()) / 0.5), height=0.22, stroke_width=0,
                                              fill_color=GOOD if score(tt.get_value()) < 0.27 else SAMPLE, fill_opacity=1).align_to(barbg, LEFT).set_y(-2.7))
        self.play(Create(cut), FadeIn(lab), FadeIn(barbg), FadeIn(bar), run_time=0.6)
        self.mark("sweep"); self.play(tt.animate.set_value(41.0), run_time=3.2, rate_func=linear)
        self.mark("best"); self.play(tt.animate.set_value(26.5), run_time=1.4, rate_func=smooth)
        self.play(Flash(lab, color=GOOD, line_length=0.3, flash_radius=1.6), run_time=0.7)
        self.wait(0.8)
        for m in (cut, lab, bar): m.clear_updaters()
        self.play(*[FadeOut(m) for m in [strip, nums, cut, lab, barbg, bar]], run_time=0.6)

        # ---------- 3. Memorizar ----------
        cap = self.swap(cap, self.caption("Preguntar demasiado es memorizar", "cajitas alrededor del ruido"))
        rng = np.random.default_rng(5)
        def gen(n):
            P = rng.uniform(0.3, 9.7, (n, 2)); c = (((P[:, 0] - 5) ** 2 / 9 + (P[:, 1] - 5) ** 2 / 5) < 1.6).astype(int)
            flip = rng.random(n) < 0.14; c[flip] = 1 - c[flip]; return P, c
        P, c = gen(70)
        ax2 = Axes(x_range=[0, 10], y_range=[0, 10], x_length=8.2, y_length=5.3, axis_config={"stroke_opacity": 0}, tips=False).move_to([0, -0.85, 0])
        pts = VGroup(*[self.dot(ax2.c2p(*p), k, 0.085) for p, k in zip(P, c)])
        self.mark("noisy"); self.play(LaggedStart(*[GrowFromCenter(d) for d in pts], lag_ratio=0.02), run_time=1.2)
        part = None
        for depth in (2, 4, 12):
            clf = DecisionTreeClassifier(max_depth=depth, random_state=0).fit(P, c)
            nb = self.boxes(ax2, clf, (0, 10, 0, 10))
            lab = Text(f"profundidad {depth}: {clf.get_n_leaves()} hojas", font=FONT, color=INK).scale(0.4).move_to([0, -3.75, 0])
            self.mark("depth")
            if part is None:
                self.play(FadeIn(nb), FadeIn(lab), run_time=0.9); part, plab = nb, lab
            else:
                self.play(FadeOut(part), FadeIn(nb), FadeTransform(plab, lab), run_time=0.9); part, plab = nb, lab
            self.bring_to_front(pts); self.wait(0.7)
        Pn, cn = gen(24)
        new = VGroup(*[self.dot(ax2.c2p(*p), k, 0.1, hollow=True) for p, k in zip(Pn, cn)])
        clf = DecisionTreeClassifier(max_depth=12, random_state=0).fit(P, c)
        acc_tr = (clf.predict(P) == c).mean(); acc_te = (clf.predict(Pn) == cn).mean()
        lab2 = Text(f"aprendidos: {acc_tr:.0%} · nuevos: {acc_te:.0%}", font=FONT, color=WARN).scale(0.42).move_to([0, -3.75, 0])
        self.mark("new"); self.play(LaggedStart(*[GrowFromCenter(d) for d in new], lag_ratio=0.05), FadeTransform(plab, lab2), run_time=1.2)
        self.wait(1.4)
        self.play(FadeOut(VGroup(part, pts, new, lab2)), run_time=0.6)

        # ---------- 4. Escalera ----------
        cap = self.swap(cap, self.caption("Solo sabe cortar en recto", "una diagonal se convierte en escalera"))
        Q = rng.uniform(0.3, 9.7, (90, 2)); cq = (Q[:, 1] > Q[:, 0]).astype(int)
        qd = VGroup(*[self.dot(ax2.c2p(*p), k, 0.085) for p, k in zip(Q, cq)])
        diag = DashedLine(ax2.c2p(0, 0), ax2.c2p(10, 10), color=GOOD, stroke_width=4)
        self.mark("diag"); self.play(LaggedStart(*[GrowFromCenter(d) for d in qd], lag_ratio=0.02), Create(diag), run_time=1.2)
        part = None
        for depth in (2, 4, 8):
            clf = DecisionTreeClassifier(max_depth=depth, random_state=0).fit(Q, cq)
            nb = self.boxes(ax2, clf, (0, 10, 0, 10))
            lab = Text(f"profundidad {depth}: {clf.get_n_leaves()} hojas", font=FONT, color=INK).scale(0.4).move_to([0, -3.75, 0])
            self.mark("stairs")
            if part is None:
                self.play(FadeIn(nb), FadeIn(lab), run_time=0.8); part, plab = nb, lab
            else:
                self.play(FadeOut(part), FadeIn(nb), FadeTransform(plab, lab), run_time=0.8); part, plab = nb, lab
            self.bring_to_front(qd, diag); self.wait(0.5)
        self.wait(0.8)
        self.play(FadeOut(VGroup(part, qd, diag, plab)), run_time=0.6)

        # ---------- 5. Nervioso → bosque ----------
        cap = self.swap(cap, self.caption("Un árbol es nervioso", "otra muestra de los mismos datos, otro árbol"))
        axA = Axes(x_range=[0, 10], y_range=[0, 10], x_length=5.4, y_length=4.8, axis_config={"stroke_opacity": 0}, tips=False).move_to([-3.2, -1.0, 0])
        axB = Axes(x_range=[0, 10], y_range=[0, 10], x_length=5.4, y_length=4.8, axis_config={"stroke_opacity": 0}, tips=False).move_to([3.2, -1.0, 0])
        grpA = grpB = None
        for k in range(3):
            PA, cA = gen(60); PB, cB = gen(60)
            gA = VGroup(self.boxes(axA, DecisionTreeClassifier(max_depth=4, random_state=0).fit(PA, cA), (0, 10, 0, 10)), *[self.dot(axA.c2p(*p), q, 0.07) for p, q in zip(PA, cA)])
            gB = VGroup(self.boxes(axB, DecisionTreeClassifier(max_depth=4, random_state=0).fit(PB, cB), (0, 10, 0, 10)), *[self.dot(axB.c2p(*p), q, 0.07) for p, q in zip(PB, cB)])
            self.mark("nervous")
            if grpA is None:
                self.play(FadeIn(gA), FadeIn(gB), run_time=0.8)
            else:
                self.play(FadeOut(grpA), FadeOut(grpB), FadeIn(gA), FadeIn(gB), run_time=0.7)
            grpA, grpB = gA, gB
            self.wait(0.7)
        nxt = Text("La solución: muchos árboles que votan → Random Forest", font=FONT, weight=SEMIBOLD, color=GOOD).scale(0.48).move_to([0, -3.75, 0])
        nb = BackgroundRectangle(nxt, color=BG, fill_opacity=0.9, buff=0.12)
        self.mark("forest"); self.play(FadeIn(nb), Write(nxt), run_time=1.2)
        self.wait(1.4)

        # ---------- Cierre ----------
        self.mark("legend")
        self.play(FadeOut(VGroup(cap, grpA, grpB, nxt, nb)), run_time=0.7)
        end1 = Text("Preguntas simples, decisiones claras.", font=FONT, weight=BOLD, color=INK).scale(0.85)
        end2 = Text("Si no lo dejas memorizar.", font=FONT, color=MUTED).scale(0.5)
        VGroup(end1, end2).arrange(DOWN, buff=0.35)
        self.mark("end_text"); self.play(Write(end1), run_time=1.4); self.play(FadeIn(end2, shift=0.2 * UP), run_time=0.8)
        self.wait(2.0)
        self.mark("fade"); self.play(FadeOut(end1), FadeOut(end2), run_time=1)
        self.marks["total"] = [round(self.renderer.time, 3)]
        json.dump(self.marks, open("marks.json", "w"), indent=1)
