import json
import numpy as np
from manim import *

# Paleta oscura de la serie ML visual
BG = "#0E131B"; INK = "#E4EAF3"; MUTED = "#9BA7B9"; LINE = "#2B3545"
MEAN = "#6EA6FF"; SAMPLE = "#AE9FF3"; POINT = "#FF9A55"; GOOD = "#4CC38A"; WARN = "#F08A5D"
FONT = "Inter"
config.background_color = BG


class Desbalance(Scene):
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

    def construct(self):
        self.marks = {}
        title = Text("Clases desbalanceadas", font=FONT, weight=BOLD, color=INK).scale(1.2)
        sub = Text("cuando lo que buscas es raro", font=FONT, color=MUTED).scale(0.5)
        VGroup(title, sub).arrange(DOWN, buff=0.35)
        self.mark("title"); self.play(Write(title), run_time=1.2)
        self.mark("subtitle"); self.play(FadeIn(sub, shift=0.2 * UP)); self.wait(1.2)
        self.mark("title_out"); self.play(FadeOut(title), FadeOut(sub))
        rng = np.random.default_rng(6)

        # ---------- 1. mil camiones ----------
        cap = self.swap(None, self.caption("1 000 camiones, 23 averías esta semana", "cada cuadro es un camión; los naranjas se averiaron"))
        cols, rows, cs = 50, 20, 0.22
        bad = set(rng.choice(cols * rows, 23, replace=False).tolist())
        sq = VGroup(*[Square(cs * 0.78, stroke_width=0, fill_color=POINT if i in bad else SAMPLE, fill_opacity=1 if i in bad else 0.55).move_to([-cols * cs / 2 + cs / 2 + (i % cols) * cs, 1.75 - (i // cols) * cs, 0]) for i in range(cols * rows)])
        self.mark("grid"); self.play(FadeIn(sq, lag_ratio=0.001), run_time=1.4); self.wait(1.8)

        # ---------- 2. el acierto engaña ----------
        cap = self.swap(cap, self.caption("Un modelo que nunca avisa", "acierta con todos los camiones sanos"))
        self.play(sq.animate.scale(0.55).to_edge(LEFT, buff=0.6).shift(0.3 * DOWN), run_time=0.8)
        big = Text("97,7 % de acierto", font=FONT, weight=BOLD, color=GOOD).scale(0.8).move_to([3.0, 0.5, 0])
        self.mark("acc"); self.play(FadeIn(big, scale=1.2), run_time=0.7); self.wait(0.6)
        zero = Text("y encuentra 0 de 23 averías", font=FONT, weight=BOLD, color=WARN).scale(0.55).move_to([3.0, -0.5, 0])
        self.mark("zero"); self.play(FadeIn(zero, shift=0.2 * UP), run_time=0.7); self.wait(1.8)
        self.play(FadeOut(VGroup(sq, big, zero)), run_time=0.6)

        # ---------- 3. bajar el límite ----------
        cap = self.swap(cap, self.caption("Bajar el límite de aviso", "el modelo da a cada camión un riesgo; avisamos si pasa del límite"))
        # pares (límite, averías encontradas, revisiones) del juguete de la versión sencilla
        pts = [(50, 4, 4), (30, 8, 13), (20, 8, 18), (10, 9, 34), (5, 13, 91), (3.6, 16, 130)]
        lim = Text("límite: 50 %", font=FONT, weight=BOLD, color=INK).scale(0.6).move_to([0, 1.6, 0])
        bgA = Rectangle(width=8, height=0.5, stroke_width=0, fill_color=LINE, fill_opacity=1).move_to([0.8, 0.2, 0])
        bgB = Rectangle(width=8, height=0.5, stroke_width=0, fill_color=LINE, fill_opacity=1).move_to([0.8, -1.3, 0])
        la = Text("averías encontradas", font=FONT, color=MUTED).scale(0.38).next_to(bgA, UP, buff=0.12).align_to(bgA, LEFT)
        lb = Text("revisiones en el taller", font=FONT, color=MUTED).scale(0.38).next_to(bgB, UP, buff=0.12).align_to(bgB, LEFT)
        def fill(bg, frac, col):
            return Rectangle(width=max(0.02, 8 * frac), height=0.5, stroke_width=0, fill_color=col, fill_opacity=1).align_to(bg, LEFT).set_y(bg.get_y())
        def nums(a, b):
            return VGroup(Text(f"{a} de 23", font=FONT, weight=BOLD, color=GOOD).scale(0.42).next_to(bgA, LEFT, buff=0.25),
                          Text(f"{b}", font=FONT, weight=BOLD, color=WARN).scale(0.42).next_to(bgB, LEFT, buff=0.25))
        fa = fill(bgA, pts[0][1] / 23, GOOD); fb = fill(bgB, pts[0][2] / 140, WARN); nm = nums(pts[0][1], pts[0][2])
        self.mark("lim"); self.play(FadeIn(VGroup(lim, bgA, bgB, la, lb, fa, fb, nm)), run_time=0.8); self.wait(0.4)
        for l, a, b in pts[1:]:
            lim2 = Text(f"límite: {str(l).replace('.', ',')} %", font=FONT, weight=BOLD, color=INK).scale(0.6).move_to(lim)
            self.mark("step")
            self.play(Transform(lim, lim2), Transform(fa, fill(bgA, a / 23, GOOD)), Transform(fb, fill(bgB, b / 140, WARN)), Transform(nm, nums(a, b)), run_time=0.55)
            self.wait(0.15)
        tr = Text("más averías encontradas, a cambio de más revisiones", font=FONT, color=INK).scale(0.4).move_to([0, -2.5, 0])
        self.play(FadeIn(tr), run_time=0.5); self.wait(1.6)
        self.play(FadeOut(VGroup(lim, bgA, bgB, la, lb, fa, fb, nm, tr)), run_time=0.6)

        # ---------- 4. poner precio ----------
        cap = self.swap(cap, self.caption("Poner precio a cada error", "avería que se escapa: 4 000 €  ·  revisión innecesaria: 150 €"))
        def hbar(y, v, lab, col):
            w = v / 92000 * 7.5
            return VGroup(Text(lab, font=FONT, color=MUTED).scale(0.38).move_to([-4.6, y, 0]),
                          Rectangle(width=w, height=0.6, stroke_width=0, fill_color=col, fill_opacity=1).move_to([-3.0 + w / 2, y, 0]),
                          Text(f"{v:,} €".replace(",", " "), font=FONT, weight=BOLD, color=INK).scale(0.42).move_to([-3.0 + w + 0.9, y, 0]))
        h1 = hbar(1.2, 92000, "nunca avisar", MUTED); h2 = hbar(0.0, 76000, "límite 50 %", SAMPLE); h3 = hbar(-1.2, 51700, "límite 5 %", GOOD)
        for h in (h1, h2, h3):
            self.mark("cost"); self.play(FadeIn(h[0]), GrowFromEdge(h[1], LEFT), FadeIn(h[2]), run_time=0.7); self.wait(0.3)
        rule = Text("límite ideal = 150 / (150 + 4 000) ≈ 3,6 %", font=FONT, weight=BOLD, color=GOOD).scale(0.45).move_to([0, -2.5, 0])
        self.mark("rule"); self.play(FadeIn(rule, shift=0.2 * UP), run_time=0.6); self.wait(1.8)
        self.play(FadeOut(VGroup(h1, h2, h3, rule)), run_time=0.6)

        # ---------- 5. reequilibrar ----------
        cap = self.swap(cap, self.caption("Reequilibrar las clases infla el riesgo", "contar cada avería como 42 camiones sanos"))
        r1 = Text("40 %", font=FONT, weight=BOLD, color=WARN).scale(1.4).move_to([-2.6, 0.2, 0])
        r1l = Text("riesgo que dice el modelo", font=FONT, color=MUTED).scale(0.38).next_to(r1, DOWN, buff=0.3)
        r2 = Text("1,5 %", font=FONT, weight=BOLD, color=GOOD).scale(1.4).move_to([2.6, 0.2, 0])
        r2l = Text("riesgo real, tras corregir", font=FONT, color=MUTED).scale(0.38).next_to(r2, DOWN, buff=0.3)
        ar = Arrow([-1.2, 0.2, 0], [1.2, 0.2, 0], color=INK, stroke_width=4)
        self.mark("infl"); self.play(FadeIn(r1, scale=1.2), FadeIn(r1l), run_time=0.7); self.wait(0.4)
        self.mark("corr"); self.play(GrowArrow(ar), FadeIn(r2, scale=1.2), FadeIn(r2l), run_time=0.8)
        nt = Text("sin corregir, el taller se llena de revisiones", font=FONT, weight=BOLD, color=WARN).scale(0.42).move_to([0, -2.2, 0])
        self.play(FadeIn(nt), run_time=0.5); self.wait(1.8)
        self.play(*[FadeOut(m) for m in self.mobjects if m is not cap], run_time=0.6)

        # ---------- Cierre ----------
        self.mark("legend")
        self.play(FadeOut(cap), run_time=0.6)
        e1 = Text("Con casos raros, no mires el acierto.", font=FONT, weight=BOLD, color=INK).scale(0.7)
        e2 = Text("Pon precio a cada error y elige el límite con esos precios.", font=FONT, color=MUTED).scale(0.46)
        VGroup(e1, e2).arrange(DOWN, buff=0.4)
        self.mark("end_text"); self.play(Write(e1), run_time=1.4); self.play(FadeIn(e2, shift=0.2 * UP), run_time=0.8)
        self.wait(2.0)
        self.mark("fade"); self.play(FadeOut(VGroup(e1, e2)), run_time=1)
        self.marks["total"] = [round(self.renderer.time, 3)]
        json.dump(self.marks, open("marks.json", "w"), indent=1)
