import json
import numpy as np
from manim import *
from scipy.stats import beta as Beta

# Paleta oscura de la serie ML visual
BG = "#0E131B"; INK = "#E4EAF3"; MUTED = "#9BA7B9"; LINE = "#2B3545"
MEAN = "#6EA6FF"; SAMPLE = "#AE9FF3"; POINT = "#FF9A55"; GOOD = "#4CC38A"; WARN = "#F08A5D"
FONT = "Inter"
config.background_color = BG


class Separar(Scene):
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
        title = Text("Separar antes de mirar", font=FONT, weight=BOLD, color=INK).scale(1.25)
        sub = Text("la prueba tiene que parecerse al futuro", font=FONT, color=MUTED).scale(0.5)
        VGroup(title, sub).arrange(DOWN, buff=0.35)
        self.mark("title"); self.play(Write(title), run_time=1.2)
        self.mark("subtitle"); self.play(FadeIn(sub, shift=0.2 * UP)); self.wait(1.2)
        self.mark("title_out"); self.play(FadeOut(title), FadeOut(sub))

        def hbar(y, lab, v, col, mx=6.0, L=7.5):
            x0 = -3.2
            bgr = Rectangle(width=L, height=0.7, stroke_width=0, fill_color=LINE, fill_opacity=1).move_to([x0 + L / 2, y, 0])
            fg = Rectangle(width=max(0.02, L * v / mx), height=0.7, stroke_width=0, fill_color=col, fill_opacity=1)
            fg.move_to([x0 + L * v / mx / 2, y, 0])
            t = Text(lab, font=FONT, color=INK).scale(0.4).next_to(bgr, LEFT, buff=0.3)
            num = Text(f"{v:.2f} L".replace(".", ","), font=FONT, weight=BOLD, color=INK).scale(0.45).next_to(fg, RIGHT, buff=0.2)
            return VGroup(bgr, fg, t, num)

        # ---------- 1. promete y falla ----------
        cap = self.swap(None, self.caption("Un modelo que parecía casi perfecto", "predice el consumo diario de 60 camiones"))
        b1 = hbar(0.6, "error en la prueba", 0.81, SAMPLE, mx=7.0); b2 = hbar(-0.8, "error en producción", 5.41, POINT, mx=7.0)
        self.mark("promise"); self.play(FadeIn(b1[0]), FadeIn(b1[2]), GrowFromEdge(b1[1], LEFT), FadeIn(b1[3]), run_time=1.0); self.wait(1.0)
        self.mark("crash"); self.play(FadeIn(b2[0]), FadeIn(b2[2]), GrowFromEdge(b2[1], LEFT), FadeIn(b2[3]), run_time=1.2); self.wait(0.6)
        why = Text("usaba una columna del futuro: «litros que cargará mañana»", font=FONT, color=WARN).scale(0.42).move_to([0, -2.4, 0])
        self.mark("why"); self.play(FadeIn(why, shift=0.2 * UP), run_time=0.7); self.wait(2.2)
        self.play(FadeOut(VGroup(b1, b2, why)), run_time=0.6)

        # ---------- 2. azar frente a tiempo ----------
        cap = self.swap(cap, self.caption("¿Qué días apartas para la prueba?", "el consumo de un día se parece mucho al de sus vecinos"))
        n, w = 40, 0.3
        r = np.random.default_rng(3); rand = r.random(n) < 0.22
        cells = VGroup(*[Square(w * 0.82, stroke_width=0, fill_color=MEAN, fill_opacity=0.55).move_to([-n * w / 2 + w / 2 + i * w, 1.0, 0]) for i in range(n)])
        fut = VGroup(*[Square(w * 0.82, stroke_width=0, fill_color=POINT, fill_opacity=0.45).move_to([-n * w / 2 + w / 2 + (n + 0.6 + i) * w - 1.2, 1.0, 0]) for i in range(0)])
        lab = Text("historia (días)", font=FONT, color=MUTED).scale(0.34).next_to(cells, DOWN, buff=0.2)
        self.mark("days"); self.play(FadeIn(cells, lag_ratio=0.02), FadeIn(lab), run_time=1.2)
        ta = Text("días al azar", font=FONT, weight=SEMIBOLD, color=SAMPLE).scale(0.42).next_to(cells, UP, buff=0.25)
        self.mark("rand"); self.play(*[cells[i].animate.set_fill(SAMPLE, 1) for i in range(n) if rand[i]], FadeIn(ta), run_time=0.9)
        ba = hbar(-1.3, "promete", 2.18, SAMPLE, mx=3.0); br = hbar(-2.4, "real", 2.60, POINT, mx=3.0)
        self.play(FadeIn(ba), FadeIn(br), run_time=0.8); self.wait(1.4)
        tb = Text("los últimos días", font=FONT, weight=SEMIBOLD, color=SAMPLE).scale(0.42).next_to(cells, UP, buff=0.25)
        bt = hbar(-1.3, "promete", 2.45, SAMPLE, mx=3.0)
        self.mark("time"); self.play(*[cells[i].animate.set_fill(SAMPLE if i >= 32 else MEAN, 1 if i >= 32 else 0.55) for i in range(n)], ReplacementTransform(ta, tb), Transform(ba, bt), run_time=1.1)
        ok = Text("peor número, pero verdadero", font=FONT, color=GOOD).scale(0.42).move_to([0, -3.25, 0])
        self.mark("honest"); self.play(FadeIn(ok), run_time=0.6); self.wait(1.8)
        self.play(FadeOut(VGroup(cells, lab, tb, ba, br, ok)), run_time=0.6)

        # ---------- 3. camiones nuevos ----------
        cap = self.swap(cap, self.caption("¿Llegarán camiones nuevos?", "entonces la prueba son camiones enteros que el modelo no ha visto"))
        NC, ND, cs = 8, 24, 0.27
        r = np.random.default_rng(5); rm = r.random((NC, ND)) < 0.2
        grid = VGroup(*[Square(cs * 0.82, stroke_width=0, fill_color=SAMPLE if rm[c, d] else MEAN, fill_opacity=1 if rm[c, d] else 0.55).move_to([-ND * cs / 2 + cs / 2 + d * cs, 1.6 - c * cs, 0]) for c in range(NC) for d in range(ND)])
        gl = Text("cada fila, un camión", font=FONT, color=MUTED).scale(0.32).next_to(grid, LEFT, buff=0.3)
        ga = hbar(-1.3, "promete", 2.17, SAMPLE, mx=3.0); gr = hbar(-2.4, "camiones nuevos", 2.43, POINT, mx=3.0)
        self.mark("grid"); self.play(FadeIn(grid, lag_ratio=0.003), FadeIn(gl), run_time=1.2)
        self.play(FadeIn(ga), FadeIn(gr), run_time=0.8); self.wait(1.3)
        gb = hbar(-1.3, "promete", 2.38, SAMPLE, mx=3.0)
        self.mark("groups")
        self.play(*[grid[c * ND + d].animate.set_fill(SAMPLE if c in (2, 5) else MEAN, 1 if c in (2, 5) else 0.55) for c in range(NC) for d in range(ND)], Transform(ga, gb), run_time=1.1)
        self.wait(2.0)
        self.play(FadeOut(VGroup(grid, gl, ga, gr)), run_time=0.6)

        # ---------- 4. ¿lo sabría en ese momento? ----------
        cap = self.swap(cap, self.caption("¿Lo sabría en el momento de predecir?", "la pregunta para cada columna"))
        items = [("consumo de ayer", True), ("media de los 7 días anteriores", True), ("media de la semana centrada", False), ("litros que cargará mañana", False)]
        rows = VGroup()
        for i, (name, ok_) in enumerate(items):
            t = Text(name, font=FONT, color=INK).scale(0.46)
            badge = Text("sí" if ok_ else "no", font=FONT, weight=BOLD, color=GOOD if ok_ else WARN).scale(0.46)
            row = VGroup(t, badge); t.move_to([-1.2, 1.0 - i * 0.9, 0]); badge.move_to([3.6, 1.0 - i * 0.9, 0])
            rows.add(row)
        for i, row in enumerate(rows):
            self.mark("col"); self.play(FadeIn(row[0], shift=0.2 * RIGHT), run_time=0.5); self.play(FadeIn(row[1], scale=1.4), run_time=0.4)
            if not items[i][1]:
                self.play(row[0].animate.set_opacity(0.35), Create(Line(row[0].get_left() + 0.1 * LEFT, row[0].get_right() + 0.1 * RIGHT, color=WARN, stroke_width=3)), run_time=0.4)
        self.wait(1.8)
        self.play(*[FadeOut(m) for m in self.mobjects if m is not cap], run_time=0.6)

        # ---------- 5. selección con todos los datos ----------
        cap = self.swap(cap, self.caption("Elegir columnas mirando todo también es mirar", "2 000 columnas de puro ruido: lo honesto es acertar el 50 %"))
        s1 = hbar(0.4, "con todos los datos", 84.7, WARN, mx=100.0); s2 = hbar(-0.9, "solo con el entrenamiento", 49.5, MEAN, mx=100.0)
        for b, v in ((s1, 84.7), (s2, 49.5)):
            b[3].become(Text(f"{v:.0f} %", font=FONT, weight=BOLD, color=INK).scale(0.45).next_to(b[1], RIGHT, buff=0.2))
        self.mark("sel1"); self.play(FadeIn(s1[0]), FadeIn(s1[2]), GrowFromEdge(s1[1], LEFT), FadeIn(s1[3]), run_time=1.0); self.wait(1.0)
        self.mark("sel2"); self.play(FadeIn(s2[0]), FadeIn(s2[2]), GrowFromEdge(s2[1], LEFT), FadeIn(s2[3]), run_time=1.0)
        pl = Text("todo lo que aprende de los datos va dentro de la validación", font=FONT, color=GOOD).scale(0.42).move_to([0, -2.4, 0])
        self.mark("pipe"); self.play(FadeIn(pl, shift=0.2 * UP), run_time=0.6); self.wait(2.0)
        self.play(FadeOut(VGroup(s1, s2, pl)), run_time=0.6)

        # ---------- Cierre ----------
        self.mark("legend")
        self.play(FadeOut(cap), run_time=0.6)
        e1 = Text("Separa primero. Examina con el futuro.", font=FONT, weight=BOLD, color=INK).scale(0.7)
        e2 = Text("Y desconfía de los resultados demasiado buenos.", font=FONT, color=MUTED).scale(0.48)
        VGroup(e1, e2).arrange(DOWN, buff=0.4)
        self.mark("end_text"); self.play(Write(e1), run_time=1.4); self.play(FadeIn(e2, shift=0.2 * UP), run_time=0.8)
        self.wait(2.0)
        self.mark("fade"); self.play(FadeOut(VGroup(e1, e2)), run_time=1)
        self.marks["total"] = [round(self.renderer.time, 3)]
        json.dump(self.marks, open("marks.json", "w"), indent=1)
