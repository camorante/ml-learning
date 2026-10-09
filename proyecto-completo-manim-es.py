import json
import numpy as np
from manim import *

# Paleta oscura de la serie ML visual
BG = "#0E131B"; INK = "#E4EAF3"; MUTED = "#9BA7B9"; LINE = "#2B3545"
MEAN = "#6EA6FF"; SAMPLE = "#AE9FF3"; POINT = "#FF9A55"; GOOD = "#4CC38A"; WARN = "#F08A5D"
FONT = "Inter"
config.background_color = BG


class Proyecto(Scene):
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
        E = json.load(open("embed.json"))   # lo genera codigo-proyecto/embed.py (escribe ../embed.json)
        title = Text("Proyecto completo", font=FONT, weight=BOLD, color=INK).scale(1.2)
        sub = Text("de los datos en bruto al taller, y después", font=FONT, color=MUTED).scale(0.5)
        VGroup(title, sub).arrange(DOWN, buff=0.35)
        self.mark("title"); self.play(Write(title), run_time=1.2)
        self.mark("subtitle"); self.play(FadeIn(sub, shift=0.2 * UP)); self.wait(1.2)
        self.mark("title_out"); self.play(FadeOut(title), FadeOut(sub))

        # ---------- 1. pasos ----------
        cap = self.swap(None, self.caption("Nueve pasos, en este orden", "cada uno es un tema de la serie"))
        pasos = ["Separar", "Mirar", "Limpiar", "Atípicos", "Huecos", "Variables", "Modelo y límite", "Examen", "Vigilar"]
        boxes = VGroup()
        for i, p in enumerate(pasos):
            col = GOOD if i == 8 else (MEAN if i < 8 else MEAN)
            r = RoundedRectangle(width=2.9, height=0.75, corner_radius=0.12, stroke_color=col, stroke_width=3, fill_color=BG, fill_opacity=1)
            t = Text(f"{i + 1}. {p}", font=FONT, color=INK).scale(0.4).move_to(r)
            boxes.add(VGroup(r, t).move_to([-4.0 + (i % 3) * 4.0, 1.2 - (i // 3) * 1.25, 0]))
        self.mark("steps"); self.play(LaggedStart(*[FadeIn(b, shift=0.15 * UP) for b in boxes], lag_ratio=0.18), run_time=2.4); self.wait(1.4)
        self.play(FadeOut(boxes), run_time=0.6)

        # ---------- 2. informe y realidad ----------
        cap = self.swap(cap, self.caption("El informe y la realidad", "coste por semana: lo que promete el examen y lo que pasó después"))
        A = E["abl"]; mx = 70000; L = -3.2; W = 8.0
        def bars(inf, real, y0):
            out = VGroup()
            for k, (v, lab, col) in enumerate([(inf, "informe", SAMPLE), (real, "de verdad", MEAN)]):
                y = y0 - k * 0.75; w = v / mx * W
                out.add(VGroup(Text(lab, font=FONT, color=MUTED).scale(0.36).move_to([L - 1.1, y, 0]),
                               Rectangle(width=w, height=0.5, stroke_width=0, fill_color=col, fill_opacity=1).move_to([L + w / 2, y, 0]),
                               Text(f"{v:,} €".replace(",", " "), font=FONT, weight=BOLD, color=INK).scale(0.36).move_to([L + w + 0.85, y, 0])))
            return out
        t1 = Text("todo bien hecho", font=FONT, weight=BOLD, color=GOOD).scale(0.42).move_to([L + 0.9, 1.6, 0])
        g1 = bars(A["11111"][0], A["11111"][1], 1.0)
        self.mark("good"); self.play(FadeIn(t1), *[GrowFromEdge(b[1], LEFT) for b in g1], *[FadeIn(b[0]) for b in g1], *[FadeIn(b[2]) for b in g1], run_time=1.0); self.wait(0.8)
        t2 = Text("historial que incluye la semana a predecir", font=FONT, weight=BOLD, color=WARN).scale(0.42).move_to([L + 2.3, -0.6, 0])
        g2 = bars(A["11011"][0], A["11011"][1], -1.2)
        self.mark("leak"); self.play(FadeIn(t2), *[GrowFromEdge(b[1], LEFT) for b in g2], *[FadeIn(b[0]) for b in g2], *[FadeIn(b[2]) for b in g2], run_time=1.0)
        nt = Text("una fuga no da error: da un informe bonito", font=FONT, color=INK).scale(0.4).move_to([0, -2.9, 0])
        self.play(FadeIn(nt), run_time=0.5); self.wait(1.8)
        self.play(FadeOut(VGroup(t1, g1, t2, g2, nt)), run_time=0.6)

        # ---------- 3. el sensor ----------
        cap = self.swap(cap, self.caption("Semana 80: cambia el sensor", "120 camiones con una caja nueva que mide un 30 % menos de vibración"))
        edges = np.array(E["edges"]); href = np.array(E["href"], float); href /= href.sum()
        hs = np.array(E["hsem"][21], float); hs /= hs.sum(); h0 = np.array(E["hsem"][10], float); h0 /= h0.sum()
        X = lambda v: -5.5 + (v - 1) / 7 * 7.0; Y = lambda f: -2.3 + f * 26
        ref = VGroup(*[Rectangle(width=X(edges[i + 1]) - X(edges[i]), height=max(0.001, Y(f) + 2.3), stroke_width=0, fill_color=MUTED, fill_opacity=0.35).move_to([(X(edges[i]) + X(edges[i + 1])) / 2, (Y(f) - 2.3) / 2, 0]) for i, f in enumerate(href)])
        step = lambda h: VMobject(stroke_color=POINT, stroke_width=5).set_points_as_corners([p for i, f in enumerate(h) for p in ([X(edges[i]), Y(f), 0], [X(edges[i + 1]), Y(f), 0])])
        cur = step(h0); ax = Line([X(1), -2.3, 0], [X(8), -2.3, 0], color=MUTED)
        xl = Text("vibración medida", font=FONT, color=MUTED).scale(0.32).next_to(ax, DOWN, buff=0.15)
        self.mark("hist"); self.play(FadeIn(ref), Create(ax), FadeIn(xl), Create(cur), run_time=1.0)
        psi0 = Text("índice de cambio: 0,03", font=FONT, color=GOOD).scale(0.42).move_to([4.0, 1.2, 0])
        av0 = Text("avisos: 100 por semana", font=FONT, color=GOOD).scale(0.42).move_to([4.0, 0.5, 0])
        self.play(FadeIn(psi0), FadeIn(av0), run_time=0.5); self.wait(0.5)
        psi1 = Text("índice de cambio: 0,47", font=FONT, weight=BOLD, color=WARN).scale(0.42).move_to([4.0, 1.2, 0])
        av1 = Text("avisos: 63", font=FONT, weight=BOLD, color=WARN).scale(0.42).move_to([4.0, 0.5, 0])
        self.mark("shift"); self.play(Transform(cur, step(hs)), Transform(psi0, psi1), Transform(av0, av1), run_time=1.2)
        nt = Text("los motores no han cambiado; los números, sí", font=FONT, color=INK).scale(0.38).move_to([4.0, -0.3, 0])
        self.play(FadeIn(nt), run_time=0.5); self.wait(1.6)
        self.play(FadeOut(VGroup(ref, ax, xl, cur, psi0, av0, nt)), run_time=0.6)

        # ---------- 4. reaccionar ----------
        cap = self.swap(cap, self.caption("Buscar la causa antes de reentrenar", "ahorro frente a no avisar, media de 4 semanas"))
        P = E["prod"]; N = [r["coste"] for r in P["nunca"]]
        def sav(k):
            c = [r["coste"] for r in P[k]]; return [1 - sum(c[max(0, i - 3):i + 1]) / sum(N[max(0, i - 3):i + 1]) for i in range(len(c))]
        Xs = lambda s: -5.5 + (s - 60) / 43 * 11; Ys = lambda v: -2.4 + (v - 0.3) / 0.55 * 4.2
        axs = Line([Xs(60), -2.4, 0], [Xs(103), -2.4, 0], color=MUTED)
        v80 = DashedLine([Xs(80), -2.4, 0], [Xs(80), 1.9, 0], color=INK); l80 = Text("semana 80", font=FONT, color=MUTED).scale(0.32).next_to(v80, UP, buff=0.08)
        curve = lambda k, col, w: VMobject(stroke_color=col, stroke_width=w).set_points_as_corners([[Xs(60 + i), Ys(v), 0] for i, v in enumerate(sav(k))])
        c0 = curve("nada", MUTED, 4); c2 = curve("arreglar", GOOD, 5)
        self.mark("react"); self.play(Create(axs), Create(v80), FadeIn(l80), Create(c0), run_time=1.4)
        ln = Text("no hacer nada", font=FONT, color=MUTED).scale(0.36).move_to([Xs(97), Ys(0.38), 0])
        self.play(FadeIn(ln), run_time=0.4)
        lg = Text("corregir el sensor", font=FONT, weight=BOLD, color=GOOD).scale(0.38).move_to([Xs(96), Ys(0.75), 0])
        self.mark("fix"); self.play(Create(c2), FadeIn(lg), run_time=1.2); self.wait(1.8)
        self.play(*[FadeOut(m) for m in self.mobjects if m is not cap], run_time=0.6)

        # ---------- Cierre ----------
        self.mark("legend")
        self.play(FadeOut(cap), run_time=0.6)
        e1 = Text("Un modelo no se termina al entrenarlo.", font=FONT, weight=BOLD, color=INK).scale(0.68)
        e2 = Text("Se vigila mientras se usa.", font=FONT, color=MUTED).scale(0.5)
        VGroup(e1, e2).arrange(DOWN, buff=0.4)
        self.mark("end_text"); self.play(Write(e1), run_time=1.4); self.play(FadeIn(e2, shift=0.2 * UP), run_time=0.8)
        self.wait(2.0)
        self.mark("fade"); self.play(FadeOut(VGroup(e1, e2)), run_time=1)
        self.marks["total"] = [round(self.renderer.time, 3)]
        json.dump(self.marks, open("marks.json", "w"), indent=1)
