import json
import numpy as np
from manim import *
from scipy.stats import beta as Beta

# Paleta oscura de la serie ML visual
BG = "#0E131B"; INK = "#E4EAF3"; MUTED = "#9BA7B9"; LINE = "#2B3545"
MEAN = "#6EA6FF"; SAMPLE = "#AE9FF3"; POINT = "#FF9A55"; GOOD = "#4CC38A"; WARN = "#F08A5D"
FONT = "Inter"
config.background_color = BG


class Atipicos(Scene):
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
        title = Text("Valores atípicos", font=FONT, weight=BOLD, color=INK).scale(1.3)
        sub = Text("¿error o caso raro pero real?", font=FONT, color=MUTED).scale(0.5)
        VGroup(title, sub).arrange(DOWN, buff=0.35)
        self.mark("title"); self.play(Write(title), run_time=1.2)
        self.mark("subtitle"); self.play(FadeIn(sub, shift=0.2 * UP)); self.wait(1.2)
        self.mark("title_out"); self.play(FadeOut(title), FadeOut(sub))
        rng = np.random.default_rng(2)

        # ---------- 1. una coma ----------
        cap = self.swap(None, self.caption("Una coma mal puesta tuerce la recta", "un viaje gastó 33 L; alguien escribió 330"))
        X = lambda t: -5.5 + t / 24 * 11.0; Y = lambda c: -2.8 + (c - 15) / 30 * 4.3
        cx = np.sort(rng.uniform(1, 23, 14)); cy = 22 + 0.9 * cx + rng.normal(0, 1.3, 14)
        pts = VGroup(*[Dot([X(a), Y(b), 0], radius=0.08, color=SAMPLE) for a, b in zip(cx, cy)])
        F = Dot([X(12), Y(33), 0], radius=0.12, color=WARN)
        def line(b0, b1, col): return Line([X(0), Y(b0), 0], [X(24), Y(b0 + 24 * b1), 0], color=col, stroke_width=5)
        b = np.polyfit(np.r_[cx, 12], np.r_[cy, 33], 1)
        ln = line(b[1], b[0], MEAN)
        self.mark("pts"); self.play(FadeIn(pts, lag_ratio=0.05), FadeIn(F), run_time=0.8); self.play(Create(ln), run_time=0.6); self.wait(0.6)
        # 330 queda fuera del dibujo: se dibuja arriba con una flecha
        F2 = Dot([X(12), 2.2, 0], radius=0.12, color=WARN); tag = Text("330", font=FONT, weight=BOLD, color=WARN).scale(0.42).next_to(F2, RIGHT, buff=0.15)
        b2 = np.polyfit(np.r_[cx, 12], np.r_[cy, 330], 1)
        self.mark("jump"); self.play(Transform(F, F2), FadeIn(tag), Transform(ln, line(b2[1], b2[0], WARN)), run_time=1.2)
        p1 = Text(f"predicción con 20 t: {b2[1] + 20 * b2[0]:.0f} L".replace(".", ","), font=FONT, weight=BOLD, color=WARN).scale(0.42).move_to([0, -3.35, 0])
        self.play(FadeIn(p1), run_time=0.5); self.wait(1.0)
        ax_ = np.r_[cx, 12]; ay_ = np.r_[cy, 330]; i_, j_ = np.triu_indices(len(ax_), 1)
        s_ = np.median((ay_[j_] - ay_[i_]) / (ax_[j_] - ax_[i_])); rb = (s_, np.median(ay_ - s_ * ax_))
        p2 = Text(f"recta robusta: {rb[1] + 20 * rb[0]:.0f} L, como sin el error".replace(".", ","), font=FONT, weight=BOLD, color=GOOD).scale(0.42).move_to([0, -3.35, 0])
        self.mark("robust"); self.play(Transform(ln, line(rb[1], rb[0], GOOD)), ReplacementTransform(p1, p2), run_time=1.2); self.wait(1.6)
        self.play(FadeOut(VGroup(pts, F, tag, ln, p2)), run_time=0.6)

        # ---------- 2. error o real ----------
        cap = self.swap(cap, self.caption("Raro no quiere decir erróneo", "el contexto decide"))
        def card(lines, col):
            box = RoundedRectangle(width=4.8, height=2.3, corner_radius=0.15, stroke_color=col, stroke_width=3, fill_color=BG, fill_opacity=1)
            tx = VGroup(*[Text(l, font=FONT, color=INK).scale(0.38) for l in lines]).arrange(DOWN, aligned_edge=LEFT, buff=0.16).move_to(box)
            return VGroup(box, tx)
        c1 = card(["carga 13,9 t", "consumo 357 L/100 km", "ruta llana"], WARN).move_to([-3, 0.3, 0])
        c2 = card(["carga 16,9 t", "consumo 51 L/100 km", "ruta de montaña"], GOOD).move_to([3, 0.3, 0])
        l1 = Text("error: coma desplazada", font=FONT, weight=BOLD, color=WARN).scale(0.42).next_to(c1, DOWN, buff=0.3)
        l2 = Text("real: subir puertos gasta más", font=FONT, weight=BOLD, color=GOOD).scale(0.42).next_to(c2, DOWN, buff=0.3)
        self.mark("cards"); self.play(FadeIn(c1, shift=0.2 * UP), FadeIn(c2, shift=0.2 * UP), run_time=0.8); self.wait(0.8)
        self.mark("verdict"); self.play(FadeIn(l1), run_time=0.5); self.play(FadeIn(l2), run_time=0.5); self.wait(1.6)
        self.play(FadeOut(VGroup(c1, c2, l1, l2)), run_time=0.6)

        # ---------- 3. vallas ----------
        cap = self.swap(cap, self.caption("Buscar en una columna: las vallas", "el 50 % del medio, ampliado 1,5 veces a cada lado"))
        Xc = lambda v: -6.0 + v / 380 * 12.0
        vals = np.r_[rng.normal(33, 6, 160), [8.7, 9.1, 304, 330, 357, 258]]
        dts = VGroup(*[Dot([Xc(v), -0.5 + rng.uniform(-1, 1), 0], radius=0.05 if 10 < v < 100 else 0.1, color=SAMPLE if 10 < v < 100 else WARN) for v in vals])
        q1, q3 = np.percentile(vals, [25, 75]); lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
        box = Rectangle(width=Xc(q3) - Xc(q1), height=2.4, stroke_width=0, fill_color=MEAN, fill_opacity=0.2).move_to([(Xc(q1) + Xc(q3)) / 2, -0.5, 0])
        fen = VGroup(*[Line([Xc(v), -1.8, 0], [Xc(v), 0.8, 0], color=WARN, stroke_width=4) for v in (lo, hi)])
        ax = VGroup(*[Text(f"{t}", font=FONT, color=MUTED).scale(0.3).move_to([Xc(t), -2.1, 0]) for t in (0, 100, 200, 300)])
        self.mark("strip"); self.play(FadeIn(dts, lag_ratio=0.005), FadeIn(ax), run_time=1.0); self.play(FadeIn(box), Create(fen), run_time=0.8)
        nt = Text("fuera de las vallas: comas y galones", font=FONT, color=WARN).scale(0.4).move_to([0, 1.5, 0])
        self.mark("fence"); self.play(FadeIn(nt), run_time=0.5); self.wait(1.6)
        self.play(FadeOut(VGroup(dts, box, fen, ax, nt)), run_time=0.6)

        # ---------- 4. combinación ----------
        cap = self.swap(cap, self.caption("Atípico en la combinación", "20 toneladas es normal; 24 L es normal; las dos juntas, no"))
        bx = rng.uniform(0, 24, 120); by = 22 + 0.9 * bx + rng.normal(0, 1.5, 120)
        sc = VGroup(*[Dot([X(a), Y(b), 0], radius=0.05, color=SAMPLE).set_opacity(0.7) for a, b in zip(bx, by)])
        band = Polygon([X(0), Y(22 + 5), 0], [X(24), Y(22 + 21.6 + 5), 0], [X(24), Y(22 + 21.6 - 5), 0], [X(0), Y(22 - 5), 0], stroke_width=0, fill_color=MEAN, fill_opacity=0.15)
        bad = VGroup(*[Dot([X(a), Y(b), 0], radius=0.11, color=WARN) for a, b in [(19.1, 23.5), (20.3, 24.4), (21.6, 23.2), (22.5, 22.8)]])
        self.mark("biv"); self.play(FadeIn(sc, lag_ratio=0.005), run_time=0.8); self.play(FadeIn(band), run_time=0.5)
        self.mark("bad"); self.play(FadeIn(bad, scale=1.5), run_time=0.6)
        bt = Text("carga declarada, pero el camión iba vacío", font=FONT, weight=BOLD, color=WARN).scale(0.4).move_to([2.2, -3.35, 0])
        self.play(FadeIn(bt), run_time=0.5); self.wait(1.6)
        self.play(FadeOut(VGroup(sc, band, bad, bt)), run_time=0.6)

        # ---------- 5. qué hacer ----------
        cap = self.swap(cap, self.caption("¿Y qué se hace con cada uno?", "borrar todo lo raro se lleva por delante lo más interesante"))
        rows = [("coma o unidades", "corregir", MEAN), ("dato irrecuperable", "marcar como dato que falta", SAMPLE), ("caso raro real", "conservar y explicar (ruta de montaña)", GOOD), ("no se puede revisar", "método robusto", POINT)]
        grp = VGroup()
        for i, (a, b_, col) in enumerate(rows):
            ta = Text(a, font=FONT, color=INK).scale(0.42).move_to([-3.2, 1.2 - i * 0.9, 0])
            ar = Arrow([-1.5, 1.2 - i * 0.9, 0], [-0.5, 1.2 - i * 0.9, 0], color=MUTED, buff=0, stroke_width=3)
            tb = Text(b_, font=FONT, weight=BOLD, color=col).scale(0.42); tb.move_to([-0.3 + tb.width / 2, 1.2 - i * 0.9, 0])
            grp.add(VGroup(ta, ar, tb))
        for i, gq in enumerate(grp):
            self.mark("row"); self.play(FadeIn(gq, shift=0.2 * RIGHT), run_time=0.55)
        self.wait(1.8)
        self.play(*[FadeOut(m) for m in self.mobjects if m is not cap], run_time=0.6)

        # ---------- Cierre ----------
        self.mark("legend")
        self.play(FadeOut(cap), run_time=0.6)
        e1 = Text("Detectar es fácil. Decidir necesita contexto.", font=FONT, weight=BOLD, color=INK).scale(0.68)
        e2 = Text("Corrige los errores y conserva lo raro que es real.", font=FONT, color=MUTED).scale(0.48)
        VGroup(e1, e2).arrange(DOWN, buff=0.4)
        self.mark("end_text"); self.play(Write(e1), run_time=1.4); self.play(FadeIn(e2, shift=0.2 * UP), run_time=0.8)
        self.wait(2.0)
        self.mark("fade"); self.play(FadeOut(VGroup(e1, e2)), run_time=1)
        self.marks["total"] = [round(self.renderer.time, 3)]
        json.dump(self.marks, open("marks.json", "w"), indent=1)
