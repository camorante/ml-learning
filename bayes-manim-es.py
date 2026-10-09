import json
import numpy as np
from manim import *
from scipy.stats import beta as Beta

# Paleta oscura de la serie ML visual
BG = "#0E131B"; INK = "#E4EAF3"; MUTED = "#9BA7B9"; LINE = "#2B3545"
MEAN = "#6EA6FF"; SAMPLE = "#AE9FF3"; POINT = "#FF9A55"; GOOD = "#4CC38A"; WARN = "#F08A5D"
FONT = "Inter"
config.background_color = BG


class Bayes(Scene):
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
        title = Text("Bayes", font=FONT, weight=BOLD, color=INK).scale(1.5)
        sub = Text("cambiar de opinión, pero con orden", font=FONT, color=MUTED).scale(0.5)
        VGroup(title, sub).arrange(DOWN, buff=0.35)
        self.mark("title"); self.play(Write(title), run_time=1.2)
        self.mark("subtitle"); self.play(FadeIn(sub, shift=0.2 * UP)); self.wait(1.2)
        self.mark("title_out"); self.play(FadeOut(title), FadeOut(sub))

        # ---------- 1. mil camiones ----------
        cap = self.swap(None, self.caption("Salta una alerta: ¿está averiado?", "el sistema avisa en el 90 % de las averías y se equivoca en el 5 % de los sanos"))
        cols, rows, cs = 50, 20, 0.23
        ox, oy = -cols * cs / 2 + cs / 2, 1.85
        sq = VGroup(*[Square(cs * 0.78, stroke_width=0, fill_color=LINE, fill_opacity=1).move_to([ox + (i % cols) * cs, oy - (i // cols) * cs, 0]) for i in range(1000)])
        self.mark("grid"); self.play(FadeIn(sq, lag_ratio=0.0005), run_time=1.4)
        t1 = Text("1 000 camiones · 20 con avería (2 %)", font=FONT, weight=SEMIBOLD, color=INK).scale(0.42).move_to([0, -3.2, 0])
        self.mark("faulty"); self.play(*[sq[i].animate.set_fill(POINT, 0.35) for i in range(20)], FadeIn(t1), run_time=1.0); self.wait(0.6)
        t2 = Text("alertas: 18 verdaderas + 49 falsas = 67", font=FONT, weight=SEMIBOLD, color=INK).scale(0.42).move_to([0, -3.2, 0])
        self.mark("alerts")
        self.play(*[sq[i].animate.set_fill(POINT, 1) for i in range(18)], *[sq[i].animate.set_fill(SAMPLE, 1) for i in range(20, 69)], ReplacementTransform(t1, t2), run_time=1.4)
        self.wait(0.8)
        rest = VGroup(*[sq[i] for i in list(range(18, 20)) + list(range(69, 1000))])
        t3 = Text("de 67 alertas, solo 18 son averías: 27 %", font=FONT, weight=BOLD, color=POINT).scale(0.5).move_to([0, -3.2, 0])
        self.mark("answer"); self.play(rest.animate.set_opacity(0.12), ReplacementTransform(t2, t3), run_time=1.0); self.wait(1.8)
        self.play(FadeOut(VGroup(sq, t3)), run_time=0.6)

        # ---------- 2. odds ----------
        cap = self.swap(cap, self.caption("Lo que creías × la fuerza de la pista", "cada alerta multiplica las apuestas a favor por 18 (90 % / 5 %)"))
        H = 3.6; y0 = -2.8
        def bar(x, p, lab):
            g = VGroup(Rectangle(width=1.3, height=H, stroke_width=0, fill_color=LINE, fill_opacity=1).move_to([x, y0 + H / 2, 0]),
                       Rectangle(width=1.3, height=max(0.02, H * p), stroke_width=0, fill_color=POINT, fill_opacity=1).move_to([x, y0 + H * p / 2, 0]),
                       Text(f"{p * 100:.0f} %", font=FONT, weight=BOLD, color=INK).scale(0.5).move_to([x, y0 + H + 0.35, 0]),
                       Text(lab, font=FONT, color=MUTED).scale(0.34).move_to([x, y0 - 0.35, 0]))
            return g
        b0, b1, b2 = bar(-4.5, 0.02, "antes"), bar(0, 18 / 67, "tras una alerta"), bar(4.5, 324 / 373, "tras dos alertas")
        ar1 = VGroup(Arrow([-3.6, -1.0, 0], [-0.9, -1.0, 0], color=MUTED, buff=0, stroke_width=4), Text("× 18", font=FONT, weight=BOLD, color=INK).scale(0.5).move_to([-2.25, -0.6, 0]))
        ar2 = VGroup(Arrow([0.9, -1.0, 0], [3.6, -1.0, 0], color=MUTED, buff=0, stroke_width=4), Text("× 18", font=FONT, weight=BOLD, color=INK).scale(0.5).move_to([2.25, -0.6, 0]))
        o1 = Text("1 : 49", font=FONT, color=MUTED).scale(0.36).move_to([-4.5, -3.55, 0]); o2 = Text("18 : 49", font=FONT, color=MUTED).scale(0.36).move_to([0, -3.55, 0]); o3 = Text("324 : 49", font=FONT, color=MUTED).scale(0.36).move_to([4.5, -3.55, 0])
        self.mark("b0"); self.play(FadeIn(b0), FadeIn(o1), run_time=0.7)
        self.mark("b1"); self.play(GrowArrow(ar1[0]), FadeIn(ar1[1]), run_time=0.6); self.play(FadeIn(b1), FadeIn(o2), run_time=0.7); self.wait(0.6)
        self.mark("b2"); self.play(GrowArrow(ar2[0]), FadeIn(ar2[1]), run_time=0.6); self.play(FadeIn(b2), FadeIn(o3), run_time=0.7); self.wait(1.6)
        self.play(FadeOut(VGroup(b0, b1, b2, ar1, ar2, o1, o2, o3)), run_time=0.6)

        # ---------- 3. proporción con pocos datos ----------
        cap = self.swap(cap, self.caption("Aprender con pocos datos", "¿qué parte de sus viajes llega tarde un conductor nuevo?"))
        X = lambda p: -6.0 + 12.0 * p; Y0 = -3.0; S = 4.2
        axis = Line([X(0), Y0, 0], [X(1), Y0, 0], color=LINE, stroke_width=3)
        ticks = VGroup(*[Text(f"{int(v * 100)} %", font=FONT, color=MUTED).scale(0.28).move_to([X(v), Y0 - 0.28, 0]) for v in (0, 0.2, 0.4, 0.6, 0.8, 1)])
        def curve(a, b):
            xs = np.linspace(0.002, 0.998, 300); ys = Beta.pdf(xs, a, b); ys = ys / max(ys.max(), 5.0) * S
            return VMobject(color=MEAN, stroke_width=6).set_points_smoothly([[X(x), Y0 + y, 0] for x, y in zip(xs, ys)])
        trips = [1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0]
        a, b = 2, 8
        cur = curve(a, b)
        lab = Text("lo que se sabía de la flota: en torno al 20 %", font=FONT, color=MUTED).scale(0.34).move_to([3.0, 1.95, 0])
        self.mark("prior"); self.play(Create(axis), FadeIn(ticks), Create(cur), FadeIn(lab), run_time=1.2)
        raw = None; mline = None; k = 0
        for i, tr in enumerate(trips):
            k += tr; a += tr; b += 1 - tr; n = i + 1
            new = curve(a, b)
            m = a / (a + b); r = k / n
            nm = VGroup(DashedLine([X(m), Y0, 0], [X(m), Y0 + S + 0.05, 0], color=MEAN, stroke_width=3), Text(f"Bayes {m:.0%}", font=FONT, weight=SEMIBOLD, color=MEAN).scale(0.36).move_to([X(m) + 0.75, Y0 + S + 0.15, 0]))
            nr = VGroup(Line([X(r), Y0 - 0.05, 0], [X(r), Y0 + 0.9, 0], color=POINT, stroke_width=5), Text(f"cuenta directa {r:.0%}", font=FONT, color=POINT).scale(0.32).move_to([X(min(max(r, 0.12), 0.88)), Y0 + 1.15, 0]))
            dot = Dot([X(0.03) + i * 0.32, 1.95, 0], radius=0.12, color=POINT if tr else SAMPLE)
            self.mark("trip")
            anims = [Transform(cur, new), FadeIn(dot, scale=0.5)]
            anims += [FadeIn(nm), FadeIn(nr)] if mline is None else [Transform(mline, nm), Transform(raw, nr)]
            self.play(*anims, run_time=0.55 if i else 0.8)
            if mline is None: mline, raw = nm, nr
            self.wait(0.12)
            if i == 0: self.add(dot)
            else: self.add(dot)
        self.wait(1.4)
        self.play(*[FadeOut(m) for m in self.mobjects if m is not cap], run_time=0.6)

        # ---------- 4. Naive Bayes ----------
        cap = self.swap(cap, self.caption("Naive Bayes: un factor por pista", "¿fue agresivo este viaje? se parte de 1 de cada 10"))
        sig = [("frenadas bruscas", 6.0), ("acelerones", 0.5 / 0.15), ("exceso de velocidad", 5.0)]
        odds = 0.1 / 0.9
        def pbar(p):
            return VGroup(Rectangle(width=9.0, height=0.45, stroke_width=0, fill_color=LINE, fill_opacity=1).move_to([0, -2.6, 0]),
                          Rectangle(width=max(0.02, 9.0 * p), height=0.45, stroke_width=0, fill_color=POINT, fill_opacity=1).move_to([-4.5 + 9.0 * p / 2, -2.6, 0]),
                          Text(f"probabilidad de viaje agresivo: {p:.0%}", font=FONT, weight=BOLD, color=INK).scale(0.44).move_to([0, -1.95, 0]))
        pb = pbar(odds / (1 + odds))
        self.mark("nb"); self.play(FadeIn(pb), run_time=0.7)
        for i, (name, lr) in enumerate(sig):
            odds *= lr
            chip = VGroup(RoundedRectangle(width=4.2, height=0.6, corner_radius=0.3, stroke_color=POINT, stroke_width=2, fill_color=BG, fill_opacity=1),
                          Text(f"{name}  × {lr:.1f}".replace(".", ","), font=FONT, color=INK).scale(0.36)).move_to([-4.0 + i * 4.0, 1.0, 0])
            chip[1].move_to(chip[0])
            self.mark("sig"); self.play(FadeIn(chip, shift=0.2 * DOWN), Transform(pb, pbar(odds / (1 + odds))), run_time=0.9); self.wait(0.4)
        warn = Text("si dos pistas miden lo mismo, cuenta dos veces: demasiado seguro", font=FONT, color=WARN).scale(0.38).move_to([0, -0.4, 0])
        self.mark("warn"); self.play(FadeIn(warn), run_time=0.6); self.wait(1.8)
        self.play(*[FadeOut(m) for m in self.mobjects if m is not cap], run_time=0.6)

        # ---------- Cierre ----------
        self.mark("legend")
        self.play(FadeOut(cap), run_time=0.6)
        e1 = Text("Lo que ya sabías, corregido por la pista nueva.", font=FONT, weight=BOLD, color=INK).scale(0.68)
        e2 = Text("Y sin olvidar lo raro que es lo que buscas.", font=FONT, color=MUTED).scale(0.48)
        VGroup(e1, e2).arrange(DOWN, buff=0.4)
        self.mark("end_text"); self.play(Write(e1), run_time=1.4); self.play(FadeIn(e2, shift=0.2 * UP), run_time=0.8)
        self.wait(2.0)
        self.mark("fade"); self.play(FadeOut(VGroup(e1, e2)), run_time=1)
        self.marks["total"] = [round(self.renderer.time, 3)]
        json.dump(self.marks, open("marks.json", "w"), indent=1)
