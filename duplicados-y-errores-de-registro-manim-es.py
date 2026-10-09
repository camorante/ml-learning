import json
import numpy as np
from manim import *
from scipy.stats import beta as Beta

# Paleta oscura de la serie ML visual
BG = "#0E131B"; INK = "#E4EAF3"; MUTED = "#9BA7B9"; LINE = "#2B3545"
MEAN = "#6EA6FF"; SAMPLE = "#AE9FF3"; POINT = "#FF9A55"; GOOD = "#4CC38A"; WARN = "#F08A5D"
FONT = "Inter"
config.background_color = BG


class Registros(Scene):
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
        title = Text("Duplicados y errores de registro", font=FONT, weight=BOLD, color=INK).scale(1.05)
        sub = Text("lo que el camión hizo frente a lo que llegó al servidor", font=FONT, color=MUTED).scale(0.48)
        VGroup(title, sub).arrange(DOWN, buff=0.35)
        self.mark("title"); self.play(Write(title), run_time=1.2)
        self.mark("subtitle"); self.play(FadeIn(sub, shift=0.2 * UP)); self.wait(1.2)
        self.mark("title_out"); self.play(FadeOut(title), FadeOut(sub))

        def hbar(y, lab, v, col, mx, unit=" km", L=7.5):
            x0 = -3.0
            bgr = Rectangle(width=L, height=0.7, stroke_width=0, fill_color=LINE, fill_opacity=1).move_to([x0 + L / 2, y, 0])
            fg = Rectangle(width=max(0.02, L * v / mx), height=0.7, stroke_width=0, fill_color=col, fill_opacity=1).move_to([x0 + L * v / mx / 2, y, 0])
            t = Text(lab, font=FONT, color=INK).scale(0.4).next_to(bgr, LEFT, buff=0.3)
            num = Text(f"{v:,.0f}".replace(",", " ") + unit, font=FONT, weight=BOLD, color=INK).scale(0.45).next_to(fg, RIGHT, buff=0.2)
            return VGroup(bgr, fg, t, num)
        def card(lines, col=LINE, w=4.6):
            box = RoundedRectangle(width=w, height=0.45 * len(lines) + 0.35, corner_radius=0.15, stroke_color=col, stroke_width=2, fill_color=BG, fill_opacity=1)
            tx = VGroup(*[Text(l, font=FONT, color=INK).scale(0.34) for l in lines]).arrange(DOWN, aligned_edge=LEFT, buff=0.14).move_to(box)
            return VGroup(box, tx)

        # ---------- 1. el problema ----------
        cap = self.swap(None, self.caption("Un día de un camión", "1 200 mensajes enviados, 1 260 recibidos"))
        b1 = hbar(0.5, "suma del GPS", 1221, WARN, 1300); b2 = hbar(-0.8, "odómetro", 544, GOOD, 1300)
        self.mark("bars"); self.play(FadeIn(b2[0]), FadeIn(b2[2]), GrowFromEdge(b2[1], LEFT), FadeIn(b2[3]), run_time=0.9)
        self.play(FadeIn(b1[0]), FadeIn(b1[2]), GrowFromEdge(b1[1], LEFT), FadeIn(b1[3]), run_time=1.0); self.wait(1.6)
        self.play(FadeOut(VGroup(b1, b2)), run_time=0.6)

        # ---------- 2. reenvíos ----------
        cap = self.swap(cap, self.caption("Reenvíos: el mismo hecho, dos veces", "otro id y otra hora de llegada, mismo contenido"))
        c1 = card(["id 100012", "hora equipo 6:04:33", "40,4000  −3,7000  · 0 km/h"]).move_to([-2.8, 0.2, 0])
        c2 = card(["id 100019", "hora equipo 6:04:33", "40,4000  −3,7000  · 0 km/h"], WARN).move_to([2.8, 0.2, 0])
        self.mark("dup"); self.play(FadeIn(c1, shift=0.3 * RIGHT), run_time=0.6); self.play(FadeIn(c2, shift=0.3 * LEFT), run_time=0.6)
        eq = Text("=", font=FONT, weight=BOLD, color=WARN).scale(1.0).move_to([0, 0.2, 0])
        x = Cross(c2, stroke_color=WARN, stroke_width=5)
        k = Text("se compara el contenido, no el id", font=FONT, color=GOOD).scale(0.42).move_to([0, -1.8, 0])
        self.mark("dupx"); self.play(FadeIn(eq), run_time=0.4); self.play(Create(x), c2.animate.set_opacity(0.35), FadeIn(k), run_time=0.8); self.wait(1.6)
        self.play(FadeOut(VGroup(c1, c2, eq, x, k)), run_time=0.6)

        # ---------- 3. desorden ----------
        cap = self.swap(cap, self.caption("El orden de llegada no es el de los hechos", "tras media hora sin cobertura, los mensajes guardados llegan mezclados"))
        P = [[-5.5 + i * 1.0, -1.0 + 0.25 * np.sin(i * 0.8), 0] for i in range(12)]
        arr = [0, 1, 2, 3, 8, 4, 9, 5, 10, 6, 11, 7]
        dots = VGroup(*[Dot(p, radius=0.09, color=MEAN) for p in P])
        zig = VMobject(stroke_color=WARN, stroke_width=4).set_points_as_corners([P[i] for i in arr])
        lab = Text("unidos en orden de llegada", font=FONT, color=WARN).scale(0.4).move_to([0, 1.2, 0])
        self.mark("zig"); self.play(FadeIn(dots), run_time=0.5); self.play(Create(zig), FadeIn(lab), run_time=1.6); self.wait(0.8)
        line = VMobject(stroke_color=MEAN, stroke_width=5).set_points_as_corners(P)
        lab2 = Text("ordenados por la hora del equipo", font=FONT, color=GOOD).scale(0.42).move_to([0, 1.2, 0])
        self.mark("sort"); self.play(ReplacementTransform(zig, line), ReplacementTransform(lab, lab2), run_time=1.2); self.wait(1.6)
        self.play(FadeOut(VGroup(dots, line, lab2)), run_time=0.6)

        # ---------- 4. reloj ----------
        cap = self.swap(cap, self.caption("Un reloj que se adelanta 40 segundos cada hora", "a las 11:00 ya va 3 minutos y 20 segundos por delante"))
        tx = lambda m: -5.0 + (m - 55) / 25 * 10.0          # minutos desde las 10:00
        ax = Line([-5.2, -1.6, 0], [5.2, -1.6, 0], color=LINE, stroke_width=2)
        tk = VGroup(*[Text(f"10:{m}" if m < 60 else f"11:{m-60:02d}", font=FONT, color=MUTED).scale(0.32).move_to([tx(m), -1.95, 0]) for m in (55, 60, 65, 70, 75, 80)])
        stop = Rectangle(width=tx(78) - tx(63.33), height=0.6, stroke_width=0, fill_color=MEAN, fill_opacity=0.8).move_to([(tx(63.33) + tx(78)) / 2, -0.6, 0])
        sl = Text("camión parado según el equipo", font=FONT, color=INK).scale(0.34).move_to(stop)
        tick = VGroup(Line([tx(61), -1.5, 0], [tx(61), 0.6, 0], color=POINT, stroke_width=5), Text("ticket 11:01", font=FONT, weight=BOLD, color=POINT).scale(0.4).move_to([tx(61), 0.95, 0]))
        alert = Text("alerta: repostaje sin el camión", font=FONT, weight=BOLD, color=WARN).scale(0.42).move_to([0, 1.8, 0])
        self.mark("clock"); self.play(Create(ax), FadeIn(tk), FadeIn(stop), FadeIn(sl), run_time=0.8); self.play(FadeIn(tick), run_time=0.5)
        self.play(FadeIn(alert), run_time=0.5); self.wait(1.0)
        ok = Text("hora corregida: la parada empieza a las 11:00", font=FONT, weight=BOLD, color=GOOD).scale(0.42).move_to([0, 1.8, 0])
        sl2 = Text("camión parado (hora corregida)", font=FONT, color=INK).scale(0.34).move_to(stop.get_center() + LEFT * (tx(63.33) - tx(60)))
        self.mark("fix"); self.play(stop.animate.shift(LEFT * (tx(63.33) - tx(60))), ReplacementTransform(sl, sl2), ReplacementTransform(alert, ok), run_time=1.2); self.wait(1.6)
        self.play(FadeOut(VGroup(ax, tk, stop, sl2, tick, ok)), run_time=0.6)

        # ---------- 5. saltos ----------
        cap = self.swap(cap, self.caption("Posiciones imposibles", "para llegar a ese punto y volver en 30 segundos haría falta ir a 600 km/h"))
        Q = [[-5.5 + i * 1.0, -1.2, 0] for i in range(12)]; Q[6] = [0.5, 1.6, 0]
        rd = VMobject(stroke_color=MEAN, stroke_width=4).set_points_as_corners(Q)
        qd = VGroup(*[Dot(q, radius=0.08, color=WARN if i == 6 else MEAN) for i, q in enumerate(Q)])
        vl = Text("600 km/h", font=FONT, weight=BOLD, color=WARN).scale(0.42).move_to([-1.3, 0.4, 0]); vr = Text("509 km/h", font=FONT, weight=BOLD, color=WARN).scale(0.42).move_to([2.3, 0.4, 0])
        self.mark("jump"); self.play(Create(rd), FadeIn(qd), run_time=1.0); self.play(FadeIn(vl), FadeIn(vr), run_time=0.6); self.wait(0.8)
        Q2 = Q[:6] + Q[7:]; rd2 = VMobject(stroke_color=MEAN, stroke_width=4).set_points_as_corners(Q2)
        self.mark("unjump"); self.play(ReplacementTransform(rd, rd2), qd[6].animate.set_opacity(0.25), FadeOut(vl), FadeOut(vr), run_time=1.0); self.wait(1.4)
        self.play(FadeOut(VGroup(rd2, qd)), run_time=0.6)

        # ---------- 6. sensor congelado ----------
        cap = self.swap(cap, self.caption("Un sensor congelado", "el combustible no cambia en dos horas mientras el camión recorre 163 km"))
        fx = lambda h: -5.5 + (h - 6) / 10 * 11.0; fy = lambda p: -2.6 + (p - 40) / 60 * 4.0
        def lvl(h):
            if h < 11.1: return 70 - 0.05 * 52 * (h - 6)
            if h < 13.07: return 70 - 0.05 * 52 * (h - 6) + 38
            if h < 15.08: return 70 - 0.05 * 52 * 7.07 + 38
            return 70 - 0.05 * 52 * (h - 6) + 38
        fl = VMobject(stroke_color=MEAN, stroke_width=4).set_points_as_corners([[fx(h), fy(lvl(h)), 0] for h in np.arange(6, 16.01, 0.05)])
        band = Rectangle(width=fx(15.08) - fx(13.07), height=4.2, stroke_width=0, fill_color=WARN, fill_opacity=0.18).move_to([(fx(13.07) + fx(15.08)) / 2, -0.5, 0])
        bl = Text("163 km sin que baje", font=FONT, weight=BOLD, color=WARN).scale(0.4).move_to([(fx(13.07) + fx(15.08)) / 2, 1.85, 0])
        self.mark("fuel"); self.play(Create(fl), run_time=1.4); self.mark("frozen"); self.play(FadeIn(band), FadeIn(bl), run_time=0.7); self.wait(1.8)
        self.play(*[FadeOut(m) for m in self.mobjects if m is not cap], run_time=0.6)

        # ---------- Cierre ----------
        self.mark("legend")
        self.play(FadeOut(cap), run_time=0.6)
        e1 = Text("Antes de calcular, deshaz los errores del registro.", font=FONT, weight=BOLD, color=INK).scale(0.62)
        e2 = Text("Deduplica, ordena, corrige el reloj y marca lo imposible.", font=FONT, color=MUTED).scale(0.46)
        VGroup(e1, e2).arrange(DOWN, buff=0.4)
        self.mark("end_text"); self.play(Write(e1), run_time=1.4); self.play(FadeIn(e2, shift=0.2 * UP), run_time=0.8)
        self.wait(2.0)
        self.mark("fade"); self.play(FadeOut(VGroup(e1, e2)), run_time=1)
        self.marks["total"] = [round(self.renderer.time, 3)]
        json.dump(self.marks, open("marks.json", "w"), indent=1)
