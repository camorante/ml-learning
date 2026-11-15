import json
import numpy as np
from manim import *

# Paleta oscura de la serie
BG = "#0E131B"; INK = "#E4EAF3"; MUTED = "#9BA7B9"; LINE = "#2B3545"
MEAN = "#6EA6FF"; SAMPLE = "#AE9FF3"; POINT = "#FF9A55"; GOOD = "#4CC38A"; WARN = "#F08A5D"
FONT = "Inter"
config.background_color = BG
D = json.load(open("../rn/r7/embed.json", encoding="utf-8"))


def T(s, sc=0.45, col=INK, w=None):
    return Text(s, font=FONT, color=col, weight=w or NORMAL).scale(sc)


def fmt(v, d=1):
    s = f"{abs(v):.{d}f}".replace(".", ",")
    if d == 0 and abs(v) >= 1000:
        s = f"{abs(v):,.0f}".replace(",", " ")
    return ("−" if v < 0 and round(abs(v), d) != 0 else "") + s


class Panel:
    """Ejes 2D sencillos: mapea (x, y) de datos a la escena dentro de una caja. logy para escala logarítmica."""

    def __init__(self, x0, x1, y0, y1, box, xt=(), xf=None, xl=None, yt=(), yf=None, yl=None, logy=False):
        self.logy = logy
        self.x0, self.x1 = x0, x1
        self.y0, self.y1 = (np.log10(y0), np.log10(y1)) if logy else (y0, y1)
        self.L, self.R, self.B, self.Tp = box
        g = VGroup(Line([self.L, self.B, 0], [self.R, self.B, 0], stroke_color=MUTED, stroke_width=2))
        for v in xt:
            g.add(Line(self.P(v, y0), self.P(v, y0) + 0.1 * UP, stroke_color=MUTED, stroke_width=2))
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
        yy = np.log10(y) if self.logy else y
        return np.array([self.L + (x - self.x0) / (self.x1 - self.x0) * (self.R - self.L),
                         self.B + (yy - self.y0) / (self.y1 - self.y0) * (self.Tp - self.B), 0])

    def poly(self, pts, col, w=4):
        lo, hi = (10 ** self.y0, 10 ** self.y1) if self.logy else (self.y0, self.y1)
        q = [self.P(min(self.x1, max(self.x0, x)), min(hi, max(lo, y))) for x, y in pts]
        return VMobject(stroke_color=col, stroke_width=w).set_points_as_corners(q)


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

    def consumo(self, box=(-6.0, 6.0, -2.9, 1.7)):
        Pc = Panel(15, 90, 20, 44, box, xt=(20, 40, 60, 80), xl="velocidad media (km/h)", yt=(20, 30, 40), yl="consumo (L/100 km)")
        verdad = DashedVMobject(Pc.poly(list(zip(D["g"], D["f0"])), MUTED, 3), num_dashes=40)
        return Pc, verdad

    def puntos(self, Pc, v, y, col, r=0.06, hollow=False):
        g = VGroup()
        for a, b in zip(v, y):
            d = Circle(radius=r, color=col, stroke_width=2.5) if hollow else Dot(radius=r, color=col)
            g.add(d.move_to(Pc.P(a, b)))
        return g

    def construct(self):
        self.marks = {}
        G = D["g"]
        title = Text("Sobreajuste y regularización", font=FONT, weight=BOLD, color=INK).scale(0.9)
        sub = Text("memorizar el ruido y cómo frenarlo", font=FONT, color=MUTED).scale(0.46)
        VGroup(title, sub).arrange(DOWN, buff=0.35)
        self.mark("title"); self.play(Write(title), run_time=1.4)
        self.mark("subtitle"); self.play(FadeIn(sub, shift=0.2 * UP)); self.wait(1.4)
        self.mark("title_out"); self.play(FadeOut(title), FadeOut(sub))

        # ---------- 1. memorizar ----------
        cap = self.swap(None, self.caption("Memorizar no es aprender", "20 viajes medidos · una red de 4 353 números"))
        toy = D["toy"]["0|0"]
        Pc, ver = self.consumo((-6.4, -0.6, -2.5, 1.7))
        pt = self.puntos(Pc, D["vt"], D["yt"], MEAN, 0.07)
        self.play(FadeIn(Pc.g), Create(ver), FadeIn(pt), run_time=1.0)
        Pe = Panel(0, 3000, 0, 5, (0.9, 6.4, -2.5, 1.7), xt=(0, 1500, 3000), xf=lambda v: fmt(v, 0), xl="época", yt=(0, 2, 4), yl="error (L/100 km)")
        self.play(FadeIn(Pe.g), run_time=0.5)
        fit = Pc.poly(list(zip(G, toy["ff"])), SAMPLE, 5)
        tr = Pe.poly([(10 * i, v) for i, v in enumerate(toy["tr"])], MEAN, 4); va = Pe.poly([(10 * i, v) for i, v in enumerate(toy["va"])], POINT, 4)
        self.mark("train"); self.play(Create(tr), Create(fit), run_time=2.2)
        self.mark("val"); self.play(Create(va), run_time=1.6)
        l1 = T(f"entrenamiento: {fmt(toy['tf'], 2)}", 0.34, MEAN, BOLD).move_to([3.65, -3.6, 0])
        l2 = T(f"validación: {fmt(toy['vf'], 2)}", 0.34, POINT, BOLD).move_to(Pe.P(1700, 4.75))
        l3 = T("la red pasa por cada viaje, ruido incluido", 0.34, SAMPLE, BOLD).move_to([-3.5, -3.6, 0])
        self.play(FadeIn(l1), FadeIn(l2), FadeIn(l3), run_time=0.6); self.wait(2.4)
        self.clear_but(cap)

        # ---------- 2. más datos ----------
        cap = self.swap(cap, self.caption("Más datos: el mejor freno", "la misma red sin regularizar, con 20 o con 300 viajes"))
        Pc, ver = self.consumo()
        self.play(FadeIn(Pc.g), Create(ver), run_time=0.7)
        p20 = self.puntos(Pc, D["vp"][:20], D["yp"][:20], MEAN, 0.07)
        f20 = Pc.poly(list(zip(G, D["cap"]["64|20"]["f"])), SAMPLE, 5)
        t20 = T(f"20 viajes: validación {fmt(D['cap']['64|20']['vf'], 2)}", 0.36, POINT, BOLD).move_to([0, -3.6, 0])
        self.mark("n20"); self.play(FadeIn(p20), Create(f20), FadeIn(t20), run_time=1.5); self.wait(0.8)
        p300 = self.puntos(Pc, D["vp"][20:300], D["yp"][20:300], MEAN, 0.04)
        f300 = Pc.poly(list(zip(G, D["cap"]["64|300"]["f"])), SAMPLE, 5)
        t300 = T(f"300 viajes: validación {fmt(D['cap']['64|300']['vf'], 2)}", 0.36, GOOD, BOLD).move_to([0, -3.6, 0])
        self.mark("n300"); self.play(FadeIn(p300, lag_ratio=0.01), Transform(f20, f300), FadeOut(t20), FadeIn(t300), run_time=2.0); self.wait(2.0)
        self.clear_but(cap)

        # ---------- 3. parar a tiempo ----------
        cap = self.swap(cap, self.caption("Parar a tiempo", "guardar los pesos de la mejor época de validación"))
        Pe = Panel(0, 3000, 0, 5, (-6.0, 6.0, -2.8, 1.7), xt=(0, 1000, 2000, 3000), xf=lambda v: fmt(v, 0), xl="época", yt=(0, 1, 2, 3, 4, 5), yf=lambda v: fmt(v, 0), yl="error (L/100 km)")
        self.play(FadeIn(Pe.g), run_time=0.5)
        tr = Pe.poly([(10 * i, v) for i, v in enumerate(toy["tr"])], MEAN, 4); va = Pe.poly([(10 * i, v) for i, v in enumerate(toy["va"])], POINT, 4)
        self.play(Create(tr), Create(va), run_time=1.8)
        vl = DashedLine(Pe.P(toy["mejor"], 0), Pe.P(toy["mejor"], 5), color=GOOD, stroke_width=3)
        lm = T(f"mejor época: {toy['mejor']}, validación {fmt(toy['vm'], 2)}", 0.36, GOOD, BOLD).move_to(Pe.P(1500, 1.6))
        self.mark("best"); self.play(Create(vl), FadeIn(lm), run_time=0.8)
        lf = T(f"seguir hasta 3 000: {fmt(toy['vf'], 2)}", 0.36, POINT, BOLD).move_to(Pe.P(2250, 3.7))
        self.play(FadeIn(lf), run_time=0.5); self.wait(2.4)
        self.clear_but(cap)

        # ---------- 4. pesos pequeños ----------
        cap = self.swap(cap, self.caption("Pesos pequeños, curvas suaves", "weight decay: cada paso encoge todos los pesos un poco"))
        Pc, ver = self.consumo()
        pt = self.puntos(Pc, D["vt"], D["yt"], MEAN, 0.07)
        f0 = Pc.poly(list(zip(G, D["toy"]["0|0"]["ff"])), SAMPLE, 5)
        t0 = T(f"sin freno · suma de pesos² {fmt(D['pesos']['0']['s2'], 1)} · validación {fmt(D['toy']['0|0']['vf'], 2)}", 0.34, MUTED, BOLD).move_to([0, -3.6, 0])
        self.play(FadeIn(Pc.g), Create(ver), FadeIn(pt), Create(f0), FadeIn(t0), run_time=1.2); self.wait(0.6)
        f1 = Pc.poly(list(zip(G, D["toy"]["1|0"]["ff"])), SAMPLE, 5)
        t1 = T(f"λ = 1 · suma de pesos² {fmt(D['pesos']['1']['s2'], 1)} · validación {fmt(D['toy']['1|0']['vf'], 2)}", 0.34, GOOD, BOLD).move_to([0, -3.6, 0])
        self.mark("decay"); self.play(Transform(f0, f1), FadeOut(t0), FadeIn(t1), run_time=2.0); self.wait(2.2)
        self.clear_but(cap)

        # ---------- 5. dropout ----------
        cap = self.swap(cap, self.caption("Apagar neuronas al azar", "dropout del 30 %: cada paso entrena una red adelgazada distinta"))
        A = D["apag"]
        grid = VGroup()
        for c in range(2):
            for i in range(64):
                grid.add(Dot(radius=0.085, color=SAMPLE).move_to([-5.6 + c * 1.95 + (i % 8) * 0.22, 0.75 - (i // 8) * 0.22, 0]))
        cl = VGroup(T("capa 1", 0.3, MUTED).move_to([-4.83, 1.1, 0]), T("capa 2", 0.3, MUTED).move_to([-2.88, 1.1, 0]))
        Pc, ver = self.consumo((-1.2, 6.2, -2.9, 1.7))
        self.play(FadeIn(grid), FadeIn(cl), FadeIn(Pc.g), Create(ver), run_time=1.0)
        thin = VGroup()
        for k in range(6):
            m = A["m"][k][0] + A["m"][k][1]
            anim = [grid[i].animate.set_color(SAMPLE if m[i] else LINE) for i in range(128)]
            cur = Pc.poly(list(zip(G, A["f"][k])), POINT, 3)
            self.mark("drop")
            self.play(*anim, Create(cur), run_time=0.6)
            self.play(cur.animate.set_stroke(MUTED, 2, 0.55), run_time=0.25)
            thin.add(cur)
        fe = Pc.poly(list(zip(G, A["fe"])), GOOD, 6)
        te = T(f"todas encendidas: va por el medio (85 km/h: {fmt(A['fe'][56], 2)})", 0.34, GOOD, BOLD).move_to([0, -3.6, 0])
        self.mark("ensemble"); self.play(*[grid[i].animate.set_color(SAMPLE) for i in range(128)], Create(fe), FadeIn(te), run_time=1.2); self.wait(2.0)
        self.clear_but(cap)

        # ---------- 6. inicialización ----------
        cap = self.swap(cap, self.caption("Empezar con los pesos justos", "la señal a través de 20 capas ReLU recién creadas"))
        I = D["ini"]
        Pi = Panel(0.5, 20.5, 1e-8, 1e8, (-5.4, 6.0, -2.8, 1.7), xt=(1, 5, 10, 15, 20), xl="capa", yt=(1e-8, 1e-4, 1, 1e4, 1e8),
                   yf=lambda v: {1e-8: "10⁻⁸", 1e-4: "10⁻⁴", 1: "1", 1e4: "10⁴", 1e8: "10⁸"}[v], yl="tamaño de la señal", logy=True)
        self.play(FadeIn(Pi.g), run_time=0.5)
        seq = [("relu|xavier|0", MUTED, "Xavier: se apaga", (16.5, 0.00015)), ("relu|grandes|0", WARN, "grandes: explota", (12.5, 3e6)),
               ("relu|he|0", POINT, "He: aguanta las 20 capas", (14.5, 12)), ("relu|grandes|1", GOOD, "normalizando cada capa: siempre ≈ 0,6", (6.5, 40))]
        for k, col, txt, pos in seq:
            ln = Pi.poly([(i + 1, v) for i, v in enumerate(I[k])], col, 4)
            lb = T(txt, 0.34, col, BOLD).move_to(Pi.P(*pos))
            self.mark("init"); self.play(Create(ln), FadeIn(lb), run_time=1.1); self.wait(0.5)
        self.wait(1.4)
        self.clear_but(cap)

        # ---------- Cierre ----------
        self.mark("legend")
        self.play(FadeOut(cap), run_time=0.6)
        e1 = Text("Sobreajuste y regularización", font=FONT, weight=BOLD, color=INK).scale(0.62)
        e2 = VGroup(T("separa validación y prueba; mira la prueba una vez", 0.42, MUTED),
                    T("más datos antes que más frenos", 0.42, MUTED),
                    T("guarda la mejor época de validación", 0.42, MUTED),
                    T("weight decay y dropout, de menos a más", 0.42, MUTED),
                    T("Xavier o He, y modo evaluación al usar la red", 0.42, MUTED)).arrange(DOWN, buff=0.18)
        VGroup(e1, e2).arrange(DOWN, buff=0.45)
        self.mark("end_text"); self.play(Write(e1), run_time=1.0); self.play(FadeIn(e2, shift=0.2 * UP, lag_ratio=0.15), run_time=1.0)
        self.wait(3.5)
        self.mark("fade"); self.play(FadeOut(VGroup(e1, e2)), run_time=1)
        self.marks["total"] = [round(self.renderer.time, 3)]
        json.dump(self.marks, open("marks.json", "w"), indent=1)
