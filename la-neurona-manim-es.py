import json
import math
import numpy as np
from manim import *

# Paleta oscura de la serie
BG = "#0E131B"; INK = "#E4EAF3"; MUTED = "#9BA7B9"; LINE = "#2B3545"
MEAN = "#6EA6FF"; SAMPLE = "#AE9FF3"; POINT = "#FF9A55"; GOOD = "#4CC38A"; WARN = "#F08A5D"
FONT = "Inter"
config.background_color = BG
D = json.load(open("../rn/r1/embed.json", encoding="utf-8"))
J = D["J"]


def T(s, sc=0.45, col=INK, w=None):
    return Text(s, font=FONT, color=col, weight=w or NORMAL).scale(sc)


def fmt(v, d=1):
    s = f"{abs(v):.{d}f}".replace(".", ",")
    return ("−" if v < 0 and round(abs(v), d) != 0 else "") + s


def perceptron_pasos(w):
    """Las correcciones del perceptrón desde w (misma regla que la página)."""
    w = list(w); pasos = []
    for pasada in range(1, 100):
        f = 0
        for i, (a, c, t) in enumerate(J):
            s = 1 if t else -1
            if s * (w[0] * a + w[1] * c + w[2]) <= 0:
                w = [w[0] + 0.1 * s * a, w[1] + 0.1 * s * c, w[2] + 0.1 * s]; f += 1
                pasos.append((pasada, i, list(w)))
        if f == 0:
            return pasos, pasada


class Panel:
    """Ejes 2D sencillos: mapea (x, y) de datos a la escena dentro de una caja."""

    def __init__(self, x0, x1, y0, y1, box, xt=(), xf=None, xl=None, yt=(), yf=None, yl=None):
        self.x0, self.x1, self.y0, self.y1 = x0, x1, y0, y1
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
        return np.array([self.L + (x - self.x0) / (self.x1 - self.x0) * (self.R - self.L),
                         self.B + (y - self.y0) / (self.y1 - self.y0) * (self.Tp - self.B), 0])

    def curve(self, f, a, b, col, w=4, n=240):
        return VMobject(stroke_color=col, stroke_width=w).set_points_as_corners([self.P(x, f(x)) for x in np.linspace(a, b, n)])

    def frontera(self, w, col=MEAN, sw=5, dash=False):
        """Segmento de w1·x + w2·y + b = 0 dentro del rectángulo del panel."""
        f = lambda p: w[0] * p[0] + w[1] * p[1] + w[2]
        C = [(self.x0, self.y0), (self.x1, self.y0), (self.x1, self.y1), (self.x0, self.y1)]
        pts = []
        for k in range(4):
            A, B = C[k], C[(k + 1) % 4]; fa, fb = f(A), f(B)
            if fa * fb < 0:
                t = fa / (fa - fb); pts.append((A[0] + t * (B[0] - A[0]), A[1] + t * (B[1] - A[1])))
        if len(pts) < 2:
            pts = [(self.x0, self.y0), (self.x0, self.y0)]
        cls = DashedLine if dash else Line
        return cls(self.P(*pts[0]), self.P(*pts[-1]), color=col, stroke_width=sw)


def box(title, value, col=MEAN, w=2.0, h=1.15):
    r = RoundedRectangle(corner_radius=0.15, width=w, height=h, stroke_color=MUTED, stroke_width=2, fill_color=BG, fill_opacity=1)
    t = T(title, 0.3, INK).move_to(r.get_top() + 0.27 * DOWN)
    v = T(value, 0.48, col, BOLD).move_to(r.get_center() + 0.08 * DOWN)
    return VGroup(r, t, v)


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

    def dots(self, Pn, r=0.1):
        return VGroup(*[Dot(Pn.P(a, c), radius=r, color=POINT if t else SAMPLE) for a, c, t in J])

    def rings(self, Pn, w):
        g = VGroup()
        for a, c, t in J:
            if ((w[0] * a + w[1] * c + w[2]) > 0) != (t == 1):
                g.add(Circle(radius=0.19, color=WARN, stroke_width=4).move_to(Pn.P(a, c)))
        return g

    def construct(self):
        self.marks = {}
        title = Text("La neurona", font=FONT, weight=BOLD, color=INK).scale(1.0)
        sub = Text("una suma con pesos que aprende a separar dos grupos", font=FONT, color=MUTED).scale(0.46)
        VGroup(title, sub).arrange(DOWN, buff=0.35)
        self.mark("title"); self.play(Write(title), run_time=1.3)
        self.mark("subtitle"); self.play(FadeIn(sub, shift=0.2 * UP)); self.wait(1.5)
        self.mark("title_out"); self.play(FadeOut(title), FadeOut(sub))

        # ---------- 1. suma con pesos ----------
        cap = self.swap(None, self.caption("Una suma con pesos que decide", "una jornada: 4,1 frenadas cada 100 km y 2,2 h al ralentí"))
        ys = [1.25, -0.15, -1.55]; xin = -5.3
        ins = VGroup(*[VGroup(Circle(radius=0.48, color=INK if k < 2 else MUTED, stroke_width=3),
                              T(v, 0.4, INK, BOLD)).move_to([xin, yy, 0]) for k, (v, yy) in enumerate(zip(["4,1", "2,2 h", "1"], ys))])
        for k, s in enumerate(["frenadas", "ralentí", "fijo"]):
            ins[k].add(T(s, 0.28, MUTED).next_to(ins[k][0], UP, buff=0.08))
        sx, sy = -1.3, -0.15
        sig = VGroup(Circle(radius=0.68, color=MEAN, stroke_width=4).move_to([sx, sy, 0]), T("Σ", 0.55, MEAN, BOLD).move_to([sx, sy + 0.24, 0]))
        lines = VGroup(*[Line([xin + 0.5, yy, 0], [sx - 0.66, sy + (yy - sy) * 0.25, 0], color=MUTED, stroke_width=3) for yy in ys])
        at = lambda k, t: lines[k].get_start() + t * (lines[k].get_end() - lines[k].get_start())
        wl = VGroup(*[T(s, 0.36, POINT, BOLD) for s in ["× 0,83", "× 1,28", "× −6"]])
        prods = VGroup(*[T(s, 0.36, INK) for s in ["= 3,40", "= 2,82", "= −6,00"]])
        for k in range(3):
            VGroup(wl[k], prods[k]).arrange(RIGHT, buff=0.15).move_to(at(k, 0.42) + 0.3 * UP)
        self.mark("inputs"); self.play(LaggedStart(*[FadeIn(i) for i in ins], lag_ratio=0.2), run_time=1.0)
        self.mark("weights"); self.play(LaggedStart(*[Create(l) for l in lines], lag_ratio=0.15), LaggedStart(*[FadeIn(w) for w in wl], lag_ratio=0.15), run_time=1.0)
        self.play(FadeIn(sig), LaggedStart(*[FadeIn(p) for p in prods], lag_ratio=0.2), run_time=1.0)
        zt = T("0,22", 0.42, INK, BOLD).move_to([sx, sy - 0.2, 0])
        self.mark("sum"); self.play(FadeIn(zt, scale=1.3), run_time=0.6)
        bxs = RoundedRectangle(corner_radius=0.12, width=1.3, height=1.1, stroke_color=INK, stroke_width=3).move_to([1.3, sy, 0])
        stp = VMobject(stroke_color=INK, stroke_width=4).set_points_as_corners([[0.85, sy - 0.32, 0], [1.3, sy - 0.32, 0], [1.3, sy + 0.32, 0], [1.75, sy + 0.32, 0]])
        a1 = Arrow([sx + 0.7, sy, 0], [0.62, sy, 0], buff=0, color=MUTED, stroke_width=4, max_tip_length_to_length_ratio=0.25)
        q = T("¿mayor que 0?", 0.28, MUTED).next_to(bxs, DOWN, buff=0.15)
        a2 = Arrow([1.98, sy, 0], [3.0, sy, 0], buff=0, color=MUTED, stroke_width=4, max_tip_length_to_length_ratio=0.25)
        out = VGroup(RoundedRectangle(corner_radius=0.3, width=1.9, height=0.62, stroke_color=POINT, stroke_width=3, fill_color=POINT, fill_opacity=0.18),
                     T("revisar", 0.42, POINT, BOLD)).move_to([4.05, sy, 0])
        self.play(GrowArrow(a1), FadeIn(bxs), Create(stp), FadeIn(q), run_time=0.8)
        self.mark("decide"); self.play(GrowArrow(a2), FadeIn(out, shift=0.2 * RIGHT), run_time=0.7)
        eq = T("0,83 × 4,1 + 1,28 × 2,2 − 6 = 0,22 > 0  →  revisar", 0.42, INK, BOLD).move_to([0, -2.9, 0])
        self.play(FadeIn(eq, shift=0.1 * UP), run_time=0.6); self.wait(3.4)
        self.clear_but(cap)

        # ---------- 2. la frontera es una recta ----------
        cap = self.swap(cap, self.caption("Su frontera es una recta", "los pesos la giran · el sesgo la desplaza"))
        Pp = Panel(0, 10, 0, 5, (-6.3, 1.3, -3.1, 1.9), xt=(0, 2, 4, 6, 8, 10), xl="frenadas cada 100 km", yt=(0, 1, 2, 3, 4, 5), yf=lambda v: f"{v} h", yl="horas al ralentí")
        dts = self.dots(Pp)
        self.play(FadeIn(Pp.g), LaggedStart(*[FadeIn(d, scale=0.5) for d in dts], lag_ratio=0.04), run_time=1.2)
        w = ValueTracker(0.2); bb = ValueTracker(-6.0)
        fr = always_redraw(lambda: Pp.frontera([1.0, w.get_value(), bb.get_value()]))
        rg = always_redraw(lambda: self.rings(Pp, [1.0, w.get_value(), bb.get_value()]))
        def nerr():
            return sum(1 for a, c, t in J if ((a + w.get_value() * c + bb.get_value()) > 0) != (t == 1))
        rd = always_redraw(lambda: VGroup(
            T("puntos = frenadas", 0.36, INK), T(f"+ {fmt(w.get_value(), 2)} × horas", 0.36, INK), T(f"{fmt(bb.get_value(), 1)} de partida", 0.36, INK),
            T(f"{nerr()} de 20 mal", 0.56, WARN if nerr() else GOOD, BOLD)).arrange(DOWN, aligned_edge=LEFT, buff=0.16).move_to([4.4, 0.2, 0]).align_to([2.3, 0, 0], LEFT))
        self.mark("line"); self.play(Create(fr), FadeIn(rg), FadeIn(rd), run_time=0.8); self.wait(1.2)
        self.mark("shift"); self.play(bb.animate.set_value(-5.0), run_time=1.6); self.play(bb.animate.set_value(-6.0), run_time=1.0)
        self.mark("rotate"); self.play(w.animate.set_value(2.0), run_time=2.4)
        self.play(bb.animate.set_value(-8.0), run_time=1.4); self.wait(2.4)
        self.clear_but(cap)

        # ---------- 3. el perceptrón ----------
        cap = self.swap(cap, self.caption("Aprender de los errores: el perceptrón", "si falla con una jornada: cada peso ± 0,1 × su dato"))
        Pp = Panel(0, 10, 0, 5, (-6.3, 1.3, -3.1, 1.9), xt=(0, 2, 4, 6, 8, 10), xl="frenadas cada 100 km", yt=(0, 1, 2, 3, 4, 5), yf=lambda v: f"{v} h", yl="horas al ralentí")
        dts = self.dots(Pp); W0 = [1.0, 0.2, -6.0]
        line = Pp.frontera(W0); rg = self.rings(Pp, W0)
        nm = lambda ww: sum(1 for a, c, t in J if ((ww[0] * a + ww[1] * c + ww[2]) > 0) != (t == 1))
        info = lambda k, pas, ww: VGroup(T(f"pasada {pas}", 0.36, MUTED), T(f"{k} correcciones", 0.5, INK, BOLD),
                                         T(f"{nm(ww)} de 20 mal", 0.5, WARN if nm(ww) else GOOD, BOLD),
                                         T(f"pesos {fmt(ww[0], 2)} · {fmt(ww[1], 2)}", 0.34, MUTED), T(f"sesgo {fmt(ww[2], 1)}", 0.34, MUTED)).arrange(DOWN, aligned_edge=LEFT, buff=0.14).move_to([4.4, 0.0, 0]).align_to([2.3, 0, 0], LEFT)
        inf = info(0, 1, W0)
        self.play(FadeIn(Pp.g), FadeIn(dts), Create(line), FadeIn(rg), FadeIn(inf), run_time=1.0); self.wait(0.6)
        pasos, npas = perceptron_pasos(W0)
        self.mark("learn")
        hl = None
        for k, (pas, i, ww) in enumerate(pasos, 1):
            ring = Circle(radius=0.27, color=INK, stroke_width=5).move_to(Pp.P(J[i][0], J[i][1]))
            nl, nr, ni = Pp.frontera(ww), self.rings(Pp, ww), info(k, pas, ww)
            rt = 0.55 if k <= 4 else 0.28
            anims = [Transform(line, nl), Transform(rg, nr), Transform(inf, ni)]
            anims.append(Transform(hl, ring) if hl is not None else FadeIn(ring))
            self.play(*anims, run_time=rt)
            if hl is None:
                hl = ring
        last = info(len(pasos), npas, pasos[-1][2])
        done = T(f"pasada {npas}: sin fallos, para", 0.36, GOOD, BOLD).next_to(last, DOWN, buff=0.3).align_to(last, LEFT)
        self.mark("converged"); self.play(FadeOut(hl), Transform(inf, last), FadeIn(done), run_time=0.7); self.wait(3.0)
        self.clear_but(cap)

        # ---------- 4. sigmoide ----------
        cap = self.swap(cap, self.caption("De «sí o no» a una probabilidad", "la sigmoide: puntuación 0 → 50 %"))
        Ps = Panel(-8, 8, 0, 1, (-6.0, 1.4, -2.9, 1.9), xt=(-8, -4, 0, 4, 8), xf=lambda v: fmt(v, 0), xl="puntuación", yt=(0, 0.5, 1), yf=lambda v: f"{round(v * 100)} %")
        stepc = VMobject(stroke_color=MUTED, stroke_width=4).set_points_as_corners([Ps.P(-8, 0), Ps.P(0, 0), Ps.P(0, 1), Ps.P(8, 1)])
        stepc = DashedVMobject(stepc, num_dashes=40)
        sg = lambda z: 1 / (1 + math.exp(-z))
        sc = Ps.curve(sg, -8, 8, MEAN, 5)
        lab1 = T("escalón: 0 % o 100 %", 0.36, MUTED).move_to([4.4, 1.6, 0]).align_to([2.3, 0, 0], LEFT)
        self.play(FadeIn(Ps.g), Create(stepc), FadeIn(lab1), run_time=1.0)
        lab2 = T("sigmoide: un porcentaje", 0.36, MEAN, BOLD).next_to(lab1, DOWN, buff=0.25).align_to(lab1, LEFT)
        self.mark("sigmoid"); self.play(Create(sc), FadeIn(lab2), run_time=1.3)
        Wt = D["WT"]; Z = [Wt[0] * a + Wt[1] * c + Wt[2] for a, c, t in J]
        sd = VGroup(*[Dot(Ps.P(z, sg(z)), radius=0.09, color=POINT if t else SAMPLE) for z, (a, c, t) in zip(Z, J)])
        self.play(LaggedStart(*[FadeIn(d, scale=0.5) for d in sd], lag_ratio=0.04), run_time=1.0)
        z18 = Z[17]; r18 = Circle(radius=0.2, color=INK, stroke_width=4).move_to(Ps.P(z18, sg(z18)))
        t18 = VGroup(T("jornada 18: puntuación −0,01", 0.34, INK), T("escalón: «bien»", 0.34, MUTED), T("sigmoide: 49,8 %", 0.46, SAMPLE, BOLD)).arrange(DOWN, aligned_edge=LEFT, buff=0.12).next_to(lab2, DOWN, buff=0.45).align_to(lab1, LEFT)
        self.mark("doubt"); self.play(Create(r18), FadeIn(t18), run_time=0.7); self.wait(1.2)
        zn = 0.219; rn = Circle(radius=0.2, color=POINT, stroke_width=4).move_to(Ps.P(zn, sg(zn)))
        tn = VGroup(T("jornada nueva: 0,22", 0.34, INK), T("revisar, con un 55 %", 0.46, POINT, BOLD)).arrange(DOWN, aligned_edge=LEFT, buff=0.12).next_to(t18, DOWN, buff=0.35).align_to(lab1, LEFT)
        self.mark("newday"); self.play(Create(rn), FadeIn(tn), run_time=0.7)
        lg = VGroup(T("aprender con probabilidades:", 0.32, MEAN), T("regresión logística", 0.4, MEAN, BOLD)).arrange(DOWN, aligned_edge=LEFT, buff=0.1).next_to(tn, DOWN, buff=0.4).align_to(lab1, LEFT)
        self.mark("logistic"); self.play(FadeIn(lg), run_time=0.5); self.wait(3.0)
        self.clear_but(cap)

        # ---------- 5. la franja ----------
        cap = self.swap(cap, self.caption("Lo que una recta no puede separar", "consumo que no cuadra con la carga: por encima o por debajo"))
        Pb = Panel(0, 25, 18, 42, (-6.0, 1.4, -3.1, 1.9), xt=(0, 5, 10, 15, 20, 25), xf=lambda v: f"{v} t", xl="carga", yt=(20, 30, 40), yl="consumo (L/100 km)")
        C, Q, Y = D["cc"], D["cq"], D["cy"]
        bd = VGroup(*[Dot(Pb.P(c, q), radius=0.1, color=POINT if t else SAMPLE) for c, q, t in zip(C, Q, Y)])
        self.play(FadeIn(Pb.g), LaggedStart(*[FadeIn(d, scale=0.5) for d in bd], lag_ratio=0.03), run_time=1.2)
        m, a = 0.3, 30.8
        l1 = Line(Pb.P(0, a), Pb.P(25, a + m * 25), color=MEAN, stroke_width=5)
        er = VGroup(*[Circle(radius=0.19, color=WARN, stroke_width=4).move_to(Pb.P(c, q)) for c, q, t in zip(C, Q, Y) if ((q - (a + m * c)) > 0) != (t == 1)])
        t1 = VGroup(T("la mejor recta posible", 0.36, INK), T(f"{len(er)} de 30 mal", 0.56, WARN, BOLD), T("los raros de abajo se escapan", 0.34, MUTED)).arrange(DOWN, aligned_edge=LEFT, buff=0.14).move_to([4.4, 0.7, 0]).align_to([2.3, 0, 0], LEFT)
        self.mark("oneline"); self.play(Create(l1), FadeIn(er), FadeIn(t1), run_time=1.0); self.wait(1.6)
        l2 = VGroup(Line(Pb.P(0, 26.5), Pb.P(25, 26.5 + 0.55 * 25), color=MEAN, stroke_width=5), Line(Pb.P(0, 21.5), Pb.P(25, 21.5 + 0.55 * 25), color=MEAN, stroke_width=5))
        t2 = VGroup(T("dos rectas: 0 de 30 mal", 0.46, GOOD, BOLD), T("= varias neuronas juntas", 0.34, MUTED), T("tema 3: la red multicapa", 0.36, INK, BOLD)).arrange(DOWN, aligned_edge=LEFT, buff=0.14).next_to(t1, DOWN, buff=0.5).align_to(t1, LEFT)
        self.mark("twolines"); self.play(ReplacementTransform(l1, l2), FadeOut(er), FadeIn(t2), run_time=1.0); self.wait(3.4)
        self.clear_but(cap)

        # ---------- Cierre ----------
        self.mark("legend")
        self.play(FadeOut(cap), run_time=0.6)
        e1 = Text("Una neurona: suma con pesos + sesgo + activación", font=FONT, weight=BOLD, color=INK).scale(0.62)
        e2 = VGroup(T("su frontera es una recta: los pesos la giran, el sesgo la desplaza", 0.42, MUTED),
                    T("el perceptrón la aprende corrigiéndose en cada fallo", 0.42, MUTED),
                    T("con sigmoide da probabilidades: es la regresión logística", 0.42, MUTED),
                    T("lo que no se separa con una recta necesita varias neuronas", 0.42, MUTED)).arrange(DOWN, buff=0.18)
        VGroup(e1, e2).arrange(DOWN, buff=0.45)
        self.mark("end_text"); self.play(Write(e1), run_time=1.2); self.play(FadeIn(e2, shift=0.2 * UP, lag_ratio=0.15), run_time=1.0)
        self.wait(4.0)
        self.mark("fade"); self.play(FadeOut(VGroup(e1, e2)), run_time=1)
        self.marks["total"] = [round(self.renderer.time, 3)]
        json.dump(self.marks, open("marks.json", "w"), indent=1)
