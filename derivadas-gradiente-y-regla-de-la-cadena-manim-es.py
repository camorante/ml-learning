import json
import math
import numpy as np
from manim import *

# Paleta oscura de la serie
BG = "#0E131B"; INK = "#E4EAF3"; MUTED = "#9BA7B9"; LINE = "#2B3545"
MEAN = "#6EA6FF"; SAMPLE = "#AE9FF3"; POINT = "#FF9A55"; GOOD = "#4CC38A"; WARN = "#F08A5D"
FONT = "Inter"
config.background_color = BG
D = json.load(open("../mt/m16/embed.json"))


def T(s, sc=0.45, col=INK, w=None):
    return Text(s, font=FONT, color=col, weight=w or NORMAL).scale(sc)


def fmt(v, d=1):
    s = f"{abs(v):.{d}f}".replace(".", ",")
    return ("−" if v < 0 and round(abs(v), d) != 0 else "") + s


# posición GPS del juguete (km) y velocidad (km/h), t en minutos
ESC = [(0, 3, 50), (8, 2, 40), (19, 1, -60), (23, 2, 50)]
S = lambda u: 0 if u <= 0 else 1 if u >= 1 else 3 * u * u - 2 * u ** 3
I = lambda u: 0 if u <= 0 else (u - 0.5 if u >= 1 else u ** 3 - u ** 4 / 2)
vel = lambda t: sum(A * S((t - t0) / d) for t0, d, A in ESC)
pos = lambda t: sum(A * d * I((t - t0) / d) for t0, d, A in ESC) / 60


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

    def construct(self):
        self.marks = {}
        title = Text("Derivadas, gradiente y regla de la cadena", font=FONT, weight=BOLD, color=INK).scale(0.8)
        sub = Text("lo mínimo de cálculo para entender cómo aprende un modelo", font=FONT, color=MUTED).scale(0.46)
        VGroup(title, sub).arrange(DOWN, buff=0.35)
        self.mark("title"); self.play(Write(title), run_time=1.4)
        self.mark("subtitle"); self.play(FadeIn(sub, shift=0.2 * UP)); self.wait(1.5)
        self.mark("title_out"); self.play(FadeOut(title), FadeOut(sub))

        # ---------- 1. la derivada: acercar dos lecturas ----------
        cap = self.swap(None, self.caption("La derivada: acercar dos lecturas del GPS", "velocidad media entre dos lecturas → velocidad en un instante"))
        Pp = Panel(0, 30, 0, 34, (-6.2, 1.0, -3.0, 1.9), xt=(0, 10, 20, 30), xl="minutos", yt=(0, 10, 20, 30), yl="km recorridos")
        curve = Pp.curve(pos, 0, 30, MEAN, 5, 300)
        self.mark("curve"); self.play(FadeIn(Pp.g), Create(curve), run_time=1.4)
        t0 = 15.0; k = ValueTracker(1.0); h = type("H", (), {"get_value": lambda self: 10 ** k.get_value()})()
        def sec():
            hv = h.get_value(); sl = (pos(t0 + hv) - pos(t0)) / hv            # km por minuto
            a = max(0.0, t0 - pos(t0) / sl); b = min(30.0, t0 + (33.5 - pos(t0)) / sl)                 # sin salirse por arriba
            return Line(Pp.P(a, pos(t0) + sl * (a - t0)), Pp.P(b, pos(t0) + sl * (b - t0)), color=SAMPLE, stroke_width=4)
        secl = always_redraw(sec)
        d1 = Dot(Pp.P(t0, pos(t0)), radius=0.1, color=POINT)
        d2 = always_redraw(lambda: Dot(Pp.P(t0 + h.get_value(), pos(t0 + h.get_value())), radius=0.1, color=POINT))
        rd = always_redraw(lambda: VGroup(
            T("entre lecturas: " + (fmt(h.get_value(), 1) + " min" if h.get_value() >= 1 else fmt(h.get_value() * 60, 1) + " s"), 0.36, MUTED),
            T("velocidad media", 0.36, SAMPLE),
            T(fmt((pos(t0 + h.get_value()) - pos(t0)) / h.get_value() * 60, 1) + " km/h", 0.62, SAMPLE, BOLD)).arrange(DOWN, aligned_edge=LEFT, buff=0.14).move_to([4.4, 0.9, 0]).align_to([2.6, 0, 0], LEFT))
        speedo = VGroup(T("velocímetro (la derivada)", 0.36, POINT), T("90,0 km/h", 0.62, POINT, BOLD)).arrange(DOWN, aligned_edge=LEFT, buff=0.14).move_to([4.4, -1.4, 0]).align_to([2.6, 0, 0], LEFT)
        self.mark("dots"); self.play(FadeIn(d1), FadeIn(d2), Create(secl), FadeIn(rd), run_time=0.9)
        self.play(FadeIn(speedo), run_time=0.5); self.wait(1.5)
        self.mark("shrink"); self.play(k.animate.set_value(-1.7), run_time=5.0, rate_func=linear)
        note = VGroup(T("la recta entre las lecturas", 0.36, INK), T("se vuelve la tangente", 0.36, INK)).arrange(DOWN, aligned_edge=LEFT, buff=0.1).move_to([4.4, -2.7, 0]).align_to([2.6, 0, 0], LEFT)
        self.mark("tangent"); self.play(FadeIn(note, shift=0.1 * UP), run_time=0.6); self.wait(1.8)
        self.clear_but(cap)

        # ---------- 2. el mínimo: derivada cero ----------
        cap = self.swap(cap, self.caption("El mínimo está donde la derivada vale cero", "coste cada 100 km = gasóleo + tiempo de conductor y camión"))
        c = lambda v: 1.6 * (20 + 0.0016 * v * v) + 3000 / v
        dc = lambda v: 0.00512 * v - 3000 / v ** 2
        Pc = Panel(50, 110, 80, 100, (-6.2, 1.2, -3.0, 1.9), xt=(50, 60, 70, 80, 90, 100, 110), xl="velocidad (km/h)", yt=(80, 85, 90, 95, 100), yl="€ cada 100 km")
        cc = Pc.curve(c, 50, 110, MEAN, 5)
        self.play(FadeIn(Pc.g), Create(cc), run_time=1.0)
        v = ValueTracker(60.0)
        tan = always_redraw(lambda: Line(Pc.P(v.get_value() - 9, c(v.get_value()) - 9 * dc(v.get_value())),
                                         Pc.P(v.get_value() + 9, c(v.get_value()) + 9 * dc(v.get_value())), color=INK, stroke_width=4))
        dv = always_redraw(lambda: Dot(Pc.P(v.get_value(), c(v.get_value())), radius=0.11, color=POINT))
        rv = always_redraw(lambda: VGroup(T(f"{fmt(v.get_value(), 1)} km/h", 0.5, POINT, BOLD), T("derivada", 0.36, MUTED),
                                          T(("+" if dc(v.get_value()) > 0.0005 else "") + fmt(dc(v.get_value()), 3) + " € por km/h", 0.46, INK, BOLD))
                           .arrange(DOWN, aligned_edge=LEFT, buff=0.14).move_to([4.4, 0.6, 0]).align_to([2.6, 0, 0], LEFT))
        self.mark("slide"); self.play(FadeIn(tan), FadeIn(dv), FadeIn(rv), run_time=0.5)
        self.play(v.animate.set_value(100), run_time=2.4, rate_func=smooth)
        self.play(v.animate.set_value(83.68), run_time=1.8, rate_func=smooth)
        vs = DashedLine(Pc.P(83.68, 80), Pc.P(83.68, 100), color=GOOD, stroke_width=3)
        best = VGroup(T("tangente plana: 83,7 km/h", 0.4, GOOD, BOLD), T("85,78 € cada 100 km", 0.4, GOOD, BOLD)).arrange(DOWN, aligned_edge=LEFT, buff=0.12).move_to([4.4, -1.6, 0]).align_to([2.6, 0, 0], LEFT)
        self.mark("flat"); self.play(Create(vs), FadeIn(best), run_time=0.8); self.wait(2.4)
        self.clear_but(cap)

        # ---------- 3. gradiente ----------
        cap = self.swap(cap, self.caption("Con dos números: el gradiente apunta cuesta arriba", "error de la recta del tema 4 según altura y pendiente"))
        x, y = np.array(D["x"]), np.array(D["y"]); bt = D["beta"]
        n = len(x); s11, s12, s22 = n, float(np.sum(x / 10)), float(np.sum((x / 10) ** 2))
        tr, det = s11 + s22, s11 * s22 - s12 * s12
        l1 = tr / 2 + math.sqrt(tr * tr / 4 - det); l2 = tr / 2 - math.sqrt(tr * tr / 4 - det); th = math.atan2(l1 - s11, s12)
        Pg = Panel(20, 32, 0, 8, (-5.6, 1.6, -3.1, 1.7), xt=(20, 24, 28, 32), xl="altura", yt=(0, 4, 8), yl="pendiente")
        ells = VGroup()
        for lev in (2, 6, 15, 30, 55, 90, 140):
            pts = []
            for j in range(121):
                t = 2 * math.pi * j / 120; u = math.sqrt(lev / l1) * math.cos(t); w = math.sqrt(lev / l2) * math.sin(t)
                pts.append(Pg.P(bt[0] + u * math.cos(th) - w * math.sin(th), bt[1] + u * math.sin(th) + w * math.cos(th)))
            ells.add(VMobject(stroke_color=MEAN, stroke_width=2, stroke_opacity=0.7).set_points_as_corners(pts))
        frame = Rectangle(width=Pg.R - Pg.L, height=Pg.Tp - Pg.B, stroke_color=LINE, stroke_width=2).move_to([(Pg.L + Pg.R) / 2, (Pg.B + Pg.Tp) / 2, 0])
        star = Star(5, outer_radius=0.16, color=INK, fill_color=INK, fill_opacity=1).move_to(Pg.P(*bt))
        cover = VGroup(Rectangle(width=4, height=8, fill_color=BG, fill_opacity=1, stroke_width=0).next_to(frame, RIGHT, buff=0),
                       Rectangle(width=4, height=8, fill_color=BG, fill_opacity=1, stroke_width=0).next_to(frame, LEFT, buff=0),
                       Rectangle(width=12, height=3, fill_color=BG, fill_opacity=1, stroke_width=0).next_to(frame, UP, buff=0),
                       Rectangle(width=12, height=3, fill_color=BG, fill_opacity=1, stroke_width=0).next_to(frame, DOWN, buff=0))
        cover.set_z_index(2); Pg.g.set_z_index(3); frame.set_z_index(3)
        self.play(FadeIn(ells), FadeIn(cover), FadeIn(frame), FadeIn(Pg.g), FadeIn(star), run_time=0.9)
        grad = lambda a, m: (-2 * float(np.sum(y - a - m * x / 10)), -2 * float(np.sum((y - a - m * x / 10) * x / 10)))
        arrows = VGroup()
        for a, m in ((23, 6), (30, 1.5), (22, 2.5), (29, 6.5)):
            ga, gm = grad(a, m); gn = math.hypot(ga, gm); L = min(1.6, 0.05 * gn)
            arrows.add(Arrow(Pg.P(a, m), Pg.P(a + ga / gn * L, m + gm / gn * L), buff=0, color=WARN, stroke_width=5, max_tip_length_to_length_ratio=0.3))
            arrows.add(Dot(Pg.P(a, m), radius=0.08, color=POINT))
        lg = VGroup(T("flecha roja: gradiente", 0.36, WARN, BOLD), T("hacia donde el error sube más", 0.34, MUTED),
                    T("perpendicular a las curvas de nivel", 0.34, MUTED)).arrange(DOWN, aligned_edge=LEFT, buff=0.12).move_to([4.5, 0.8, 0]).align_to([2.5, 0, 0], LEFT)
        lg.set_z_index(4)
        self.mark("arrows"); self.play(LaggedStart(*[GrowArrow(a) if isinstance(a, Arrow) else FadeIn(a) for a in arrows], lag_ratio=0.15), FadeIn(lg), run_time=1.6)
        self.wait(1.0)
        a, m = 23.0, 6.0; pts = [(a, m)]
        for _ in range(60):
            ga, gm = grad(a, m); a -= 0.03 * ga; m -= 0.03 * gm; pts.append((a, m))
        self.play(FadeOut(arrows[2:]), run_time=0.4)
        cur = Dot(Pg.P(*pts[0]), radius=0.1, color=POINT); self.add(cur)
        st = VGroup(T("pasos contra la flecha", 0.36, GOOD, BOLD), T("(descenso de gradiente)", 0.34, MUTED)).arrange(DOWN, aligned_edge=LEFT, buff=0.12).move_to([4.5, -1.4, 0]).align_to([2.5, 0, 0], LEFT)
        st.set_z_index(4)
        self.mark("steps"); self.play(FadeIn(st), run_time=0.4)
        path = VGroup()
        for k in range(1, 25):
            seg = Line(Pg.P(*pts[k - 1]), Pg.P(*pts[k]), color=POINT, stroke_width=4)
            self.play(Create(seg), cur.animate.move_to(Pg.P(*pts[k])), run_time=0.22 if k > 3 else 0.4)
            path.add(seg)
        self.play(cur.animate.move_to(Pg.P(*pts[-1])), run_time=0.5); self.wait(1.4)
        self.clear_but(cap)

        # ---------- 4. regla de la cadena ----------
        cap = self.swap(cap, self.caption("Regla de la cadena: los ritmos se multiplican", "a 80 km/h, con el gasóleo a 1,60 €/L"))
        b1 = box("velocidad", "80 km/h", w=2.6).move_to([-4.6, 0.6, 0])
        b2 = box("gasóleo", "24,19 L/h", w=2.6).move_to([0, 0.6, 0])
        b3 = box("gasto", "38,71 €/h", w=2.6).move_to([4.6, 0.6, 0])
        a1 = Arrow(b1.get_right(), b2.get_left(), buff=0.12, color=MUTED, stroke_width=4)
        a2 = Arrow(b2.get_right(), b3.get_left(), buff=0.12, color=MUTED, stroke_width=4)
        r1 = T("× 0,507", 0.46, POINT, BOLD).next_to(a1, UP, buff=0.12); r2 = T("× 1,60", 0.46, POINT, BOLD).next_to(a2, UP, buff=0.12)
        self.mark("boxes"); self.play(FadeIn(b1), run_time=0.4); self.play(GrowArrow(a1), FadeIn(r1), FadeIn(b2), run_time=0.7); self.play(GrowArrow(a2), FadeIn(r2), FadeIn(b3), run_time=0.7)
        u1 = T("L/h por km/h", 0.3, POINT).next_to(a1, DOWN, buff=0.12); u2 = T("€ por litro", 0.3, POINT).next_to(a2, DOWN, buff=0.12)
        self.play(FadeIn(u1), FadeIn(u2), run_time=0.4)
        eq = T("0,507 × 1,60 = 0,81 € más por hora por cada km/h", 0.5, INK, BOLD).move_to([0, -1.4, 0])
        eq2 = T("de verdad: +0,82 €/h  ·  exacta para cambios pequeños", 0.38, MEAN).next_to(eq, DOWN, buff=0.3)
        self.mark("product"); self.play(FadeIn(eq, shift=0.1 * UP), run_time=0.7); self.wait(0.9); self.play(FadeIn(eq2), run_time=0.5); self.wait(2.4)
        self.clear_but(cap)

        # ---------- 5. retropropagación ----------
        cap = self.swap(cap, self.caption("Retropropagación: la cadena, hacia atrás", "una neurona: 2 atascos → probabilidad de llegar tarde · llegó tarde"))
        Y0 = 0.75
        nodes = [("w · x", "u = 1,00"), ("u + b", "z = −0,500"), ("σ(z)", "p = 0,378"), ("−log p", "castigo 0,974")]
        xs = [-4.95, -1.65, 1.65, 4.95]
        bxs = VGroup(*[box(a, b, w=2.3, h=1.2).move_to([xx, Y0, 0]) for (a, b), xx in zip(nodes, xs)])
        inp = T("w = 0,5   x = 2   b = −1,5", 0.36, MUTED).next_to(bxs[0], UP, buff=0.3).align_to(bxs[0], LEFT)
        arr = VGroup(*[Arrow(bxs[i].get_right(), bxs[i + 1].get_left(), buff=0.06, color=MUTED, stroke_width=4, max_tip_length_to_length_ratio=0.35) for i in range(3)])
        fw = T("hacia delante: valores", 0.36, MEAN, BOLD).move_to([1.65, 1.75, 0])
        self.mark("forward"); self.play(FadeIn(inp), FadeIn(fw), run_time=0.4)
        for i in range(4):
            self.play(FadeIn(bxs[i], shift=0.15 * RIGHT), *([GrowArrow(arr[i - 1])] if i else []), run_time=0.5)
        back = VGroup(*[T(s_, 0.34, POINT, BOLD).next_to(bxs[i], DOWN, buff=0.25)
                        for i, s_ in enumerate(["∂/∂u = −0,622", "∂/∂z = −0,622", "∂/∂p = −2,649", "∂/∂castigo = 1"])])
        back[0].add(T("∂/∂w = × x = −1,245", 0.34, POINT, BOLD).next_to(back[0], DOWN, buff=0.12))
        rl = VGroup(*[T(r, 0.34, POINT).move_to([(xs[i] + xs[i + 1]) / 2, -0.95, 0]) for i, r in enumerate(["× 1", "× 0,235", "× (−1/p)"])])
        bk = T("hacia atrás: derivadas del castigo, multiplicando ritmos", 0.36, POINT, BOLD).move_to([0, -1.6, 0])
        self.mark("backward"); self.play(FadeIn(bk), FadeIn(back[3]), run_time=0.5)
        for i in (2, 1, 0):
            self.play(FadeIn(rl[i]), FadeIn(back[i], shift=0.15 * LEFT), run_time=0.6)
        hl = SurroundingRectangle(back[1], color=GOOD, buff=0.1, corner_radius=0.08)
        pm = T("∂/∂z = p − y = 0,378 − 1: probabilidad menos lo que pasó", 0.36, GOOD, BOLD).move_to([0, -2.3, 0])
        self.mark("pmy"); self.play(Create(hl), FadeIn(pm), run_time=0.7); self.wait(0.9)
        stp = T("un paso: w = 1,12 · b = −1,19  →  p = 74,2 % · castigo 0,299", 0.4, INK, BOLD).move_to([0, -3.15, 0])
        self.mark("learn"); self.play(FadeIn(stp, shift=0.1 * UP), run_time=0.7); self.wait(2.8)
        self.clear_but(cap)

        # ---------- Cierre ----------
        self.mark("legend")
        self.play(FadeOut(cap), run_time=0.6)
        e1 = Text("Derivar es medir ritmos de cambio", font=FONT, weight=BOLD, color=INK).scale(0.7)
        e2 = VGroup(T("tangente plana: mínimos y máximos", 0.42, MUTED), T("gradiente: hacia dónde sube el error, para todos los números", 0.42, MUTED),
                    T("regla de la cadena: los ritmos se multiplican", 0.42, MUTED), T("retropropagación: la cadena hacia atrás, en una pasada", 0.42, MUTED)).arrange(DOWN, buff=0.18)
        VGroup(e1, e2).arrange(DOWN, buff=0.45)
        self.mark("end_text"); self.play(Write(e1), run_time=1.2); self.play(FadeIn(e2, shift=0.2 * UP, lag_ratio=0.15), run_time=1.0)
        self.wait(3.3)
        self.mark("fade"); self.play(FadeOut(VGroup(e1, e2)), run_time=1)
        self.marks["total"] = [round(self.renderer.time, 3)]
        json.dump(self.marks, open("marks.json", "w"), indent=1)
