import json
import math
import numpy as np
from manim import *
from scipy import stats

# Paleta oscura de la serie
BG = "#0E131B"; INK = "#E4EAF3"; MUTED = "#9BA7B9"; LINE = "#2B3545"
MEAN = "#6EA6FF"; SAMPLE = "#AE9FF3"; POINT = "#FF9A55"; GOOD = "#4CC38A"; WARN = "#F08A5D"
FONT = "Inter"
config.background_color = BG
D = json.load(open("../mt/m14/embed.json"))


def T(s, sc=0.45, col=INK, w=None):
    return Text(s, font=FONT, color=col, weight=w or NORMAL).scale(sc)


def fmt(v, d=1):
    s = f"{abs(v):.{d}f}".replace(".", ",")
    return ("−" if v < 0 and round(abs(v), d) != 0 else "") + s


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

    def curve(self, f, a, b, col, w=4, n=240, ymin=None):
        pts = []
        for x in np.linspace(a, b, n):
            y = f(x)
            if ymin is not None and y < ymin:
                continue
            pts.append(self.P(x, y))
        return VMobject(stroke_color=col, stroke_width=w).set_points_smoothly(pts)


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
        title = Text("Máxima verosimilitud", font=FONT, weight=BOLD, color=INK).scale(0.95)
        sub = Text("qué valor del parámetro hacía más probable lo que pasó", font=FONT, color=MUTED).scale(0.48)
        VGroup(title, sub).arrange(DOWN, buff=0.35)
        self.mark("title"); self.play(Write(title), run_time=1.3)
        self.mark("subtitle"); self.play(FadeIn(sub, shift=0.2 * UP)); self.wait(1.6)
        self.mark("title_out"); self.play(FadeOut(title), FadeOut(sub))

        A = np.array(D["A"]); n = len(A)
        pmf = lambda k, l: math.exp(k * math.log(l) - l - math.lgamma(k + 1))
        llp = lambda l: float(np.sum(A * math.log(l) - l) - sum(math.lgamma(k + 1) for k in A))

        # ---------- 1. los datos quietos, el parámetro se mueve ----------
        cap = self.swap(None, self.caption("Los datos quietos, el parámetro se mueve", "alertas de frenada brusca de un camión en 14 días · Poisson con λ alertas al día"))
        lam = ValueTracker(1.0)
        cells = VGroup()
        for i, k in enumerate(A):
            cells.add(T(str(k), 0.55, POINT, BOLD).move_to([-6.1 + i * 0.94, 2.25, 0]))
        bars = always_redraw(lambda: VGroup(*[
            Rectangle(width=0.32, height=max(0.02, 0.9 * min(1, pmf(k, lam.get_value()) / 0.4)), stroke_width=0,
                      fill_color=WARN if pmf(k, lam.get_value()) < 0.02 else MEAN, fill_opacity=0.9)
            .move_to([-6.1 + i * 0.94, 0.95, 0], aligned_edge=DOWN) for i, k in enumerate(A)]))
        base = Line([-6.6, 0.95, 0], [6.6, 0.95, 0], stroke_color=LINE, stroke_width=2)
        plab = T("probabilidad que da λ a cada día", 0.3, MUTED).move_to([-4.6, 0.7, 0])
        self.mark("dots"); self.play(LaggedStart(*[FadeIn(c, shift=0.1 * DOWN) for c in cells], lag_ratio=0.05), run_time=1.0)
        self.play(Create(base), FadeIn(bars), FadeIn(plab), run_time=0.7)
        Pf = Panel(-0.6, 8.6, 0, 6, (-6.3, -0.7, -3.2, 0.0), xt=range(0, 9), xl="alertas en un día")
        cnt = np.bincount(A, minlength=9)
        dts = VGroup(*[Dot(Pf.P(k, j + 0.5), radius=0.1, color=POINT) for k in range(9) for j in range(cnt[k])])
        exp_bars = always_redraw(lambda: VGroup(*[
            Rectangle(width=Pf.P(0.36, 0)[0] - Pf.P(-0.36, 0)[0], height=max(0.01, Pf.P(0, min(6, n * pmf(k, lam.get_value())))[1] - Pf.B),
                      stroke_color=MEAN, stroke_width=2, fill_color=MEAN, fill_opacity=0.25).move_to(Pf.P(k, 0), aligned_edge=DOWN) for k in range(9)]))
        Pc = Panel(0, 6, -61, -22, (0.9, 6.5, -3.2, 0.0), xt=range(0, 7), xl="λ que se supone", yt=(-60, -40, -25), yf=lambda v: fmt(v, 0))
        ccurve = Pc.curve(llp, 0.42, 6, MEAN, 4)
        cdot = always_redraw(lambda: Dot(Pc.P(lam.get_value(), max(-61, llp(lam.get_value()))), radius=0.11, color=POINT))
        lread = always_redraw(lambda: T(f"λ = {fmt(lam.get_value())}", 0.42, POINT, BOLD).move_to([3.7, 0.45, 0]))
        self.play(FadeIn(Pf.g), FadeIn(Pc.g), FadeIn(dts), FadeIn(exp_bars), Create(ccurve), run_time=1.0)
        self.add(cdot, lread)
        self.mark("lam"); self.play(lam.animate.set_value(5.0), run_time=3.0, rate_func=smooth); self.wait(0.5)
        self.play(lam.animate.set_value(2.5), run_time=2.4, rate_func=smooth)
        peak = DashedLine(Pc.P(2.5, -61), Pc.P(2.5, llp(2.5)), color=INK, stroke_width=2.5)
        plab2 = T("cima: λ = 2,5 = 35 / 14, la media", 0.36, INK, BOLD).next_to(Pc.P(2.5, llp(2.5)), UP, buff=0.12).shift(0.6 * RIGHT)
        self.mark("peak"); self.play(Create(peak), FadeIn(plab2), FadeOut(lread), run_time=0.8); self.wait(2.2)
        self.remove(bars, exp_bars, cdot)
        self.clear_but(cap)

        # ---------- 2. multiplicar → sumar logaritmos ----------
        cap = self.swap(cap, self.caption("Multiplicar probabilidades, sumar logaritmos", "días independientes: la probabilidad de todos juntos es el producto"))
        l1 = T("0,214 × 0,205 × 0,134 × 0,082 × 0,134 × …", 0.5, INK).move_to([0, 1.7, 0])
        l2 = T("= 1,2 · 10⁻¹¹   (14 días)", 0.55, MEAN, BOLD).next_to(l1, DOWN, buff=0.3)
        self.mark("prod"); self.play(FadeIn(l1, shift=0.2 * RIGHT), run_time=1.0); self.play(FadeIn(l2), run_time=0.6); self.wait(0.8)
        rows = VGroup(T("30 días:  10⁻²⁴", 0.44, MUTED), T("365 días:  10⁻²⁸³", 0.44, MUTED),
                      T("500 días:  0   (no cabe en el ordenador)", 0.44, WARN, BOLD)).arrange(DOWN, aligned_edge=LEFT, buff=0.2).move_to([0, -0.6, 0])
        self.mark("tiny"); self.play(LaggedStart(*[FadeIn(r, shift=0.15 * UP) for r in rows], lag_ratio=0.5), run_time=1.6); self.wait(1.0)
        lg = VGroup(T("log 0,214 + log 0,205 + … = −25,1", 0.48, GOOD, BOLD), T("sumar es fácil · el logaritmo no mueve la cima", 0.38, INK)).arrange(DOWN, buff=0.18).move_to([0, -2.75, 0])
        self.mark("logsum"); self.play(FadeIn(lg, shift=0.2 * UP), run_time=0.9); self.wait(2.2)
        self.clear_but(cap)

        # ---------- 3. más datos, cima más estrecha ----------
        cap = self.swap(cap, self.caption("Más datos, cima más estrecha", "siempre un 10 % de entregas tarde · la anchura de la cima es la incertidumbre"))
        Pa = Panel(0, 0.45, 0, 1.12, (-6.2, 1.2, -3.0, 1.9), xt=(0, 0.1, 0.2, 0.3, 0.4), xf=lambda v: f"{int(round(v * 100))} %", xl="porcentaje de entregas tarde")
        self.play(FadeIn(Pa.g), run_time=0.5)
        thr = DashedLine(Pa.P(0, 0.1465), Pa.P(0.45, 0.1465), color=MUTED, stroke_width=2)
        self.play(Create(thr), run_time=0.4)
        info = VGroup(); prev = None
        for j, (m, col) in enumerate(((10, MUTED), (40, SAMPLE), (200, MEAN), (1000, GOOD))):
            k = m // 10; lm = stats.binom.logpmf(k, m, 0.1)
            cv = Pa.curve(lambda q: math.exp(stats.binom.logpmf(k, m, q) - lm), 0.002, 0.45, col, 4, 400)
            g = lambda q: stats.binom.logpmf(k, m, q) - lm + 1.92073
            from scipy.optimize import brentq
            lo, hi = brentq(g, 1e-9, 0.1), brentq(g, 0.1, 0.99)
            row = T(f"{k} de {m:,}".replace(",", " ") + f":  {fmt(100 * lo)} % a {fmt(100 * hi)} %", 0.4, col, BOLD)
            info.add(row)
            row.move_to([4.3, 1.4 - 0.62 * j, 0])
            self.mark("curve"); self.play(Create(cv), FadeIn(row, shift=0.1 * LEFT), run_time=1.0); self.wait(0.6)
        note = T("× 4 datos → anchura ÷ 2", 0.4, POINT, BOLD).move_to([4.3, -1.3, 0])
        self.play(FadeIn(note), run_time=0.5); self.wait(1.8)
        self.clear_but(cap)

        # ---------- 4. de dónde salen las pérdidas ----------
        cap = self.swap(cap, self.caption("Cada pérdida es un modelo de ruido", "pérdida = menos la log-verosimilitud"))
        xs = np.linspace(-3, 3, 200)

        def shape(kind, cx):
            P = Panel(-3, 3, 0, 1.05, (cx - 1.5, cx + 1.5, 0.3, 2.0))
            if kind == 0:
                c = P.curve(lambda x: math.exp(-x * x / 2), -3, 3, MEAN, 4)
            elif kind == 1:
                c = P.curve(lambda x: math.exp(-abs(x) * 1.4), -3, 3, MEAN, 4, 401)
            else:
                c = VGroup(Rectangle(width=0.7, height=1.0, fill_color=MEAN, fill_opacity=0.6, stroke_width=0).move_to(P.P(-1, 0), aligned_edge=DOWN),
                           Rectangle(width=0.7, height=0.55, fill_color=POINT, fill_opacity=0.8, stroke_width=0).move_to(P.P(1, 0), aligned_edge=DOWN),
                           T("no", 0.3, MUTED).next_to(P.P(-1, 0), DOWN, buff=0.08), T("sí", 0.3, MUTED).next_to(P.P(1, 0), DOWN, buff=0.08))
            return VGroup(Line(P.P(-3, 0), P.P(3, 0), stroke_color=MUTED, stroke_width=2), c)
        cols = [("ruido en campana", "mínimos cuadrados", "Σ error²  →  la media", "regresión lineal"),
                ("colas gordas (Laplace)", "error absoluto", "Σ |error|  →  la mediana", "28,2 min frente a 20"),
                ("sí / no (Bernoulli)", "pérdida logarítmica", "−Σ log p(lo que pasó)", "regresión logística")]
        for j, (a, b, c, d) in enumerate(cols):
            cx = -4.6 + 4.6 * j
            g = VGroup(shape(j, cx), T(a, 0.36, INK, BOLD).move_to([cx, 2.45 - 2.75, 0]),
                       Arrow([cx, -0.55, 0], [cx, -1.25, 0], color=MUTED, stroke_width=3, buff=0),
                       T(b, 0.44, POINT, BOLD).move_to([cx, -1.6, 0]), T(c, 0.34, INK).move_to([cx, -2.15, 0]), T(d, 0.3, MUTED).move_to([cx, -2.6, 0]))
            self.mark("loss"); self.play(FadeIn(g, shift=0.2 * UP), run_time=0.9); self.wait(1.3)
        self.wait(1.4)
        self.clear_but(cap)

        # ---------- 5. pocos datos ----------
        cap = self.swap(cap, self.caption("Pocos datos: la máxima verosimilitud se lo cree todo", "cliente nuevo: 0 entregas tarde de 5"))
        Pm = Panel(0, 0.4, 0, 1.12, (-6.2, 1.4, -3.0, 1.9), xt=(0, 0.1, 0.2, 0.3, 0.4), xf=lambda v: f"{int(round(v * 100))} %", xl="porcentaje de entregas tarde")
        lik = DashedVMobject(Pm.curve(lambda q: (1 - q) ** 5, 0, 0.4, POINT, 4), num_dashes=40)
        self.play(FadeIn(Pm.g), run_time=0.4)
        d0 = Dot(Pm.P(0, 1), radius=0.12, color=POINT)
        t0 = VGroup(T("máxima verosimilitud: 0 %", 0.42, POINT, BOLD), T("«nunca llegará tarde»", 0.38, INK)).arrange(DOWN, aligned_edge=LEFT, buff=0.15).move_to([4.3, 1.2, 0])
        self.mark("mle0"); self.play(Create(lik), FadeIn(d0), FadeIn(t0), run_time=1.1); self.wait(1.4)
        a, b = 2.6, 19.4
        bmode = (a - 1) / (a + b - 2)
        prior = Pm.curve(lambda q: math.exp(stats.beta.logpdf(q, a, b) - stats.beta.logpdf(bmode, a, b)), 0.001, 0.4, SAMPLE, 3.5)
        post = Pm.curve(lambda q: math.exp(stats.beta.logpdf(q, a, b + 5) - stats.beta.logpdf(0.064, a, b + 5)), 0.001, 0.4, MEAN, 4.5)
        t1 = VGroup(T("+ previa: 20 entregas", 0.38, SAMPLE, BOLD), T("imaginarias al 8 % de la flota", 0.38, SAMPLE, BOLD)).arrange(DOWN, aligned_edge=LEFT, buff=0.1).move_to([4.3, -0.2, 0])
        t1.align_to(t0, LEFT)
        self.mark("prior"); self.play(Create(prior), FadeIn(t1), run_time=1.0); self.wait(0.6)
        dm = Dot(Pm.P(0.064, 1), radius=0.12, color=MEAN)
        t2 = VGroup(T("MAP = (0 + 1,6) / (5 + 20)", 0.38, MEAN, BOLD), T("= 6,4 %", 0.5, MEAN, BOLD), T("= regularización", 0.36, MUTED)).arrange(DOWN, aligned_edge=LEFT, buff=0.12).move_to([4.3, -1.75, 0])
        t2.align_to(t0, LEFT)
        self.play(Create(post), FadeIn(dm), FadeIn(t2), run_time=1.0); self.wait(2.4)
        self.clear_but(cap)

        # ---------- 6. subir la colina ----------
        cap = self.swap(cap, self.caption("Sin fórmula: subir la colina a pasos", "regresión logística del retraso según los km · cada paso va hacia donde más sube"))
        km, yt = np.array(D["km"]), np.array(D["tarde"])

        def llm(u, b_):
            z = b_ * (km / 100 - u)
            return float(np.mean(yt * -np.logaddexp(0, -z) + (1 - yt) * -np.logaddexp(0, z)))
        Pg = Panel(0, 320, 0, 4.5, (-6.2, 1.6, -3.1, 2.0), xt=(0, 100, 200, 300), xl="punto medio (km)", yt=(), yl="inclinación")
        NX, NY = 40, 24; cw = (Pg.R - Pg.L) / NX; chh = (Pg.Tp - Pg.B) / NY
        lev = [-0.47, -0.48, -0.5, -0.53, -0.57, -0.62, -0.68, -0.75, -0.85, -1.0, -1.3]
        sq = VGroup()
        for i in range(NX):
            for j in range(NY):
                v = llm((i + 0.5) / NX * 3.2, (j + 0.5) / NY * 4.5)
                q = next((t for t, th in enumerate(lev) if v >= th), len(lev))
                op = 0.06 + 0.8 * (1 - q / len(lev)) ** 1.6
                sq.add(Rectangle(width=cw * 1.02, height=chh * 1.02, stroke_width=0, fill_color=interpolate_color(ManimColor(BG), ManimColor(MEAN), op), fill_opacity=1)
                       .move_to([Pg.L + (i + 0.5) * cw, Pg.B + (j + 0.5) * chh, 0]))
        self.play(FadeIn(sq), FadeIn(Pg.g), run_time=0.8)
        star = Star(5, outer_radius=0.17, color=INK, fill_color=INK, fill_opacity=1).move_to(Pg.P(D["lg"][0], D["lg"][1]))
        self.play(FadeIn(star), run_time=0.4)
        u, bb = 0.6, 0.6; pts = [(u, bb)]
        for _ in range(30):
            z = bb * (km / 100 - u); p = 1 / (1 + np.exp(-z)); r = yt - p
            gu, gb = float(np.mean(r * -bb)), float(np.mean(r * (km / 100 - u)))
            u += 3 * gu; bb += 3 * gb; pts.append((u, bb))
        readout = VGroup(T("pérdida logarítmica", 0.36, MUTED)).move_to([4.5, 1.2, 0])
        val = T(f"{fmt(-llm(*pts[0]), 3)}", 0.62, POINT, BOLD).next_to(readout, DOWN, buff=0.15)
        self.play(FadeIn(readout), FadeIn(val), run_time=0.4)
        cur = Dot(Pg.P(pts[0][0] * 100, pts[0][1]), radius=0.1, color=POINT); self.add(cur)
        path = VGroup()
        self.mark("step")
        for k in range(1, 16):
            a0, a1 = pts[k - 1], pts[k]
            seg = Line(Pg.P(a0[0] * 100, a0[1]), Pg.P(a1[0] * 100, a1[1]), color=POINT, stroke_width=4)
            nv = T(f"{fmt(-llm(*a1), 3)}", 0.62, POINT, BOLD).move_to(val)
            self.play(Create(seg), cur.animate.move_to(Pg.P(a1[0] * 100, a1[1])), Transform(val, nv), run_time=0.32 if k > 3 else 0.5)
            path.add(seg)
        a1 = pts[-1]
        nv = T(f"{fmt(-llm(*a1), 3)}", 0.62, GOOD, BOLD).move_to(val)
        self.play(cur.animate.move_to(Pg.P(a1[0] * 100, a1[1])), Transform(val, nv), run_time=0.6)
        fin = VGroup(T("en la cima: 164 km · 2,12", 0.38, INK, BOLD), T("= el ajuste de máxima verosimilitud", 0.34, MUTED),
                     T("= descenso de gradiente sobre la pérdida", 0.34, MUTED)).arrange(DOWN, aligned_edge=LEFT, buff=0.14).move_to([4.6, -1.1, 0])
        self.mark("star"); self.play(FadeIn(fin, lag_ratio=0.2), run_time=0.9); self.wait(2.2)
        self.clear_but(cap)

        # ---------- Cierre ----------
        self.mark("legend")
        self.play(FadeOut(cap), run_time=0.6)
        e1 = Text("Elegir lo que hace más probable lo que pasó", font=FONT, weight=BOLD, color=INK).scale(0.66)
        e2 = VGroup(T("conteos y porcentajes: la media, k / n", 0.42, MUTED), T("ruido en campana: mínimos cuadrados · sí/no: pérdida logarítmica", 0.42, MUTED),
                    T("la anchura de la cima dice cuánto fiarse", 0.42, MUTED), T("con pocos datos, añade una previa", 0.42, MUTED)).arrange(DOWN, buff=0.18)
        VGroup(e1, e2).arrange(DOWN, buff=0.45)
        self.mark("end_text"); self.play(Write(e1), run_time=1.2); self.play(FadeIn(e2, shift=0.2 * UP, lag_ratio=0.15), run_time=1.0)
        self.wait(3.3)
        self.mark("fade"); self.play(FadeOut(VGroup(e1, e2)), run_time=1)
        self.marks["total"] = [round(self.renderer.time, 3)]
        json.dump(self.marks, open("marks.json", "w"), indent=1)
