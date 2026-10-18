import json
import math
from itertools import combinations
import numpy as np
from manim import *
from scipy import stats

# Paleta oscura de la serie
BG = "#0E131B"; INK = "#E4EAF3"; MUTED = "#9BA7B9"; LINE = "#2B3545"
MEAN = "#6EA6FF"; SAMPLE = "#AE9FF3"; POINT = "#FF9A55"; GOOD = "#4CC38A"; WARN = "#F08A5D"
FONT = "Inter"
config.background_color = BG

SIN = [31.5, 30.7, 32.2, 30.9, 32.3, 35.3, 32.7, 33.2, 30.7, 30.5]
CON = [32.3, 29.5, 29.5, 31.7, 30.3, 32.0, 29.6, 32.1, 27.4, 29.6]
ALL = np.array(SIN + CON)


def T(s, sc=0.45, col=INK, w=None):
    return Text(s, font=FONT, color=col, weight=w or NORMAL).scale(sc)


def fmt(v, d=1, sign=False):
    s = f"{abs(v):.{d}f}".replace(".", ",")
    neg = v < 0 and round(abs(v), d) != 0
    return ("−" if neg else ("+" if sign and round(v, d) != 0 else "")) + s


class Axis:
    """Eje horizontal con marcas: mapea un valor a una x de la escena; y libre."""

    def __init__(self, x0, x1, ticks, label, left=-6.0, right=6.0, y=-3.0, fmt_t=None):
        self.x0, self.x1, self.L, self.R, self.y = x0, x1, left, right, y
        g = VGroup(Line([left, y, 0], [right, y, 0], stroke_color=MUTED, stroke_width=2))
        for v in ticks:
            g.add(Line(self.P(v, y), self.P(v, y + 0.1), stroke_color=MUTED, stroke_width=2))
            g.add(T(fmt_t(v) if fmt_t else str(v), 0.3, MUTED).next_to(self.P(v, y), DOWN, buff=0.12))
        if label: g.add(T(label, 0.32, INK).next_to([right, y, 0], DOWN, buff=0.48).align_to([right, 0, 0], RIGHT))
        self.g = g

    def X(self, v):
        return self.L + (v - self.x0) / (self.x1 - self.x0) * (self.R - self.L)

    def P(self, v, y):
        return np.array([self.X(v), y, 0])


def positions(A, lab, rows, r=0.13):
    """Posición de cada camión: su fila según la etiqueta, apilando los que se tocarían."""
    pos = [None] * 20
    for g in (0, 1):
        ix = sorted([i for i in range(20) if lab[i] == g], key=lambda i: ALL[i]); lv = []
        for i in ix:
            x = A.X(ALL[i]); k = 0
            while any(kk == k and abs(xx - x) < 2 * r + 0.02 for xx, kk in lv): k += 1
            lv.append((x, k)); pos[i] = np.array([x, rows[g] + k * (2 * r - 0.03), 0])
    return pos


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
        title = Text("Pruebas de hipótesis y p-valor", font=FONT, weight=BOLD, color=INK).scale(0.9)
        sub = Text("¿la diferencia es real o es suerte?", font=FONT, color=MUTED).scale(0.5)
        VGroup(title, sub).arrange(DOWN, buff=0.35)
        self.mark("title"); self.play(Write(title), run_time=1.3)
        self.mark("subtitle"); self.play(FadeIn(sub, shift=0.2 * UP)); self.wait(1.6)
        self.mark("title_out"); self.play(FadeOut(title), FadeOut(sub))

        # ---------- 1. la prueba A/B ----------
        cap = self.swap(None, self.caption("Un curso de conducción eficiente: ¿ahorra?", "10 camiones sin curso, 10 con curso: consumo del mes (L/100 km)"))
        A = Axis(26.8, 36, [27, 28, 29, 30, 31, 32, 33, 34, 35, 36], "consumo (L/100 km)", left=-5.4, right=6.0, y=-2.9)
        rows = [1.35, -1.45]
        lab = [0] * 10 + [1] * 10
        P = positions(A, lab, rows)
        dots = VGroup(*[Dot(P[i], radius=0.13, color=SAMPLE if i < 10 else POINT) for i in range(20)])
        rl = VGroup(T("sin curso", 0.38, SAMPLE, BOLD).move_to([-6.25, rows[0] + 0.05, 0]), T("con curso", 0.38, POINT, BOLD).move_to([-6.25, rows[1] + 0.05, 0]))
        self.play(FadeIn(A.g), FadeIn(rl), run_time=0.6)
        self.mark("dots"); self.play(LaggedStart(*[GrowFromCenter(d) for d in dots], lag_ratio=0.06), run_time=1.4)

        def mean_marks(lab):
            g = VGroup()
            for k in (0, 1):
                m = float(np.mean(ALL[[i for i in range(20) if lab[i] == k]]))
                g.add(Line(A.P(m, rows[k] - 0.3), A.P(m, rows[k] + 0.8), stroke_color=INK, stroke_width=5))
                g.add(T(f"media {fmt(m)}", 0.32, INK, BOLD).next_to(A.P(m, rows[k] - 0.3), DOWN, buff=0.08))
            return g
        mm = mean_marks(lab)
        diff = T("ahorro: 32,0 − 30,4 = 1,6 L/100 km", 0.46, POINT, BOLD).move_to([0.3, 0.05, 0])
        self.mark("means"); self.play(FadeIn(mm), run_time=0.7); self.play(FadeIn(diff, shift=0.1 * UP), run_time=0.6); self.wait(1.8)
        q = T("¿O los ahorradores cayeron por suerte en el grupo del curso?", 0.4, MUTED).move_to([0.3, 0.05, 0])
        self.play(FadeOut(diff), FadeIn(q), run_time=0.6); self.wait(2.2)

        # ---------- 2. barajar ----------
        cap = self.swap(cap, self.caption("Si el curso no hiciera nada, las etiquetas no importarían", "se barajan las etiquetas y se mira qué diferencia sale"))
        self.play(FadeOut(q), FadeOut(rl), run_time=0.3)
        rl2 = VGroup(T("«sin»", 0.38, INK, BOLD).move_to([-6.2, rows[0] + 0.05, 0]), T("«con»", 0.38, INK, BOLD).move_to([-6.2, rows[1] + 0.05, 0]))
        self.play(FadeIn(rl2), run_time=0.3)
        rng = np.random.default_rng(4); dl = None
        for j in range(3):
            idx = rng.permutation(20); lab = [1] * 20
            for i in idx[:10]: lab[i] = 0
            P = positions(A, lab, rows); mm2 = mean_marks(lab)
            d = float(np.mean(ALL[[i for i in range(20) if lab[i] == 0]]) - np.mean(ALL[[i for i in range(20) if lab[i] == 1]]))
            nd = T(f"barajada {j + 1}: diferencia {fmt(d, 2, True)}", 0.44, MEAN, BOLD).move_to([0.3, 0.05, 0])
            self.mark("shuffle")
            self.play(*[dots[i].animate.move_to(P[i]) for i in range(20)], Transform(mm, mm2), (Transform(dl, nd) if dl else FadeIn(nd)), run_time=0.9)
            if dl is None: dl = nd
            self.wait(1.0)
        self.play(FadeOut(VGroup(dots, mm, dl, rl2, A.g)), run_time=0.5)
        # histograma de todas las particiones
        H = Axis(-3.2, 3.2, [-3, -2, -1, 0, 1, 2, 3], "diferencia con las etiquetas barajadas (L/100 km)", left=-6.0, right=6.0, y=-2.9, fmt_t=lambda v: fmt(v, 0, True))
        dec = np.round(ALL * 10).astype(int); S = dec.sum()
        D = np.array([2 * dec[list(c)].sum() - S for c in combinations(range(20), 10)])
        b = np.round(D / 20).astype(int); cnt = {k: (int(np.sum((b == k) & (np.abs(D) < 160))), int(np.sum((b == k) & (np.abs(D) >= 160)))) for k in np.unique(b)}
        mx = max(a + c for a, c in cnt.values()); w = (H.X(0.2) - H.X(0)) * 0.86; bars = VGroup(); red = VGroup()
        for k, (a, c) in cnt.items():
            x = H.X(k * 0.2); hgt = 3.6 * (a + c) / mx
            if a: bars.add(Rectangle(width=w, height=3.6 * a / mx, stroke_width=0, fill_color=MUTED, fill_opacity=0.55).move_to([x, -2.9 + 3.6 * c / mx + 1.8 * a / mx, 0]))
            if c: red.add(Rectangle(width=w, height=3.6 * c / mx, stroke_width=0, fill_color=WARN, fill_opacity=0.95).move_to([x, -2.9 + 1.8 * c / mx, 0]))
        self.play(FadeIn(H.g), run_time=0.4)
        hl = T("las 184 756 formas de repartir los 20 camiones", 0.36, MUTED).move_to([0, 1.55, 0])
        self.mark("hist"); self.play(FadeIn(hl), LaggedStart(*[GrowFromEdge(r_, DOWN) for r_ in bars], lag_ratio=0.03), run_time=1.8)
        lines = VGroup(*[DashedLine(H.P(v, -2.9), H.P(v, 1.0), color=INK, stroke_width=3) for v in (-1.6, 1.6)])
        ll = T("real: +1,6", 0.36, INK, BOLD).next_to(H.P(1.6, 1.0), UP, buff=0.08)
        self.play(Create(lines), FadeIn(ll), run_time=0.7)
        self.mark("pval"); self.play(FadeIn(red), run_time=0.6)
        pv = VGroup(T("5 722 de 184 756", 0.4, WARN, BOLD), T("llegan a 1,6 o más", 0.4, WARN, BOLD), T("p-valor = 3,1 %", 0.6, WARN, BOLD)).arrange(DOWN, aligned_edge=LEFT, buff=0.15).move_to([-4.9, 0.3, 0])
        self.play(FadeIn(pv, shift=0.1 * UP), run_time=0.8); self.wait(2.4)
        nt = VGroup(T("no es la probabilidad", 0.36, MUTED), T("de que el curso no sirva", 0.36, MUTED)).arrange(DOWN, aligned_edge=LEFT, buff=0.1).move_to([4.9, 0.3, 0])
        self.play(FadeIn(nt), run_time=0.6); self.wait(1.8)
        self.clear_but(cap)

        # ---------- 3. errores y potencia ----------
        cap = self.swap(cap, self.caption("Dos maneras de equivocarse", "si el curso ahorra de verdad 1,0: ¿lo ve una prueba de 10 camiones por grupo?"))
        B = Axis(-2.4, 3.6, [-2, -1, 0, 1, 2, 3], "ahorro que mediría la prueba (L/100 km)", left=-6.0, right=3.0, y=-2.9, fmt_t=lambda v: fmt(v, 0, True))
        self.play(FadeIn(B.g), run_time=0.4)

        def bells(n):
            se = 1.6 * math.sqrt(2 / n); c = 1.96 * se
            yf = lambda x, m: -2.9 + 3.6 * math.exp(-0.5 * ((x - m) / se) ** 2)
            def area(m, a, bb, col, op):
                a, bb = max(a, -2.4), min(bb, 3.6)
                pts = [B.P(x, yf(x, m)) for x in np.linspace(a, bb, 80)]
                return Polygon(*pts, B.P(bb, -2.9), B.P(a, -2.9), stroke_width=0, fill_color=col, fill_opacity=op)
            g = VGroup(area(1.0, -c, c, SAMPLE, 0.35), area(1.0, c, 3.6, GOOD, 0.4), area(0, c, 3.6, WARN, 0.85), area(0, -2.4, -c, WARN, 0.85))
            g.add(DashedVMobject(VMobject(stroke_color=MUTED, stroke_width=3).set_points_smoothly([B.P(x, yf(x, 0)) for x in np.linspace(-2.4, 3.6, 200)]), num_dashes=50))
            g.add(VMobject(stroke_color=MEAN, stroke_width=4).set_points_smoothly([B.P(x, yf(x, 1.0)) for x in np.linspace(-2.4, 3.6, 200)]))
            g.add(*[Line(B.P(v, -2.9), B.P(v, 0.95), stroke_color=INK, stroke_width=3) for v in (-c, c) if v > -2.4])
            pw = stats.norm.sf(1.96 - 1.0 / se) + stats.norm.cdf(-1.96 - 1.0 / se)
            info = VGroup(T(f"{n} camiones por grupo", 0.44, INK, BOLD), T("falsa alarma (α): 5 %", 0.38, WARN, BOLD),
                          T(f"se escapa (β): {fmt(100 * (1 - pw), 0)} %", 0.38, SAMPLE, BOLD), T(f"potencia: {fmt(100 * pw, 0)} %", 0.5, GOOD, BOLD)).arrange(DOWN, aligned_edge=LEFT, buff=0.2).move_to([4.75, -0.6, 0])
            return g, info
        g, info = bells(10)
        lb = VGroup(T("no hace nada", 0.32, MUTED).move_to(B.P(-0.55, 0.95)), T("ahorra 1,0", 0.32, MEAN, BOLD).move_to(B.P(1.95, 0.95)))
        self.mark("bells"); self.play(FadeIn(g), FadeIn(lb), run_time=1.2); self.play(FadeIn(info, lag_ratio=0.2), run_time=1.0); self.wait(2.8)
        g2, info2 = bells(41)
        self.mark("power2"); self.play(Transform(g, g2), Transform(info, info2), FadeOut(lb), run_time=1.4)
        nn = VGroup(T("4 veces más camiones:", 0.34, MUTED), T("la mitad de error estándar", 0.34, MUTED)).arrange(DOWN, aligned_edge=LEFT, buff=0.1).move_to([4.75, -2.2, 0])
        self.play(FadeIn(nn), run_time=0.5); self.wait(2.0)
        self.clear_but(cap)

        # ---------- 4. 20 métricas ----------
        cap = self.swap(cap, self.caption("Prueba 20 métricas y alguna saldrá «significativa»", "un curso que no hace nada: p-valor de cada indicador"))
        names = ["exceso de velocidad", "tiempo en ruta", "giros bruscos", "temperatura motor", "quejas de clientes", "AdBlue", "repostajes", "km en vacío", "retrasos",
                 "acelerones", "averías", "velocidad media", "horas de motor", "frenazos", "neumáticos", "uso del crucero", "ralentí", "paradas", "consumo", "revoluciones"]
        ps = np.sort(np.random.default_rng(9).random(20)); ps[0] = 0.012; ps[1] = max(ps[1], 0.07)
        L = Axis(-4, 0, [-4, -3, -2, math.log10(0.05), 0], "p-valor (escala logarítmica)", left=-2.4, right=5.8, y=-3.15,
                 fmt_t=lambda v: {-4: "0,01 %", -3: "0,1 %", -2: "1 %", 0: "100 %"}.get(round(v, 3), "5 %"))
        rowsy = [2.0 - k * 0.255 for k in range(20)]
        lbls = VGroup(*[T(names[k], 0.27, MUTED).move_to([-2.6, rowsy[k], 0], aligned_edge=RIGHT) for k in range(20)])
        pdots = VGroup(*[Dot(L.P(math.log10(ps[k]), rowsy[k]), radius=0.075, color=WARN if ps[k] < 0.05 else MUTED) for k in range(20)])
        th = Line(L.P(math.log10(0.05), -3.15), L.P(math.log10(0.05), 2.2), stroke_color=MEAN, stroke_width=4)
        self.play(FadeIn(L.g), FadeIn(lbls), Create(th), run_time=0.8)
        self.mark("metrics"); self.play(LaggedStart(*[GrowFromCenter(d) for d in pdots[::-1]], lag_ratio=0.05), run_time=1.5)
        hit = VGroup(T("«¡el curso reduce el", 0.36, WARN, BOLD), T("exceso de velocidad!»", 0.36, WARN, BOLD), T("p = 1,2 %", 0.34, WARN)).arrange(DOWN, aligned_edge=LEFT, buff=0.1).move_to([-0.6, -0.4, 0])
        hit.add(Arrow(hit.get_top() + 0.1 * UP + 0.6 * RIGHT, pdots[0].get_center() + 0.12 * DOWN + 0.08 * LEFT, buff=0.08, color=WARN, stroke_width=3, max_tip_length_to_length_ratio=0.08))
        self.play(FadeIn(hit), run_time=0.6); self.wait(1.8)
        f64 = VGroup(T("1 − 0,95²⁰ = 64 %", 0.42, INK, BOLD), T("de los cursos inútiles dan algún «hallazgo»", 0.32, INK)).arrange(DOWN, buff=0.12).move_to([3.0, -1.9, 0])
        f64 = VGroup(BackgroundRectangle(f64, color=BG, fill_opacity=1, buff=0.08), f64).set_z_index(2)
        self.play(FadeIn(f64), run_time=0.6); self.wait(1.6)
        th2 = Line(L.P(math.log10(0.0025), -3.15), L.P(math.log10(0.0025), 2.2), stroke_color=MEAN, stroke_width=4)
        bl = T("Bonferroni: 5 % / 20 = 0,25 %", 0.34, MEAN, BOLD).next_to(th2, UP, buff=0.05).shift(0.3 * LEFT)
        self.mark("bonf"); self.play(Transform(th, th2), pdots[0].animate.set_color(MUTED), FadeOut(hit), FadeIn(bl), run_time=1.2); self.wait(2.0)
        self.clear_but(cap)

        # ---------- 5. significativo no es importante ----------
        cap = self.swap(cap, self.caption("Significativo no es importante", "5 000 camiones por grupo y un ahorro de 0,1 L/100 km"))
        C = Axis(-0.6, 2.0, [-0.5, 0, 0.5, 1, 1.5, 2], "", left=-6.0, right=6.0, y=-1.8, fmt_t=lambda v: fmt(v, 1, True) if v else "0")
        eur = VGroup(*[T((fmt(v * 1650, 0, True).replace("1650", "1 650").replace("2475", "2 475").replace("3300", "3 300") + " €") if v else "0 €", 0.28, GOOD).next_to(C.P(v, -1.8), DOWN, buff=0.45) for v in (-0.5, 0, 0.5, 1, 1.5, 2)])
        eur.add(T("ahorro en L/100 km y, en verde, en € por camión y año", 0.32, INK).move_to([2.6, -2.95, 0]))
        zero = Line(C.P(0, -1.8), C.P(0, 1.6), stroke_color=INK, stroke_width=3)
        pay = Rectangle(width=C.X(2.0) - C.X(0.182), height=3.4, stroke_width=0, fill_color=GOOD, fill_opacity=0.12).move_to([(C.X(2.0) + C.X(0.182)) / 2, -0.1, 0])
        payl = T("paga el curso de 300 € (≥ 0,18)", 0.34, GOOD, BOLD).next_to(pay.get_top(), DOWN, buff=0.15).align_to(pay, LEFT).shift(0.15 * RIGHT)
        self.play(FadeIn(C.g), FadeIn(eur), Create(zero), FadeIn(pay), FadeIn(payl), run_time=0.9)
        se = 1.6 * math.sqrt(2 / 5000); lo, hi = 0.1 - 1.96 * se, 0.1 + 1.96 * se
        bar = VGroup(Line(C.P(lo, 0.0), C.P(hi, 0.0), stroke_color=MEAN, stroke_width=10), Circle(0.1, color=MEAN, stroke_width=5, fill_color=BG, fill_opacity=1).move_to(C.P(0.1, 0)))
        bt = VGroup(T(f"p = 0,2 %: muy significativo", 0.4, MEAN, BOLD), T(f"pero ahorra de {fmt(lo * 1650, 0)} a {fmt(hi * 1650, 0)} € al año: no paga el curso", 0.38, INK)).arrange(DOWN, buff=0.15).move_to([2.6, -0.95, 0])
        self.mark("ci"); self.play(Create(bar), run_time=0.8); self.play(FadeIn(bt, lag_ratio=0.2), run_time=0.8); self.wait(2.8)
        self.clear_but(cap)

        # ---------- Cierre ----------
        self.mark("legend")
        self.play(FadeOut(cap), run_time=0.6)
        e1 = Text("p-valor: cuánto sorprendería el resultado si no pasara nada", font=FONT, weight=BOLD, color=INK).scale(0.6)
        e2 = VGroup(T("no es la probabilidad de que la nula sea cierta", 0.42, MUTED), T("con pocos camiones, los efectos reales se escapan", 0.42, MUTED),
                    T("significativo no es importante: mira el intervalo en €", 0.42, MUTED), T("una métrica principal, decidida antes, y mirar al final", 0.42, MUTED)).arrange(DOWN, buff=0.18)
        VGroup(e1, e2).arrange(DOWN, buff=0.45)
        self.mark("end_text"); self.play(Write(e1), run_time=1.2); self.play(FadeIn(e2, shift=0.2 * UP, lag_ratio=0.15), run_time=1.0)
        self.wait(3.5)
        self.mark("fade"); self.play(FadeOut(VGroup(e1, e2)), run_time=1)
        self.marks["total"] = [round(self.renderer.time, 3)]
        json.dump(self.marks, open("marks.json", "w"), indent=1)
