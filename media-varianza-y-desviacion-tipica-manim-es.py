import json
import numpy as np
from manim import *

# Paleta oscura de la serie
BG = "#0E131B"; INK = "#E4EAF3"; MUTED = "#9BA7B9"; LINE = "#2B3545"
MEAN = "#6EA6FF"; SAMPLE = "#AE9FF3"; POINT = "#FF9A55"; GOOD = "#4CC38A"; WARN = "#F08A5D"
FONT = "Inter"
config.background_color = BG

X = [32, 35, 30, 33, 37, 31, 34, 32]          # L/100 km de los 8 viajes
R = [12, 25, 8, 15, 40, 10, 18, 16]           # minutos de ralentí


def T(s, sc=0.45, col=INK, w=None):
    return Text(s, font=FONT, color=col, weight=w or NORMAL).scale(sc)


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
        title = Text("Media, varianza y desviación típica", font=FONT, weight=BOLD, color=INK).scale(0.85)
        sub = Text("el centro de una columna y lo que se alejan los datos de él", font=FONT, color=MUTED).scale(0.5)
        VGroup(title, sub).arrange(DOWN, buff=0.35)
        self.mark("title"); self.play(Write(title), run_time=1.3)
        self.mark("subtitle"); self.play(FadeIn(sub, shift=0.2 * UP)); self.wait(1.8)
        self.mark("title_out"); self.play(FadeOut(title), FadeOut(sub))

        # ---------- 1. punto de equilibrio ----------
        cap = self.swap(None, self.caption("La media es el punto de equilibrio", "consumo de 8 viajes de un camión (L/100 km)"))
        u = 1.05; x0 = 33; by = -1.2
        sx = lambda v: (v - x0) * u
        ang = ValueTracker(0.0); piv = ValueTracker(30.0)

        def beam():
            a = ang.get_value(); p = np.array([sx(piv.get_value()), by, 0])
            rot = lambda v: p + np.array([(sx(v) - p[0]) * np.cos(a), (sx(v) - p[0]) * np.sin(a), 0])
            g = VGroup(Line(rot(27.5), rot(38.5), color=INK, stroke_width=7))
            cnt = {}
            for v in X:
                cnt[v] = cnt.get(v, 0) + 1
                q = rot(v) + np.array([-np.sin(a), np.cos(a), 0]) * (0.2 + 0.36 * (cnt[v] - 1))
                g.add(Dot(q, radius=0.16, color=POINT))
            g.add(Triangle(color=MEAN, fill_color=MEAN, fill_opacity=1).scale(0.3).move_to(p + 0.27 * DOWN))
            return g
        b = always_redraw(beam)
        ticks = VGroup(*[T(str(v), 0.36, MUTED).move_to([sx(v), by - 0.95, 0]) for v in range(28, 39, 2)])
        self.mark("beam"); self.play(FadeIn(b), FadeIn(ticks), run_time=0.8)
        # apoyado en 30: se inclina hacia la derecha
        self.play(ang.animate.set_value(-0.16), run_time=1.0, rate_func=smooth)
        lab = T("apoyado en 30: se inclina", 0.45, MUTED).move_to([0, 1.2, 0])
        self.play(FadeIn(lab), run_time=0.4); self.wait(0.6)
        self.mark("balance")
        self.play(ang.animate.set_value(0.0), piv.animate.set_value(33.0), run_time=1.8, rate_func=smooth)
        lab2 = T("en 33 se equilibra:  264 / 8 = 33", 0.55, MEAN, BOLD).move_to([0, 1.2, 0])
        self.play(Transform(lab, lab2), run_time=0.5); self.wait(3.0)
        b.clear_updaters()
        self.clear_but(cap)

        # ---------- 2. desviaciones y cuadrados ----------
        cap = self.swap(cap, self.caption("Las desviaciones suman 0: se elevan al cuadrado", "cada viaje menos la media"))
        ox = -1.0; rows = VGroup(); rh = 0.52; top = 1.65
        mline = DashedLine([ox, top + 0.2, 0], [ox, top - 8 * rh, 0], color=MEAN)
        bars = VGroup(); vals = VGroup(); sqs = VGroup()
        for i, v in enumerate(X):
            y = top - rh / 2 - i * rh; d = v - 33
            bars.add(Line([ox, y, 0], [ox + d * 0.85, y, 0], color=SAMPLE, stroke_width=12, stroke_opacity=0.8) if d else VMobject())
            rows.add(Dot([ox + d * 0.85, y, 0], radius=0.13, color=POINT))
            vals.add(T(("+" if d > 0 else "−" if d < 0 else "") + str(abs(d)), 0.4, SAMPLE, BOLD).move_to([3.2, y, 0]))
            sqs.add(T(str(d * d), 0.4, POINT, BOLD).move_to([4.6, y, 0]))
        hd = VGroup(T("d", 0.42, SAMPLE, BOLD).move_to([3.2, top + 0.3, 0]), T("d²", 0.42, POINT, BOLD).move_to([4.6, top + 0.3, 0]))
        self.mark("dev"); self.play(Create(mline), FadeIn(rows), run_time=0.6)
        self.play(*[Create(x) for x in bars], FadeIn(vals), FadeIn(hd[0]), run_time=1.2)
        s0 = T("suma: (2 + 4 + 1) − (1 + 3 + 2 + 1) = 0", 0.42, INK).move_to([0.6, -3.1, 0])
        self.play(FadeIn(s0), run_time=0.5); self.wait(1.6)
        self.mark("sq"); self.play(FadeIn(sqs, shift=0.2 * LEFT), FadeIn(hd[1]), run_time=0.8)
        s1 = T("cuadrados: 36    varianza 36 / 7 = 5,14    desviación típica √5,14 = 2,27 L/100 km", 0.36, INK, BOLD).move_to([0.3, -3.1, 0])
        self.play(Transform(s0, s1), run_time=0.6); self.wait(0.6)
        band = Rectangle(width=2 * 2.27 * 0.85, height=8 * rh + 0.2, stroke_width=0, fill_color=MEAN, fill_opacity=0.15).move_to([ox, top - 4 * rh, 0])
        self.mark("sd"); self.play(FadeIn(band), run_time=0.7); self.wait(3.4)
        self.clear_but(cap)

        # ---------- 3. el dato erróneo ----------
        cap = self.swap(cap, self.caption("Un registro erróneo: la media se va", "el 37 del viaje 5 se guarda como 370"))
        lo, hi = 20, 380; L, Rr = -6.0, 6.0
        sxe = lambda v: L + (v - lo) / (hi - lo) * (Rr - L)
        ax = NumberLine(x_range=[lo, hi, 40], length=Rr - L, color=MUTED, include_numbers=False).move_to([0, -1.6, 0])
        nums = VGroup(*[T(str(v), 0.32, MUTED).move_to([sxe(v), -2.0, 0]) for v in range(60, 381, 80)])
        val = ValueTracker(37.0)

        def strip():
            A = X.copy(); A[4] = val.get_value(); g = VGroup(); cnt = {}
            for v in A:
                k = round(v); cnt[k] = cnt.get(k, 0) + 1
                g.add(Dot([sxe(v), -1.3 + 0.0 * cnt[k], 0], radius=0.11, color=POINT))
            m = np.mean(A); md = np.median(A)
            g.add(Line([sxe(m), -1.75, 0], [sxe(m), 0.2, 0], color=MEAN, stroke_width=5))
            g.add(T(f"media {m:.1f}".replace(".", ","), 0.42, MEAN, BOLD).move_to([sxe(m) + 0.95, 0.45, 0]))
            g.add(Line([sxe(md), -1.75, 0], [sxe(md), -0.5, 0], color=GOOD, stroke_width=5))
            g.add(T(f"mediana {md:.1f}".replace(".", ","), 0.42, GOOD, BOLD).move_to([sxe(md) + 0.6, -2.5, 0]))
            return g
        st = always_redraw(strip)
        self.mark("err"); self.play(Create(ax), FadeIn(nums), FadeIn(st), run_time=0.8); self.wait(0.6)
        self.mark("fly"); self.play(val.animate.set_value(370), run_time=2.2, rate_func=smooth)
        info = VGroup(T("media 33 → 74,6", 0.48, MEAN, BOLD), T("mediana 32,5 → 32,5", 0.48, GOOD, BOLD),
                      T("y la z del error es solo 2,47: con 8 datos no puede pasar de ahí", 0.38, MUTED)).arrange(DOWN, aligned_edge=LEFT, buff=0.2).move_to([1.5, 1.6, 0])
        self.play(FadeIn(info, shift=0.2 * UP), run_time=0.7); self.wait(3.6)
        st.clear_updaters()
        self.clear_but(cap)

        # ---------- 4. n − 1 ----------
        cap = self.swap(cap, self.caption("¿Por qué dividir entre n − 1?", "las distancias a la media de la propia muestra salen cortas"))
        lab = VGroup(T("varianza verdadera", 0.42, INK), T("media de las varianzas ÷ n", 0.42, INK), T("media de las varianzas ÷ (n − 1)", 0.42, INK))
        cols = [MEAN, SAMPLE, GOOD]; fr = [1.0, 2 / 3, 1.0]; txts = ["100 %", "67 %", "100 %"]
        bx = -1.0; bw = 5.0; grp = VGroup()
        for i in range(3):
            y = 0.6 - i * 1.15
            lab[i].next_to([bx - 0.2, y, 0], LEFT, buff=0)
            grp.add(Rectangle(width=bw, height=0.5, stroke_width=0, fill_color=LINE, fill_opacity=1).move_to([bx + bw / 2, y, 0]))
        vline = Line([bx + bw, 1.05, 0], [bx + bw, -2.05, 0], color=INK, stroke_width=4)
        bars = VGroup(*[Rectangle(width=bw * fr[i], height=0.5, stroke_width=0, fill_color=cols[i], fill_opacity=0.9).move_to([bx + bw * fr[i] / 2, 0.6 - i * 1.15, 0]) for i in range(3)])
        pct = VGroup(*[T(txts[i], 0.44, INK, BOLD).move_to([bx + bw + 0.75, 0.6 - i * 1.15, 0]) for i in range(3)])
        note = T("muestras de 3 viajes:  (n − 1) / n = 2/3", 0.42, MUTED).move_to([0, -2.75, 0])
        self.mark("bes"); self.play(FadeIn(lab), FadeIn(grp), Create(vline), run_time=0.8)
        for i in range(3):
            self.play(GrowFromEdge(bars[i], LEFT), FadeIn(pct[i]), run_time=0.7)
        self.play(FadeIn(note), run_time=0.5); self.wait(3.2)
        self.clear_but(cap)

        # ---------- 5. estandarizar ----------
        cap = self.swap(cap, self.caption("Estandarizar: z = (valor − media) / desviación típica", "dos columnas en unidades distintas, en la misma escala"))
        t = ValueTracker(0.0)
        mX, sX, mR, sR = 33, np.std(X, ddof=1), 18, np.std(R, ddof=1)
        rowsd = [("consumo", X, mX, sX, 28, 38, 0.8), ("ralentí", R, mR, sR, 0, 45, -1.4)]
        L2, R2 = -3.6, 5.6

        def zfig():
            g = VGroup(); tv = t.get_value()
            for name, A, m, s, lo, hi, y in rowsd:
                pos = lambda v, s=s, m=m, lo=lo, hi=hi: (L2 + (v - lo) / (hi - lo) * (R2 - L2)) * (1 - tv) + (L2 + ((v - m) / s + 2.5) / 5 * (R2 - L2)) * tv
                g.add(Line([L2, y, 0], [R2, y, 0], color=MUTED, stroke_width=2))
                g.add(Rectangle(width=pos(m + s) - pos(m - s), height=0.6, stroke_width=0, fill_color=MEAN, fill_opacity=0.18).move_to([(pos(m + s) + pos(m - s)) / 2, y + 0.2, 0]))
                g.add(Line([pos(m), y - 0.1, 0], [pos(m), y + 0.55, 0], color=MEAN, stroke_width=5))
                for i, v in enumerate(A):
                    g.add(Dot([pos(v), y + 0.22, 0], radius=0.12, color=POINT))
                    if i == 4:
                        g.add(Circle(0.22, color=INK, stroke_width=3).move_to([pos(v), y + 0.22, 0]))
                g.add(T(name, 0.46, INK, BOLD).move_to([-5.2, y + 0.2, 0]))
            return g
        zf = always_redraw(zfig)
        u1 = VGroup(T("L/100 km: 28 … 38", 0.34, MUTED).move_to([1.0, 0.35, 0]), T("minutos: 0 … 45", 0.34, MUTED).move_to([1.0, -1.85, 0]))
        self.mark("z"); self.play(FadeIn(zf), FadeIn(u1), run_time=0.8); self.wait(1.4)
        self.mark("zmove"); self.play(FadeOut(u1), t.animate.set_value(1.0), run_time=2.0, rate_func=smooth)
        zl = VGroup(*[T(s, 0.36, MUTED).move_to([L2 + (k + 2.5) / 5 * (R2 - L2), -2.3, 0]) for k, s in zip(range(-2, 3), ["−2", "−1", "0", "+1", "+2"])])
        zt = VGroup(T("viaje 5:  z = 1,76 en consumo,  2,13 en ralentí", 0.44, INK, BOLD), T("CV: consumo 6,9 %, ralentí 57 %", 0.4, MUTED)).arrange(DOWN, buff=0.18).move_to([0.6, -3.0, 0])
        self.play(FadeIn(zl), FadeIn(zt), run_time=0.6); self.wait(3.8)
        zf.clear_updaters()
        self.clear_but(cap)

        # ---------- Cierre ----------
        self.mark("legend")
        self.play(FadeOut(cap), run_time=0.6)
        e1 = Text("Centro y dispersión.", font=FONT, weight=BOLD, color=INK).scale(0.8)
        e2 = Text("Media y mediana juntas, cuadrados, raíz, n − 1 y z para comparar.", font=FONT, color=MUTED).scale(0.42)
        VGroup(e1, e2).arrange(DOWN, buff=0.4)
        self.mark("end_text"); self.play(Write(e1), run_time=1.3); self.play(FadeIn(e2, shift=0.2 * UP), run_time=0.8)
        self.wait(3.0)
        self.mark("fade"); self.play(FadeOut(VGroup(e1, e2)), run_time=1)
        self.marks["total"] = [round(self.renderer.time, 3)]
        json.dump(self.marks, open("marks.json", "w"), indent=1)
