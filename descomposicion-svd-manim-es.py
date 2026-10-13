import json
import numpy as np
from manim import *

# Paleta oscura de la serie
BG = "#0E131B"; INK = "#E4EAF3"; MUTED = "#9BA7B9"; LINE = "#2B3545"
MEAN = "#6EA6FF"; SAMPLE = "#AE9FF3"; POINT = "#FF9A55"; GOOD = "#4CC38A"; WARN = "#F08A5D"
FONT = "Inter"
config.background_color = BG

D = json.load(open("../mt/m6/embed.json"))
F = D["F"]; X = np.array(F["X"], float); U = np.array(F["U"]); S = np.array(F["s"]); Vt = np.array(F["Vt"])
CI = D["camion"]


def T(s, sc=0.45, col=INK, w=None):
    return Text(s, font=FONT, color=col, weight=w or NORMAL).scale(sc)


def recon(k):
    return (U[:, :k] * S[:k]) @ Vt[:k]


def heat(M, cs=0.27, div=False, vmax=680.0):
    g = VGroup()
    m, n = M.shape
    for i in range(m):
        for j in range(n):
            v = M[i, j]
            if div:
                op = min(1.0, abs(v) / 300.0); col = POINT if v >= 0 else SAMPLE
            else:
                op = max(0.0, min(1.0, v / vmax)); col = MEAN
            sq = Square(cs, stroke_width=0.6, stroke_color=LINE, fill_color=col, fill_opacity=0.04 + 0.96 * op)
            sq.move_to([j * cs, -i * cs, 0]); g.add(sq)
    return g.move_to(ORIGIN)


def img_mob(k, h=2.6):
    Ui = np.array(CI["U"]); si = np.array(CI["s"]); Vi = np.array(CI["Vt"])
    R = np.clip((Ui[:, :k] * si[:k]) @ Vi[:k], 0, 1)
    bg = np.array([22, 29, 40]); fg = np.array([228, 234, 243])
    arr = (bg[None, None, :] * (1 - R[:, :, None]) + fg[None, None, :] * R[:, :, None]).astype(np.uint8)
    im = ImageMobject(arr); im.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"]); im.height = h
    return im


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
        title = Text("Descomposición SVD", font=FONT, weight=BOLD, color=INK).scale(1.0)
        sub = Text("toda matriz es girar, estirar y girar", font=FONT, color=MUTED).scale(0.5)
        VGroup(title, sub).arrange(DOWN, buff=0.35)
        self.mark("title"); self.play(Write(title), run_time=1.3)
        self.mark("subtitle"); self.play(FadeIn(sub, shift=0.2 * UP)); self.wait(1.8)
        self.mark("title_out"); self.play(FadeOut(title), FadeOut(sub))

        # ---------- 1. girar, estirar, girar ----------
        cap = self.swap(None, self.caption("Toda matriz: girar, estirar y girar", "el círculo de flechas acaba siempre en una elipse"))
        O = np.array([-1.6, -0.75, 0]); u = 0.78
        th = np.arctan2(2, 1); ph = np.arctan2(1, 2)
        A = np.array([[2.0, 2.0], [-1.0, 2.0]])
        t = ValueTracker(0.0)

        def MT(T):
            Rot = lambda a: np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
            if T <= 1: return Rot(-th * T)
            if T <= 2:
                w = T - 1; return np.diag([1 + 2 * w, 1 + w]) @ Rot(-th)
            w = T - 2; return Rot(ph * w) @ np.diag([3.0, 2.0]) @ Rot(-th)

        P = lambda p: O + u * np.array([p[0], p[1], 0])
        ang = np.linspace(0, 2 * np.pi, 97)
        ghost = VMobject(stroke_color=MUTED, stroke_width=2).set_points_as_corners([P((np.cos(a), np.sin(a))) for a in ang])
        ghost = DashedVMobject(ghost, num_dashes=40)
        target = DashedVMobject(VMobject(stroke_color=POINT, stroke_width=3).set_points_as_corners([P(A @ (np.cos(a), np.sin(a))) for a in ang]), num_dashes=60)

        def fig():
            M = MT(t.get_value()); g = VGroup()
            g.add(VMobject(stroke_color=MEAN, stroke_width=2.5, fill_color=MEAN, fill_opacity=0.12).set_points_as_corners([P(M @ (np.cos(a), np.sin(a))) for a in ang]))
            for k in range(24):
                a = k / 24 * 2 * np.pi; q = M @ (np.cos(a), np.sin(a))
                g.add(Line(P((0, 0)), P(q), stroke_color=MUTED, stroke_width=1.5, stroke_opacity=0.6))
            v1 = np.array([np.cos(th), np.sin(th)]); v2 = np.array([-np.sin(th), np.cos(th)])
            g.add(Arrow(P((0, 0)), P(M @ v2), buff=0, color=SAMPLE, stroke_width=7, max_tip_length_to_length_ratio=0.18))
            g.add(Arrow(P((0, 0)), P(M @ v1), buff=0, color=MEAN, stroke_width=7, max_tip_length_to_length_ratio=0.12))
            return g
        f = always_redraw(fig)
        axes = VGroup(Line(O + 3.2 * LEFT, O + 3.2 * RIGHT, color=LINE), Line(O + 2.4 * DOWN, O + 2.4 * UP, color=LINE))
        step = T("0 · un círculo de flechas", 0.5, INK, BOLD).move_to([4.2, 1.4, 0])
        self.play(Create(axes), Create(ghost), FadeIn(f), FadeIn(step), run_time=1.0)
        self.play(Create(target), run_time=0.8); self.wait(1.0)
        for name, lab, T1 in [("rot1", "1 · girar −63,4°", 1), ("stretch", "2 · estirar × 3 y × 2", 2), ("rot2", "3 · girar +26,6°", 3)]:
            new = T(lab, 0.5, INK, BOLD).move_to([4.2, 1.4, 0])
            self.mark(name); self.play(Transform(step, new), run_time=0.4); self.play(t.animate.set_value(T1), run_time=1.6, rate_func=smooth); self.wait(0.8)
        nt = T("= lo que hace la matriz", 0.45, POINT, BOLD).move_to([4.2, 0.7, 0])
        self.play(FadeIn(nt), run_time=0.5); self.wait(2.4)
        f.clear_updaters()
        self.clear_but(cap)

        # ---------- 2. ejes de la elipse ----------
        cap = self.swap(cap, self.caption("Los valores singulares son los ejes de la elipse", "lo máximo y lo mínimo que estira la matriz"))
        O2 = np.array([-2.0, -0.8, 0]); u2 = 0.8
        e = Ellipse(width=6 * u2, height=4 * u2, color=MEAN, fill_opacity=0.12).rotate(ph).move_to(O2)
        c = DashedVMobject(Circle(radius=u2, color=MUTED), num_dashes=30).move_to(O2)
        a1 = Arrow(O2, O2 + 3 * u2 * np.array([np.cos(ph), np.sin(ph), 0]), buff=0, color=MEAN, stroke_width=7, max_tip_length_to_length_ratio=0.1)
        a2 = Arrow(O2, O2 + 2 * u2 * np.array([-np.sin(ph), np.cos(ph), 0]), buff=0, color=SAMPLE, stroke_width=7, max_tip_length_to_length_ratio=0.14)
        self.mark("axes"); self.play(FadeIn(c), DrawBorderThenFill(e), run_time=1.0)
        self.play(GrowArrow(a1), GrowArrow(a2), run_time=0.8)
        info = VGroup(T("σ₁ = 3   (eje largo)", 0.55, MEAN, BOLD), T("σ₂ = 2   (eje corto)", 0.55, SAMPLE, BOLD),
                      T("área: 3 × 2 = 6", 0.5, INK), T("deshacer agranda errores", 0.42, MUTED), T("hasta σ₁ / σ₂ = 1,5 veces", 0.42, MUTED)).arrange(DOWN, aligned_edge=LEFT, buff=0.25).move_to([3.6, -0.6, 0])
        self.mark("sig"); self.play(FadeIn(info[:2], shift=0.2 * RIGHT), run_time=0.7); self.wait(0.6)
        self.play(FadeIn(info[2:], shift=0.2 * RIGHT), run_time=0.7); self.wait(2.6)
        self.clear_but(cap)

        # ---------- 3. capas ----------
        cap = self.swap(cap, self.caption("Una tabla es una suma de capas", "km por día de 12 camiones durante dos semanas"))
        hd = heat(X).move_to([-3.4, -0.7, 0]); ld = T("datos", 0.42, MUTED).next_to(hd, DOWN, buff=0.15)
        self.mark("heat"); self.play(FadeIn(hd, lag_ratio=0.002), FadeIn(ld), run_time=1.2)
        hk = heat(recon(1)).move_to([1.4, -0.7, 0]); lk = T("1 capa: rutina × intensidad", 0.42, MEAN).next_to(hk, DOWN, buff=0.15)
        num = T("error medio 36,4 km", 0.45, INK, BOLD).move_to([4.95, -0.4, 0])
        self.mark("k1"); self.play(FadeIn(hk, lag_ratio=0.002), FadeIn(lk), FadeIn(num), run_time=1.2); self.wait(2.0)
        hk2 = heat(recon(2)).move_to([1.4, -0.7, 0]); lk2 = T("2 capas: + reparto en sábado", 0.42, MEAN).next_to(hk2, DOWN, buff=0.15)
        num2 = T("error medio 22,7 km", 0.45, GOOD, BOLD).move_to([4.95, -0.4, 0])
        self.mark("k2"); self.play(Transform(hk, hk2), Transform(lk, lk2), Transform(num, num2), run_time=1.0); self.wait(2.6)
        self.clear_but(cap)

        # ---------- 4. comprimir ----------
        cap = self.swap(cap, self.caption("Comprimir: guardar solo las primeras capas", "un dibujo de 48 × 80 = 3 840 números"))
        orig = img_mob(30).move_to([-3.2, -0.6, 0]); lo = T("original", 0.42, MUTED).next_to(orig, DOWN, buff=0.15)
        self.mark("img"); self.play(FadeIn(orig), FadeIn(lo), run_time=0.8)
        cur = img_mob(1).move_to([2.6, -0.6, 0]); lc = T("1 capa · 129 números", 0.42, MEAN, BOLD).next_to(cur, DOWN, buff=0.15)
        self.play(FadeIn(cur), FadeIn(lc), run_time=0.6)
        for k in (3, 5, 12):
            nm = img_mob(k).move_to([2.6, -0.6, 0]); nl = T(f"{k} capas · {k * 129} números", 0.42, MEAN, BOLD).next_to(nm, DOWN, buff=0.15)
            self.mark("kimg"); self.play(FadeOut(cur), FadeIn(nm), Transform(lc, nl), run_time=0.6); cur = nm; self.wait(1.4)
        self.wait(1.0)
        self.clear_but(cap)

        # ---------- 5. lo raro ----------
        cap = self.swap(cap, self.caption("Lo que las capas no explican: lo raro", "datos menos 2 capas"))
        R = X - recon(2); hr = heat(R, cs=0.36, div=True).move_to([-1.4, -0.75, 0])
        self.mark("resid"); self.play(FadeIn(hr, lag_ratio=0.002), run_time=1.2); self.wait(0.4)
        def cell(i, j): return hr[i * 14 + j].get_center()
        r1 = Circle(0.3, color=INK, stroke_width=5).move_to(cell(3, 10)); r2 = Circle(0.3, color=INK, stroke_width=5).move_to(cell(10, 6))
        t1 = VGroup(T("camión 4, jueves:", 0.42, SAMPLE, BOLD), T("363 km menos (¿avería?)", 0.42, INK)).arrange(DOWN, aligned_edge=LEFT, buff=0.1).move_to([4.6, 0.6, 0])
        t2 = VGroup(T("camión 11, domingo:", 0.42, POINT, BOLD), T("sale sin estar previsto", 0.42, INK)).arrange(DOWN, aligned_edge=LEFT, buff=0.1).move_to([4.6, -1.0, 0])
        self.mark("rings"); self.play(Create(r1), FadeIn(t1), run_time=0.7); self.play(Create(r2), FadeIn(t2), run_time=0.7)
        t3 = T("el resto: ± 12 km de cada día", 0.4, MUTED).move_to([4.6, -2.3, 0])
        self.play(FadeIn(t3), run_time=0.5); self.wait(3.0)
        self.clear_but(cap)

        # ---------- Cierre ----------
        self.mark("legend")
        self.play(FadeOut(cap), run_time=0.6)
        e1 = Text("Girar, estirar, girar.", font=FONT, weight=BOLD, color=INK).scale(0.8)
        e2 = Text("Y una tabla de datos, una suma de capas ordenadas de la que más pesa a la que menos.", font=FONT, color=MUTED).scale(0.4)
        VGroup(e1, e2).arrange(DOWN, buff=0.4)
        self.mark("end_text"); self.play(Write(e1), run_time=1.3); self.play(FadeIn(e2, shift=0.2 * UP), run_time=0.8)
        self.wait(3.0)
        self.mark("fade"); self.play(FadeOut(VGroup(e1, e2)), run_time=1)
        self.marks["total"] = [round(self.renderer.time, 3)]
        json.dump(self.marks, open("marks.json", "w"), indent=1)
