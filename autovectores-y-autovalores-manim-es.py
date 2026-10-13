import json
import numpy as np
from manim import *

# Paleta oscura de la serie
BG = "#0E131B"; INK = "#E4EAF3"; MUTED = "#9BA7B9"; LINE = "#2B3545"
MEAN = "#6EA6FF"; SAMPLE = "#AE9FF3"; POINT = "#FF9A55"; GOOD = "#4CC38A"; WARN = "#F08A5D"
FONT = "Inter"
config.background_color = BG


def T(s, sc=0.45, col=INK, w=None):
    return Text(s, font=FONT, color=col, weight=w or NORMAL).scale(sc)


def M3(M):
    return np.array([[M[0][0], M[0][1], 0], [M[1][0], M[1][1], 0], [0, 0, 1]])


def fan(O, s, n=16, col=MEAN, w=4):
    return VGroup(*[Arrow(O, O + s * np.array([np.cos(2 * np.pi * k / n), np.sin(2 * np.pi * k / n), 0]), buff=0, color=col,
                          stroke_width=w, max_tip_length_to_length_ratio=0.22, max_stroke_width_to_length_ratio=20) for k in range(n)])


def spokes(O, s, n=16):
    return VGroup(*[Line(O, O + s * np.array([np.cos(2 * np.pi * k / n), np.sin(2 * np.pi * k / n), 0]), color=MUTED, stroke_width=2.5, stroke_opacity=0.7) for k in range(n)])


def mulberry32(a):
    a &= 0xFFFFFFFF
    def r():
        nonlocal a
        a = (a + 0x6D2B79F5) & 0xFFFFFFFF
        t = ((a ^ (a >> 15)) * (1 | a)) & 0xFFFFFFFF
        t = (t + (((t ^ (t >> 7)) * (61 | t)) & 0xFFFFFFFF)) & 0xFFFFFFFF ^ t
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296
    return r


def cloud_js(n=180, seed=11):
    r = mulberry32(seed); out = []
    for _ in range(2 * n):
        u = 0.0
        while u == 0: u = r()
        v = r(); out.append(np.sqrt(-2 * np.log(u)) * np.cos(2 * np.pi * v))
    return np.array(out).reshape(n, 2)


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
        title = Text("Autovectores y autovalores", font=FONT, weight=BOLD, color=INK).scale(0.95)
        sub = Text("las direcciones que una matriz no gira", font=FONT, color=MUTED).scale(0.5)
        VGroup(title, sub).arrange(DOWN, buff=0.35)
        self.mark("title"); self.play(Write(title), run_time=1.3)
        self.mark("subtitle"); self.play(FadeIn(sub, shift=0.2 * UP)); self.wait(1.3)
        self.mark("title_out"); self.play(FadeOut(title), FadeOut(sub))

        # ---------- 1. el abanico ----------
        cap = self.swap(None, self.caption("Casi todas las flechas giran", "unas pocas solo se estiran: son los autovectores"))
        O = np.array([-2.6, -0.85, 0]); s = 0.85
        A = [[2, 1], [1, 2]]
        sp = spokes(O, s); fa = fan(O, s)
        circ = DashedVMobject(Circle(radius=s, color=MUTED, stroke_width=2).move_to(O), num_dashes=40)
        mg = VGroup(*[T(str(A[i][j]), 0.6, MEAN) for i in range(2) for j in range(2)]).arrange_in_grid(2, 2, buff=(0.45, 0.2)).move_to([4.0, 1.4, 0])
        br = VGroup(T("(", 1.9).next_to(mg, LEFT, buff=0.1), T(")", 1.9).next_to(mg, RIGHT, buff=0.1))
        mt = VGroup(T("A =", 0.6).next_to(br[0], LEFT, buff=0.2), mg)
        self.play(FadeIn(sp), FadeIn(circ), FadeIn(fa), FadeIn(mt), FadeIn(br), run_time=1.0)
        self.mark("apply")
        self.play(ApplyMatrix(M3(A), fa, about_point=O), run_time=2.2)
        d1 = np.array([1, 1, 0]) / np.sqrt(2); d2 = np.array([-1, 1, 0]) / np.sqrt(2)
        l1 = DashedLine(O - 3.3 * s * d1, O + 3.6 * s * d1, color=POINT, stroke_width=4, dash_length=0.15)
        l2 = DashedLine(O - 2.2 * s * d2, O + 2.2 * s * d2, color=SAMPLE, stroke_width=4, dash_length=0.15)
        a1 = Arrow(O, O + 3 * s * d1, buff=0, color=POINT, stroke_width=10, max_tip_length_to_length_ratio=0.12)
        a2 = Arrow(O, O + 1 * s * d2, buff=0, color=SAMPLE, stroke_width=10, max_tip_length_to_length_ratio=0.3)
        t1 = T("× 3", 0.7, POINT, BOLD).next_to(a1.get_end(), RIGHT, buff=0.15)
        t2 = T("× 1", 0.7, SAMPLE, BOLD).next_to(a2.get_end(), LEFT, buff=0.15)
        info = VGroup(T("45°: se estira × 3", 0.48, POINT), T("135°: se queda igual (× 1)", 0.48, SAMPLE), T("las demás giran", 0.48, MUTED)).arrange(DOWN, aligned_edge=LEFT, buff=0.2).move_to([3.6, -1.2, 0])
        self.mark("highlight")
        self.play(fa.animate.set_opacity(0.35), Create(l1), Create(l2), run_time=0.8)
        self.play(GrowArrow(a1), GrowArrow(a2), FadeIn(t1), FadeIn(t2), FadeIn(info, lag_ratio=0.3), run_time=1.0)
        self.wait(3.6)
        self.clear_but(cap)

        # ---------- 2. un giro no tiene ninguna ----------
        cap = self.swap(cap, self.caption("Un giro no tiene ninguna", "una matriz simétrica siempre tiene dos, en ángulo recto"))
        O2 = np.array([-3.4, -0.85, 0]); s2 = 1.3
        R = [[0.8, -0.6], [0.6, 0.8]]
        sp2 = spokes(O2, s2); fa2 = fan(O2, s2, col=MEAN, w=4)
        self.play(FadeIn(sp2), FadeIn(fa2), run_time=0.6)
        self.mark("rot")
        self.play(ApplyMatrix(M3(R), fa2, about_point=O2), run_time=1.6)
        lab = VGroup(T("girar 37°", 0.55, INK, BOLD), T("todas giran lo mismo:", 0.45, MUTED), T("ninguna dirección propia", 0.45, WARN)).arrange(DOWN, aligned_edge=LEFT, buff=0.2).move_to([-0.0, -0.4, 0]).align_to([-1.0, 0, 0], LEFT)
        self.play(FadeIn(lab), run_time=0.5)
        Os = np.array([4.3, -0.85, 0])
        sq = VGroup(Line(Os - 1.5 * d1, Os + 1.5 * d1, color=POINT, stroke_width=4), Line(Os - 1.5 * d2, Os + 1.5 * d2, color=SAMPLE, stroke_width=4))
        ra = VMobject(color=INK, stroke_width=2).set_points_as_corners([Os + 0.28 * d1, Os + 0.28 * (d1 + d2), Os + 0.28 * d2])
        sl = T("simétrica: dos, a 90°", 0.45, GOOD).next_to(sq, DOWN, buff=0.35)
        self.mark("sym"); self.play(Create(sq), FadeIn(ra), FadeIn(sl), run_time=0.8)
        self.wait(3.4)
        self.clear_but(cap)

        # ---------- 3. método de la potencia ----------
        cap = self.swap(cap, self.caption("Aplicar la matriz una y otra vez", "cualquier flecha acaba en la dirección del autovalor mayor"))
        O3 = np.array([-3.3, -2.2, 0]); s3 = 3.2
        l13 = DashedLine(O3 + 0.0 * d1, O3 + 1.35 * s3 * d1, color=POINT, stroke_width=4, dash_length=0.15)
        arc = Arc(radius=s3, start_angle=0, angle=PI / 2, arc_center=O3, color=LINE, stroke_width=2)
        ax = VGroup(Line(O3, O3 + RIGHT * (s3 + 0.6), color=MUTED, stroke_opacity=0.5), Line(O3, O3 + UP * (s3 + 0.4), color=MUTED, stroke_opacity=0.5))
        self.play(FadeIn(ax), FadeIn(arc), Create(l13), run_time=0.6)
        vecs = [(1, 0), (2, 1), (5, 4), (14, 13), (41, 40)]
        cur = Arrow(O3, O3 + s3 * RIGHT, buff=0, color=MUTED, stroke_width=8, max_tip_length_to_length_ratio=0.08)
        self.play(GrowArrow(cur), run_time=0.4)
        rows = VGroup()
        for k, (x, y) in enumerate(vecs):
            ang = np.degrees(np.arctan2(y, x))
            r = VGroup(T(f"paso {k}", 0.42, MUTED), T(f"({x}, {y})", 0.46, INK), T(f"{ang:.1f}°".replace(".", ","), 0.46, POINT if k == 4 else INK)).arrange(RIGHT, buff=0.35)
            rows.add(r)
        rows.arrange(DOWN, aligned_edge=LEFT, buff=0.22).move_to([3.7, -0.6, 0])
        self.play(FadeIn(rows[0]), run_time=0.3)
        self.mark("steps")
        trail = VGroup()
        for k in range(1, 5):
            x, y = vecs[k]; u = np.array([x, y, 0]) / np.hypot(x, y)
            new = Arrow(O3, O3 + s3 * u, buff=0, color=MEAN, stroke_width=8, max_tip_length_to_length_ratio=0.08)
            old = cur.copy().set_opacity(0.3); trail.add(old); self.add(old)
            self.play(Transform(cur, new), FadeIn(rows[k]), run_time=0.75)
        nt = T("A = (2 1; 1 2): autovalores 3 y 1", 0.42, MUTED).next_to(rows, DOWN, buff=0.4)
        self.play(FadeIn(nt), run_time=0.4)
        self.wait(3.2)
        self.clear_but(cap)

        # ---------- 4. Markov ----------
        cap = self.swap(cap, self.caption("La flota llega a su reparto estacionario", "el autovector de autovalor 1 de la tabla de cambios"))
        P = np.array([[0.90, 0.03, 0.07], [0.30, 0.60, 0.10], [0.50, 0.02, 0.48]])
        pi = np.array([0.8110, 0.0669, 0.1220])
        p = np.array([0.0, 1.0, 0.0]); days = [p]
        for _ in range(14):
            p = p @ P; days.append(p)
        x0, y0, Hh, bw = -6.2, -3.3, 4.4, 0.52
        cols = [MEAN, POINT, SAMPLE]
        axis = Line([x0 - 0.1, y0, 0], [x0 + 15 * bw + 0.1, y0, 0], color=MUTED, stroke_width=2)
        dl = VGroup(*[DashedLine([x0, y0 + Hh * c, 0], [x0 + 15 * bw, y0 + Hh * c, 0], color=INK, stroke_width=2, dash_length=0.1) for c in (pi[0], pi[0] + pi[1])])
        leg = VGroup(*[VGroup(Square(0.22, fill_color=c, fill_opacity=1, stroke_width=0), T(n, 0.42, INK)).arrange(RIGHT, buff=0.15) for c, n in zip(cols, ["en ruta", "en taller", "parado"])]).arrange(DOWN, aligned_edge=LEFT, buff=0.25).move_to([3.9, 0.9, 0])
        d0 = T("día 0: todos en el taller", 0.45, MUTED).move_to([3.9, -0.5, 0])
        self.play(Create(axis), FadeIn(leg), FadeIn(d0), run_time=0.6)
        bars = VGroup()
        self.mark("bars")
        for t, q in enumerate(days):
            acc = 0; b = VGroup()
            for j in range(3):
                h = Hh * q[j]
                if h > 1e-3:
                    b.add(Rectangle(width=bw * 0.82, height=h, stroke_width=0, fill_color=cols[j], fill_opacity=0.9).move_to([x0 + (t + 0.5) * bw, y0 + Hh * acc + h / 2, 0]))
                acc += q[j]
            bars.add(b)
            self.play(FadeIn(b, shift=0.1 * UP), run_time=0.22 if t else 0.4)
        res = VGroup(T("final: 81 % · 7 % · 12 %", 0.5, INK, BOLD), T("empiece donde empiece", 0.45, GOOD)).arrange(DOWN, aligned_edge=LEFT, buff=0.15).move_to([3.9, -1.6, 0])
        self.mark("settle"); self.play(Create(dl), FadeIn(res), run_time=0.7)
        self.wait(3.2)
        self.clear_but(cap)

        # ---------- 5. PCA ----------
        cap = self.swap(cap, self.caption("Los ejes de una nube de datos", "autovectores de la covarianza: eso es PCA"))
        Z = cloud_js()   # mismos 180 puntos que la idea 4 de la versión sencilla (mulberry32(11) + Box-Muller)
        th = np.radians(40); c, s_ = np.cos(th), np.sin(th)
        Pt = Z @ np.diag([2.4, 0.8]) @ np.array([[c, s_], [-s_, c]])
        O5 = np.array([-2.6, -0.9, 0]); k5 = 0.42
        C = np.cov(Pt.T); w, V = np.linalg.eigh(C); w, V = w[::-1], V[:, ::-1]
        if V[1, 0] < 0: V[:, 0] *= -1
        dots = VGroup(*[Dot(O5 + k5 * np.array([a, b, 0]), radius=0.045, color=MUTED, fill_opacity=0.75) for a, b in Pt])
        axs = VGroup(Line(O5 + LEFT * 3.3, O5 + RIGHT * 3.3, color=MUTED, stroke_opacity=0.4), Line(O5 + DOWN * 2.5, O5 + UP * 2.6, color=MUTED, stroke_opacity=0.4))
        axl = VGroup(T("motor (°C)", 0.36, MUTED).next_to(O5 + RIGHT * 3.3, DOWN, buff=0.1), T("aceite (°C)", 0.36, MUTED).next_to(O5 + UP * 2.6, RIGHT, buff=0.1))
        self.mark("cloud"); self.play(FadeIn(axs), FadeIn(axl), FadeIn(dots, lag_ratio=0.01), run_time=1.0)
        ang = np.arctan2(V[1, 0], V[0, 0])
        ell = Ellipse(width=2 * 2.45 * np.sqrt(w[0]) * k5, height=2 * 2.45 * np.sqrt(w[1]) * k5, color=MEAN, stroke_width=3).rotate(ang).move_to(O5)
        v1 = np.array([V[0, 0], V[1, 0], 0]); v2 = np.array([-V[1, 0], V[0, 0], 0])
        e1 = Arrow(O5, O5 + 2 * np.sqrt(w[0]) * k5 * v1, buff=0, color=POINT, stroke_width=9, max_tip_length_to_length_ratio=0.15)
        e2 = Arrow(O5, O5 + 2 * np.sqrt(w[1]) * k5 * v2, buff=0, color=SAMPLE, stroke_width=9, max_tip_length_to_length_ratio=0.35)
        pct = 100 * w[0] / w.sum()
        info = VGroup(T("eje largo:", 0.45, MUTED), T(f"{pct:.0f} % de la variación", 0.55, POINT, BOLD), T("eje corto: lo raro", 0.45, SAMPLE)).arrange(DOWN, aligned_edge=LEFT, buff=0.2).move_to([3.9, -0.6, 0])
        self.mark("axes"); self.play(Create(ell), GrowArrow(e1), GrowArrow(e2), run_time=1.0); self.play(FadeIn(info), run_time=0.5)
        self.wait(3.4)
        self.clear_but(cap)

        # ---------- 6. estabilidad ----------
        cap = self.swap(cap, self.caption("Tamaño menor que 1: se calma", "mayor que 1: se descontrola"))
        def curve(a, b, x0, w, h, col, ymax=25):
            M = np.array([[1 - a, 1 - a], [-b, 1 - b]]); e = np.array([20.0, 0]); pts = []
            for t in range(31):
                y = np.clip(e[0], -ymax, ymax); pts.append([x0 + t / 30 * w, y / ymax * h / 2, 0]); e = M @ e
            return VMobject(color=col, stroke_width=5).set_points_as_corners(pts)
        panels = VGroup()
        for i, (a, b, nm, col, rho) in enumerate([(0.5, 0.2, "filtro equilibrado", GOOD, "0,71"), (1.5, 1.2, "filtro pasado de vueltas", WARN, "1,14")]):
            x0 = -6.3 + i * 6.5; yc = -0.9
            box = RoundedRectangle(corner_radius=0.15, width=5.9, height=4.4, stroke_color=LINE, stroke_width=2).move_to([x0 + 2.85, yc, 0])
            zl = Line([x0 + 0.2, yc, 0], [x0 + 5.5, yc, 0], color=MUTED, stroke_opacity=0.5)
            cv = curve(a, b, x0 + 0.25, 5.2, 3.4, col).shift(yc * UP)
            ttl = T(nm, 0.45, INK, BOLD).next_to(box, UP, buff=0.12)
            rl = T(f"mayor autovalor: {rho}", 0.42, col).next_to(box, DOWN, buff=0.12)
            panels.add(VGroup(box, zl, ttl, rl, cv))
        self.play(FadeIn(VGroup(*[VGroup(p[0], p[1], p[2]) for p in panels])), run_time=0.6)
        self.mark("calm"); self.play(Create(panels[0][4]), run_time=1.6); self.play(FadeIn(panels[0][3]), run_time=0.4)
        self.mark("explode"); self.play(Create(panels[1][4]), run_time=1.6); self.play(FadeIn(panels[1][3]), run_time=0.4)
        boom = T("a los 30 s: 590 m de error", 0.42, WARN, BOLD).move_to(panels[1][0].get_top() + 0.35 * DOWN).set_z_index(3)
        bg = BackgroundRectangle(boom, color=BG, fill_opacity=0.9, buff=0.08)
        self.play(FadeIn(bg), FadeIn(boom), run_time=0.4)
        self.wait(3.2)
        self.clear_but(cap)

        # ---------- Cierre ----------
        self.mark("legend")
        self.play(FadeOut(cap), run_time=0.6)
        e1 = Text("A·v = λ·v", font=FONT, weight=BOLD, color=INK).scale(0.95)
        e2 = Text("Las direcciones que no giran deciden qué pasa a la larga.", font=FONT, color=MUTED).scale(0.45)
        VGroup(e1, e2).arrange(DOWN, buff=0.4)
        self.mark("end_text"); self.play(Write(e1), run_time=1.2); self.play(FadeIn(e2, shift=0.2 * UP), run_time=0.8)
        self.wait(2.4)
        self.mark("fade"); self.play(FadeOut(VGroup(e1, e2)), run_time=1)
        self.marks["total"] = [round(self.renderer.time, 3)]
        json.dump(self.marks, open("marks.json", "w"), indent=1)
