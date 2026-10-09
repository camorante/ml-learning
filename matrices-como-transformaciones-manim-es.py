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


def circ(cx, cy, r, n=16):
    return [[cx + r * np.cos(2 * np.pi * k / n), cy + r * np.sin(2 * np.pi * k / n)] for k in range(n)]


TRUCK = {"body": [[-1.7, 0.35], [0.55, 0.35], [0.55, 1.6], [-1.7, 1.6]],
         "cab": [[0.65, 0.35], [1.45, 0.35], [1.45, 0.85], [1.18, 1.22], [0.65, 1.22]],
         "win": [[0.92, 0.82], [1.3, 0.82], [1.12, 1.08], [0.92, 1.08]],
         "wheels": [circ(-1.25, 0.3, 0.27), circ(-0.6, 0.3, 0.27), circ(1.05, 0.3, 0.27)]}


def truck(o, s, dy=0.0, ghost=False):
    """Camión en coordenadas de un plano con origen o (punto de la escena) y escala s."""
    P = lambda pts: [o + s * np.array([x, y + dy, 0]) for x, y in pts]
    if ghost:
        kw = dict(stroke_color=MUTED, stroke_width=2, fill_opacity=0)
        return VGroup(*[DashedVMobject(Polygon(*P(p), **kw), num_dashes=14) for p in [TRUCK["body"], TRUCK["cab"]] + TRUCK["wheels"]])
    g = VGroup(Polygon(*P(TRUCK["body"]), stroke_color=POINT, stroke_width=3, fill_color=POINT, fill_opacity=0.3),
               Polygon(*P(TRUCK["cab"]), stroke_color=POINT, stroke_width=3, fill_color=POINT, fill_opacity=0.55),
               Polygon(*P(TRUCK["win"]), stroke_color=POINT, stroke_width=1.5, fill_color=BG, fill_opacity=1),
               *[Polygon(*P(w), stroke_color=BG, stroke_width=2, fill_color=INK, fill_opacity=1) for w in TRUCK["wheels"]])
    return g


def M3(M):
    return np.array([[M[0][0], M[0][1], 0], [M[1][0], M[1][1], 0], [0, 0, 1]])


def matrix_tex(M, cols=(MEAN, SAMPLE), sc=0.6):
    f = lambda v: (f"{v:g}").replace("-", "−").replace(".", ",")
    m = VGroup(*[T(f(M[i][j]), sc, cols[j]) for i in range(2) for j in range(2)]).arrange_in_grid(2, 2, buff=(0.45, 0.2))
    br = VGroup(Text("(", font=FONT, color=INK).scale(sc * 3.1).next_to(m, LEFT, buff=0.12), Text(")", font=FONT, color=INK).scale(sc * 3.1).next_to(m, RIGHT, buff=0.12))
    return VGroup(br, m)


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
        title = Text("Matrices como transformaciones", font=FONT, weight=BOLD, color=INK).scale(0.95)
        sub = Text("girar, estirar y aplastar el plano", font=FONT, color=MUTED).scale(0.5)
        VGroup(title, sub).arrange(DOWN, buff=0.35)
        self.mark("title"); self.play(Write(title), run_time=1.3)
        self.mark("subtitle"); self.play(FadeIn(sub, shift=0.2 * UP)); self.wait(1.3)
        self.mark("title_out"); self.play(FadeOut(title), FadeOut(sub))

        # ---------- 1. una matriz mueve todo el plano ----------
        cap = self.swap(None, self.caption("Una matriz mueve todo el plano", "sus columnas dicen dónde acaban  →  y  ↑"))
        s = 0.55; O = np.array([-2.6, -1.5, 0])
        bg = NumberPlane(x_range=[-7, 7, 1], y_range=[-5, 5, 1], x_length=14 * s, y_length=10 * s,
                         background_line_style={"stroke_color": MUTED, "stroke_opacity": 0.18, "stroke_width": 1}, axis_config={"stroke_opacity": 0}).move_to(O)
        pl = NumberPlane(x_range=[-12, 12, 1], y_range=[-12, 12, 1], x_length=24 * s, y_length=24 * s,
                         background_line_style={"stroke_color": MEAN, "stroke_opacity": 0.35, "stroke_width": 1.4},
                         axis_config={"stroke_color": MEAN, "stroke_opacity": 0.7}).move_to(O)
        x0, x1, y1 = O[0] - 7 * s, O[0] + 7 * s, O[1] + 5 * s
        mk = lambda a, b, c, d: Rectangle(width=b - a, height=d - c, stroke_width=0, fill_color=BG, fill_opacity=1).move_to([(a + b) / 2, (c + d) / 2, 0]).set_z_index(5)
        masks = VGroup(mk(x1, 8, -5, 5), mk(-8, x0, -5, 5), mk(-8, 8, y1 + 0.75, 5)); self.add(masks)
        tr = truck(O, s, dy=0.3); gh = truck(O, s, dy=0.3, ghost=True)
        e1 = Arrow(O, O + s * RIGHT, buff=0, color=MEAN, stroke_width=8, max_tip_length_to_length_ratio=0.35)
        e2 = Arrow(O, O + s * UP, buff=0, color=SAMPLE, stroke_width=8, max_tip_length_to_length_ratio=0.35)
        self.mark("grid"); self.play(FadeIn(bg), Create(pl), FadeIn(tr), GrowArrow(e1), GrowArrow(e2), run_time=1.0)
        self.add(gh)
        A = [[2, -1], [1, 1]]
        mt = matrix_tex(A, sc=0.75).move_to([4.1, 1.0, 0]).set_z_index(6)
        lab = VGroup(T("→ acaba en (2, 1)", 0.45, MEAN), T("↑ acaba en (−1, 1)", 0.45, SAMPLE)).arrange(DOWN, aligned_edge=LEFT, buff=0.18).next_to(mt, DOWN, buff=0.4).set_z_index(6)
        self.play(FadeIn(mt), run_time=0.5)
        self.mark("apply")
        self.play(ApplyMatrix(M3(A), pl, about_point=O), ApplyMatrix(M3(A), tr, about_point=O), ApplyMatrix(M3(A), e1, about_point=O), ApplyMatrix(M3(A), e2, about_point=O), run_time=2.2)
        self.play(FadeIn(lab, shift=0.2 * UP), run_time=0.6)
        # punto (3, 2) -> 3·col1 + 2·col2 = (4, 5)
        c1 = O + s * np.array([2, 1, 0]); a1 = Arrow(O, O + 3 * s * np.array([2, 1, 0]), buff=0, color=MEAN, stroke_width=7, max_tip_length_to_length_ratio=0.08)
        a2 = Arrow(a1.get_end(), O + s * np.array([4, 5, 0]), buff=0, color=SAMPLE, stroke_width=7, max_tip_length_to_length_ratio=0.15)
        dot = Dot(O + s * np.array([4, 5, 0]), radius=0.11, color=POINT)
        eq = VGroup(T("(3, 2)  →", 0.45, INK), T("3·(2, 1) + 2·(−1, 1)", 0.45, INK), T("= (4, 5)", 0.55, POINT, BOLD)).arrange(DOWN, aligned_edge=LEFT, buff=0.15).move_to([4.2, -1.9, 0]).set_z_index(6)
        self.mark("point"); self.play(GrowArrow(a1), run_time=0.7); self.play(GrowArrow(a2), run_time=0.6); self.play(FadeIn(dot, scale=1.5), FadeIn(eq), run_time=0.6)
        self.wait(3.0)
        self.clear_but(cap)

        # ---------- 2. catálogo ----------
        cap = self.swap(cap, self.caption("Girar, estirar, reflejar, inclinar", "cada una conserva cosas distintas"))
        kinds = [("girar 37°", [[0.8, -0.6], [0.6, 0.8]], "área × 1"), ("estirar × 2", [[2, 0], [0, 1]], "área × 2"),
                 ("reflejar", [[-1, 0], [0, 1]], "área × −1"), ("inclinar", [[1, 1], [0, 1]], "área × 1")]
        panels = VGroup(); trucks = []
        for k, (name, M, ar) in enumerate(kinds):
            c = np.array([-4.9 + k * 3.25, -0.6, 0]); s2 = 0.42
            box = RoundedRectangle(corner_radius=0.15, width=3.0, height=3.6, stroke_color=LINE, stroke_width=2).move_to(c)
            o = c + np.array([0, -0.6, 0])
            ax = VGroup(Line(o + LEFT * 1.4, o + RIGHT * 1.4, stroke_color=MUTED, stroke_opacity=0.4, stroke_width=1.5), Line(o + DOWN * 1.0, o + UP * 2.0, stroke_color=MUTED, stroke_opacity=0.4, stroke_width=1.5))
            t0 = truck(o, s2)
            nm = T(name, 0.45, INK, BOLD).next_to(box, UP, buff=0.15); arl = T(ar, 0.42, WARN if "−" in ar else GOOD).next_to(box, DOWN, buff=0.15)
            panels.add(VGroup(box, ax, nm)); trucks.append((t0, M, o, arl))
        self.play(FadeIn(panels), *[FadeIn(t[0]) for t in trucks], run_time=0.8)
        self.mark("kinds")
        for t0, M, o, arl in trucks:
            self.play(ApplyMatrix(M3(M), t0, about_point=o), FadeIn(arl), run_time=0.9)
        self.wait(2.6)
        self.clear_but(cap)

        # ---------- 3. el orden importa ----------
        cap = self.swap(cap, self.caption("Encadenar es multiplicar: el orden importa", "girar 90° y estirar × 2 a lo ancho"))
        R = [[0, -1], [1, 0]]; E = [[2, 0], [0, 1]]
        rows = [("primero girar, luego estirar", R, E, "E·R"), ("primero estirar, luego girar", E, R, "R·E")]
        s3 = 0.33
        for r, (txt, M1, M2, nm) in enumerate(rows):
            y = 0.55 - r * 2.55
            lbl = T(txt, 0.42, MUTED).move_to([-4.6, y + 0.95, 0]).align_to([-6.6, 0, 0], LEFT)
            os_ = [np.array([-5.0 + i * 3.4, y - 0.35, 0]) for i in range(3)]
            t1 = truck(os_[0], s3); arr1 = T("→", 0.7, INK).move_to((os_[0] + os_[1]) / 2 + 0.2 * UP); arr2 = T("→", 0.7, INK).move_to((os_[1] + os_[2]) / 2 + 0.2 * UP)
            self.mark("order"); self.play(FadeIn(lbl), FadeIn(t1), run_time=0.5)
            t2 = t1.copy(); self.play(t2.animate.shift(os_[1] - os_[0]), FadeIn(arr1), run_time=0.5); self.play(ApplyMatrix(M3(M1), t2, about_point=os_[1]), run_time=0.8)
            t3 = t2.copy(); self.play(t3.animate.shift(os_[2] - os_[1]), FadeIn(arr2), run_time=0.5); self.play(ApplyMatrix(M3(M2), t3, about_point=os_[2]), run_time=0.8)
            P = np.array(M2) @ np.array(M1)
            res = VGroup(T(nm + " =", 0.45, INK), matrix_tex(P.tolist(), sc=0.45)).arrange(RIGHT, buff=0.2).move_to([5.3, y - 0.3, 0])
            self.play(FadeIn(res), run_time=0.4)
        self.mark("diff")
        nt = VGroup(T("mismos pasos, otro orden,", 0.45, WARN), T("otro camión", 0.45, WARN)).arrange(DOWN, aligned_edge=LEFT, buff=0.12).move_to([4.6, -3.2, 0])
        self.play(FadeIn(nt), run_time=0.5); self.wait(2.8)
        self.clear_but(cap)

        # ---------- 4. determinante 0 ----------
        cap = self.swap(cap, self.caption("Área 0: no se puede deshacer", "la flecha violeta gira hasta la azul"))
        s4 = 1.1; O4 = np.array([-3.2, -1.4, 0])
        t = ValueTracker(0.0)
        def M4():
            v = t.get_value(); return [[1.5, (1 - v) * -0.5 + v * 0.75], [0.5, (1 - v) * 1.2 + v * 0.25]]
        def para():
            M = M4(); pts = [O4 + s4 * np.array([*np.array(M) @ q, 0]) for q in ([0, 0], [1, 0], [1, 1], [0, 1])]
            return Polygon(*pts, stroke_color=MEAN, stroke_width=2, fill_color=MEAN, fill_opacity=0.2)
        def cols():
            M = M4()
            return VGroup(Arrow(O4, O4 + s4 * np.array([M[0][0], M[1][0], 0]), buff=0, color=MEAN, stroke_width=7, max_tip_length_to_length_ratio=0.15),
                          Arrow(O4, O4 + s4 * np.array([M[0][1], M[1][1], 0]), buff=0, color=SAMPLE, stroke_width=7, max_tip_length_to_length_ratio=0.15))
        def pts():
            M = np.array(M4())
            return VGroup(*[Dot(O4 + s4 * np.array([*(M @ q), 0]), radius=0.09, color=POINT) for q in ([1, 1], [2, -1], [-0.6, 1.4], [0.4, -0.6], [-1.2, 0.2])])
        def ro():
            M = M4(); d = M[0][0] * M[1][1] - M[0][1] * M[1][0]
            return VGroup(T("factor de área", 0.45, MUTED), T(f"× {d:.2f}".replace(".", ","), 1.1, WARN if d < 0.3 else INK, BOLD)).arrange(DOWN, buff=0.2).move_to([4.0, 0.6, 0])
        axes = VGroup(Line(O4 + LEFT * 2.8, O4 + RIGHT * 5.4, stroke_color=MUTED, stroke_opacity=0.4), Line(O4 + DOWN * 1.6, O4 + UP * 3.4, stroke_color=MUTED, stroke_opacity=0.4))
        pg, cl, pp, r4 = always_redraw(para), always_redraw(cols), always_redraw(pts), always_redraw(ro)
        self.mark("det"); self.play(FadeIn(axes), FadeIn(pg), FadeIn(cl), FadeIn(pp), FadeIn(r4), run_time=0.8); self.wait(0.6)
        self.mark("squash"); self.play(t.animate.set_value(1.0), run_time=3.0, rate_func=smooth)
        ring = Circle(radius=0.22, color=WARN, stroke_width=4).move_to(O4 + s4 * np.array([2.25, 0.75, 0]))
        nt = VGroup(T("dos camiones distintos,", 0.45, WARN), T("el mismo punto final", 0.45, WARN)).arrange(DOWN, aligned_edge=LEFT, buff=0.12).move_to([3.9, -1.0, 0])
        nt2 = T("como guardar km y millas: dos columnas, un solo dato", 0.42, MUTED).move_to([0, -3.4, 0])
        self.mark("lost"); self.play(Create(ring), FadeIn(nt), run_time=0.7); self.play(FadeIn(nt2), run_time=0.5); self.wait(2.8)
        self.clear_but(cap)

        # ---------- 5. capa de red neuronal ----------
        cap = self.swap(cap, self.caption("Una capa de red: matriz, desplazamiento y doblez", "temperatura del motor (→) y del aceite (↑)"))
        D = json.load(open("../mt/m3/embed.json")); X = np.array(D["X"]); Y = np.array(D["y"])
        O5 = np.array([-2.8, -0.75, 0]); s5 = 0.82
        ax5 = VGroup(Line(O5 + LEFT * 2.7, O5 + RIGHT * 2.7, stroke_color=MUTED, stroke_opacity=0.5), Line(O5 + DOWN * 2.7, O5 + UP * 2.7, stroke_color=MUTED, stroke_opacity=0.5))
        P0 = [O5 + s5 * np.array([a, b, 0]) for a, b in X]
        c = 0.8
        Z = np.c_[X[:, 0] - X[:, 1] - c, X[:, 1] - X[:, 0] - c]; H = np.maximum(0, Z)
        dots = VGroup(*[Dot(p, radius=0.055, color=POINT if y else SAMPLE) for p, y in zip(P0, Y)])
        self.mark("cloud"); self.play(FadeIn(ax5), FadeIn(dots, lag_ratio=0.01), run_time=1.0)
        th, tt_ = D["line"]; n = np.array([np.cos(th), np.sin(th)]); q = tt_ * n; dvec = np.array([-n[1], n[0]])
        bl = Line(O5 + s5 * np.array([*(q - 3.3 * dvec), 0]), O5 + s5 * np.array([*(q + 3.3 * dvec), 0]), color=GOOD, stroke_width=4)
        i1 = VGroup(T("la mejor recta:", 0.45, MUTED), T("23 errores de 160", 0.55, WARN, BOLD)).arrange(DOWN, aligned_edge=LEFT, buff=0.15).move_to([3.4, 1.2, 0])
        self.play(Create(bl), FadeIn(i1), run_time=0.8); self.wait(1.6)
        i2 = VGroup(T("matriz + desplazamiento:", 0.45, MUTED), T("motor − aceite − margen", 0.42, INK), T("aceite − motor − margen", 0.42, INK)).arrange(DOWN, aligned_edge=LEFT, buff=0.15).move_to([3.4, 1.2, 0])
        self.mark("matrix"); self.play(FadeOut(bl), FadeOut(i1), FadeIn(i2), *[d.animate.move_to(O5 + s5 * np.array([*z, 0])) for d, z in zip(dots, Z)], run_time=1.6); self.wait(1.3)
        i3 = VGroup(T("doblez (ReLU):", 0.45, MUTED), T("lo negativo pasa a 0", 0.42, INK)).arrange(DOWN, aligned_edge=LEFT, buff=0.15).move_to([3.4, 1.2, 0])
        self.mark("fold"); self.play(FadeOut(i2), FadeIn(i3), *[d.animate.move_to(O5 + s5 * np.array([*h, 0])) for d, h in zip(dots, H)], run_time=1.6)
        sep = Line(O5 + s5 * np.array([-1.0, 1.15, 0]), O5 + s5 * np.array([1.15, -1.0, 0]), color=GOOD, stroke_width=4)
        i4 = VGroup(T("ahora una recta separa:", 0.45, MUTED), T("0 errores", 0.6, GOOD, BOLD)).arrange(DOWN, aligned_edge=LEFT, buff=0.15).next_to(i3, DOWN, buff=0.5).align_to(i3, LEFT)
        self.mark("sep"); self.play(Create(sep), FadeIn(i4), run_time=0.8); self.wait(3.0)
        self.clear_but(cap)

        # ---------- Cierre ----------
        self.mark("legend")
        self.play(FadeOut(cap), run_time=0.6)
        e1 = Text("Una matriz mueve el plano entero.", font=FONT, weight=BOLD, color=INK).scale(0.72)
        e2 = Text("Sus columnas dicen cómo; su determinante, si se puede deshacer.", font=FONT, color=MUTED).scale(0.42)
        VGroup(e1, e2).arrange(DOWN, buff=0.4)
        self.mark("end_text"); self.play(Write(e1), run_time=1.4); self.play(FadeIn(e2, shift=0.2 * UP), run_time=0.8)
        self.wait(2.4)
        self.mark("fade"); self.play(FadeOut(VGroup(e1, e2)), run_time=1)
        self.marks["total"] = [round(self.renderer.time, 3)]
        json.dump(self.marks, open("marks.json", "w"), indent=1)
