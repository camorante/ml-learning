import json
import numpy as np
from manim import *

# Paleta oscura de la serie
BG = "#0E131B"; INK = "#E4EAF3"; MUTED = "#9BA7B9"; LINE = "#2B3545"
MEAN = "#6EA6FF"; SAMPLE = "#AE9FF3"; POINT = "#FF9A55"; GOOD = "#4CC38A"; WARN = "#F08A5D"
FONT = "Inter"
config.background_color = BG
D = json.load(open("../rn/r10/embed.json", encoding="utf-8"))   # los mismos datos que las páginas
SC = [POINT, SAMPLE, MEAN, GOOD]                                # motor, escape, frenos, eléctrico
COD = D["cod"]; SIS = D["s"]


def T(s, sc=0.45, col=INK, w=None):
    return Text(s, font=FONT, color=col, weight=w or NORMAL).scale(sc)


def fmt(v, d=1):
    s = f"{abs(v):.{d}f}".replace(".", ",")
    return ("−" if v < 0 and round(abs(v), d) != 0 else "") + s


def coseno(E):
    """Coseno entre filas con el mismo orden de sumas que el JS."""
    n = [np.sqrt(sum(v * v for v in r)) for r in E]
    out = []
    for i, r in enumerate(E):
        fila = []
        for j, q in enumerate(E):
            a = 0.0
            for k in range(len(r)):
                a += r[k] * q[k]
            fila.append(a / (n[i] * n[j]))
        out.append(fila)
    return out


def vecinos(S, i, k=3):
    return sorted((j for j in range(len(S)) if j != i), key=lambda j: (-S[i][j], j))[:k]


def pureza(E):
    S = coseno(E)
    return sum(SIS[j] == SIS[i] for i in range(len(E)) for j in vecinos(S, i))



def separa(labs, gap=0.06):
    """Push apart overlapping labels vertically, keeping their x position."""
    labs = sorted(labs, key=lambda m: -m.get_center()[1])
    for a, b in zip(labs, labs[1:]):
        if abs(a.get_center()[0] - b.get_center()[0]) < (a.width + b.width) / 2:
            need = a.get_bottom()[1] - gap - b.get_top()[1]
            if need < 0:
                b.shift([0, need, 0])
    return labs

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
        title = Text("Embeddings", font=FONT, weight=BOLD, color=INK).scale(1.05)
        sub = Text("categorías y palabras convertidas en vectores", font=FONT, color=MUTED).scale(0.46)
        VGroup(title, sub).arrange(DOWN, buff=0.35)
        self.mark("title"); self.play(Write(title), run_time=1.4)
        self.mark("subtitle"); self.play(FadeIn(sub, shift=0.2 * UP)); self.wait(1.4)
        self.mark("title_out"); self.play(FadeOut(title), FadeOut(sub))

        # ---------- 1. el one-hot ----------
        cap = self.swap(None, self.caption("El one-hot: una casilla por categoría", "20 códigos de avería → 20 casillas, todas a 0 menos una"))
        filas = VGroup()
        for r, c in enumerate([0, 1, 10]):
            cells = VGroup(*[Square(0.44, stroke_color=POINT if k == c else LINE, stroke_width=2, fill_color=POINT if k == c else BG,
                                    fill_opacity=0.9 if k == c else 1).move_to([-4.35 + 0.5 * k, 1.35 - 1.05 * r, 0]) for k in range(20)])
            nums = VGroup(*[T("1" if k == c else "0", 0.3, BG if k == c else MUTED, BOLD if k == c else None).move_to(cells[k]) for k in range(20)])
            lab = T(COD[c], 0.36, SC[SIS[c]], BOLD).next_to(cells, LEFT, buff=0.3)
            filas.add(VGroup(cells, nums, lab))
        self.mark("rows"); self.play(LaggedStart(*[FadeIn(f, shift=0.2 * RIGHT) for f in filas], lag_ratio=0.3), run_time=1.6); self.wait(0.8)
        t1 = T("entre dos códigos cualesquiera: distancia 1,41 y coseno 0", 0.38, INK, BOLD).move_to([0, -1.75, 0])
        t2 = T("para el one-hot, el termostato está tan lejos del motor como un sensor de rueda", 0.32, MUTED).move_to([0, -2.4, 0])
        t3 = T("y con 5 000 rutas, 5 000 casillas por fila", 0.34, WARN, BOLD).move_to([0, -3.2, 0])
        self.mark("same"); self.play(FadeIn(t1), run_time=0.6); self.play(FadeIn(t2), run_time=0.6); self.wait(1.0)
        self.play(FadeIn(t3), run_time=0.6); self.wait(1.6)
        self.clear_but(cap)

        # ---------- 2. la tabla ----------
        cap = self.swap(cap, self.caption("Un embedding es una fila de una tabla", "one-hot × tabla = la fila del código"))
        E = D["E"]["60"]; y0, rh = 1.8, 0.268
        tab = VGroup()
        for i in range(20):
            y = y0 - i * rh
            tab.add(VGroup(T(COD[i], 0.24, SC[SIS[i]]).move_to([-3.9, y, 0]),
                           T("0", 0.24, MUTED).move_to([-2.6, y, 0]),
                           T(fmt(E[i][0], 2), 0.24, INK).move_to([-1.3, y, 0]), T(fmt(E[i][1], 2), 0.24, INK).move_to([-0.3, y, 0])))
        hdr = VGroup(T("one-hot", 0.28, MUTED, BOLD).move_to([-2.6, y0 + 0.38, 0]), T("tabla: 20 filas × 2 números", 0.28, MUTED, BOLD).move_to([-0.8, y0 + 0.38, 0]))
        self.play(FadeIn(hdr), LaggedStart(*[FadeIn(r) for r in tab], lag_ratio=0.04), run_time=1.4)
        res_lab = T("embedding", 0.32, MUTED).move_to([3.4, 0.9, 0])
        prev = None
        for i in (0, 10):
            y = y0 - i * rh
            hl = RoundedRectangle(width=4.6, height=0.27, corner_radius=0.06, stroke_width=0, fill_color=POINT, fill_opacity=0.22).move_to([-2.1, y, 0])
            one = T("1", 0.26, POINT, BOLD).move_to([-2.6, y, 0])
            res = VGroup(T(COD[i], 0.4, SC[SIS[i]], BOLD), T(f"({fmt(E[i][0], 2)}; {fmt(E[i][1], 2)})", 0.5, POINT, BOLD)).arrange(DOWN, buff=0.2).move_to([3.4, 0.0, 0])
            arr = Arrow([0.3, y, 0], [2.0, 0.0, 0], color=POINT, stroke_width=4, buff=0.1)
            self.mark("lookup")
            if prev is None:
                self.play(FadeIn(hl), Transform(tab[i][1], one), GrowArrow(arr), FadeIn(res), FadeIn(res_lab), run_time=0.9)
            else:
                self.play(FadeOut(prev[0]), Transform(tab[prev[3]][1], T("0", 0.24, MUTED).move_to(tab[prev[3]][1])), FadeOut(prev[1]), FadeOut(prev[2]),
                          FadeIn(hl), Transform(tab[i][1], one), GrowArrow(arr), FadeIn(res), run_time=0.9)
            prev = (hl, arr, res, i)
            self.wait(1.4)
        nt = VGroup(T("las 19 filas × 0 desaparecen:", 0.32, GOOD, BOLD), T("la red coge la fila y no multiplica nada", 0.32, GOOD, BOLD)).arrange(DOWN, buff=0.15).move_to([3.4, -1.7, 0])
        self.play(FadeIn(nt), run_time=0.5); self.wait(1.6)
        self.clear_but(cap)

        # ---------- 3. el entrenamiento ----------
        cap = self.swap(cap, self.caption("Se aprende jugando a adivinar", "visto un código de una visita, ¿qué otro código hay?"))
        R = 3.0; S0 = 2.75; C0 = np.array([-3.2, -0.65, 0])
        P = lambda p: C0 + np.array([p[0] / R * S0, p[1] / R * S0, 0])
        marco = Square(2 * S0, stroke_color=LINE, stroke_width=2).move_to(C0)
        ejes = VGroup(Line(P((-R, 0)), P((R, 0)), stroke_color=LINE, stroke_width=1.5), Line(P((0, -R)), P((0, R)), stroke_color=LINE, stroke_width=1.5))
        fotos = D["fotos"]; snaps = [np.array(D["E"][str(ep)]) for ep in fotos]
        q = ValueTracker(0)

        def pos():
            v = q.get_value(); k = min(int(v), len(snaps) - 2); a = v - k
            return snaps[k] * (1 - a) + snaps[k + 1] * a

        def puntos():
            E_ = pos()
            return VGroup(*[Dot(P(E_[i]), radius=0.09, color=SC[SIS[i]]) for i in range(20)])

        def contador():
            k = int(round(q.get_value()))
            ep = fotos[k]
            return VGroup(T(f"época {ep}", 0.42, INK, BOLD), T(f"vecinos del mismo sistema: {pureza(snaps[k])} de 60", 0.34, MUTED),
                          T(f"aciertos (entre los 3 primeros): {fmt(100 * round(D['hist'][str(ep)][1], 3), 1)} %", 0.34, MUTED)).arrange(DOWN, aligned_edge=LEFT, buff=0.2).move_to([3.6, 1.2, 0], aligned_edge=LEFT).shift(LEFT * 1.0)
        leyenda = VGroup(*[VGroup(Dot(radius=0.09, color=SC[k]), T(n, 0.32, SC[k])).arrange(RIGHT, buff=0.15) for k, n in enumerate(D["sis"])]).arrange(RIGHT, buff=0.35).move_to([3.6, -1.0, 0])
        ley2 = T("los colores los ponemos nosotros: la red nunca los ve", 0.28, MUTED).next_to(leyenda, DOWN, buff=0.25)
        pts, cnt = always_redraw(puntos), always_redraw(contador)
        self.play(Create(marco), FadeIn(ejes), FadeIn(pts), FadeIn(cnt), FadeIn(leyenda), FadeIn(ley2), run_time=1.0)
        self.wait(1.0)
        self.mark("train"); self.play(q.animate.set_value(10), run_time=6.0, rate_func=linear)
        self.play(q.animate.set_value(len(snaps) - 1), run_time=2.0, rate_func=smooth)
        pts.clear_updaters(); cnt.clear_updaters()
        Ef = snaps[-1]
        gt = VGroup(T("cuatro grupos, uno por sistema,", 0.34, GOOD, BOLD), T("sin que nadie se lo diga", 0.34, GOOD, BOLD)).arrange(DOWN, buff=0.15).move_to([3.6, -2.4, 0])
        self.mark("groups"); self.play(FadeIn(gt), run_time=0.8); self.wait(1.8)

        # ---------- 4. coseno ----------
        cap = self.swap(cap, self.caption("Parecido = misma dirección", "el coseno: 1 si dos flechas apuntan igual, −1 si al revés"))
        self.play(FadeOut(gt), FadeOut(cnt), FadeOut(leyenda), FadeOut(ley2), run_time=0.4)
        S = coseno(Ef.tolist())
        prev = None
        for nm, txt in (("P0401", "válvula EGR: entre el motor y el escape"), ("U0121", "sin comunicación con el ABS: entre eléctrico y frenos")):
            i = COD.index(nm); vs = vecinos(S, i)
            rays = VGroup(Line(C0, P(Ef[i]), stroke_color=INK, stroke_width=5),
                          *[DashedLine(C0, P(Ef[j]), stroke_color=SC[SIS[j]], stroke_width=3) for j in vs],
                          Circle(radius=0.2, stroke_color=INK, stroke_width=4).move_to(P(Ef[i])),
                          *separa([T(COD[j], 0.24, SC[SIS[j]], BOLD).next_to(P(Ef[j]), RIGHT if Ef[j][0] > Ef[i][0] else LEFT, buff=0.3 if j == i else 0.12) for j in [i] + vs]))
            info = VGroup(T(nm, 0.5, INK, BOLD), T(txt, 0.3, MUTED),
                          *[T(f"{COD[j]}   coseno {fmt(S[i][j], 3)}", 0.36, SC[SIS[j]], BOLD) for j in vs]).arrange(DOWN, aligned_edge=LEFT, buff=0.22).move_to([3.4, 0.0, 0])
            self.mark("cos")
            if prev is None:
                self.play(Create(rays), FadeIn(info), run_time=0.9)
            else:
                self.play(FadeOut(prev[0]), FadeOut(prev[1]), Create(rays), FadeIn(info), run_time=0.9)
            prev = (rays, info); self.wait(2.2)
        self.clear_but(cap)

        # ---------- 5. direcciones ----------
        cap = self.swap(cap, self.caption("Las direcciones también significan algo", "8 tipos de tramo: 4 zonas, de día y de noche"))
        TR = np.array(D["tramos"]["E"]); Z = D["tramos"]["zonas"]
        c2 = np.array([-2.0, -0.6, 0]); sc = 1.25; cm = TR.mean(0)
        Q = lambda p: c2 + np.array([(p[0] - cm[0]) * sc, (p[1] - cm[1]) * sc, 0])
        dots = VGroup(*[Dot(Q(TR[c]), radius=0.1, color=SAMPLE if c >= 4 else MEAN) for c in range(8)])
        nl = VGroup(*[T(f"{Z[c % 4]} {'noche' if c >= 4 else 'día'}", 0.26, SAMPLE if c >= 4 else MEAN).next_to(Q(TR[c]), RIGHT if c >= 4 else LEFT, buff=0.12) for c in range(8)])
        self.play(FadeIn(dots), FadeIn(nl), run_time=0.8)
        flechas = VGroup(*[Arrow(Q(TR[k]), Q(TR[k + 4]), color=POINT, stroke_width=4, buff=0.12, max_tip_length_to_length_ratio=0.12) for k in range(4)])
        self.mark("arrows"); self.play(LaggedStart(*[GrowArrow(a) for a in flechas], lag_ratio=0.35), run_time=1.8)
        ang = [np.degrees(np.arctan2(*(TR[k + 4] - TR[k])[::-1])) for k in range(4)]
        info = VGroup(T("de día a de noche:", 0.36, INK, BOLD), T("casi la misma flecha en las 4 zonas", 0.34, POINT, BOLD),
                      T(f"ángulos de {fmt(min(ang), 1)}° a {fmt(max(ang), 1)}°", 0.32, MUTED),
                      T("nadie le dijo a la red qué es «noche»", 0.32, GOOD, BOLD)).arrange(DOWN, aligned_edge=LEFT, buff=0.22).move_to([4.0, -0.4, 0])
        self.play(FadeIn(info), run_time=0.7); self.wait(2.4)
        self.clear_but(cap)

        # ---------- 6. dimensión ----------
        cap = self.swap(cap, self.caption("¿Cuántos números por código?", "pérdida en 1 000 visitas de prueba: menos es mejor"))
        K = ["1", "2", "3", "4", "8", "16", "one-hot"]; dm = D["dims"]
        L, Rr, B, Tp = -5.0, 5.6, -2.2, 1.7; y0_, y1_ = 2.35, 3.0
        X = lambda i: L + (i + 0.5) * (Rr - L) / 7
        Y = lambda v: B + (v - y0_) / (y1_ - y0_) * (Tp - B)
        lines = VGroup(*[DashedLine([L, Y(v), 0], [Rr, Y(v), 0], color=GOOD, stroke_width=2.5) for v in (D["ref"], D["azar"])])
        ltx = VGroup(T(f"lo mejor posible {fmt(D['ref'], 3)}", 0.28, GOOD).next_to([Rr, Y(D["ref"]), 0], DOWN, buff=0.1).align_to([Rr, 0, 0], RIGHT),
                     T(f"al azar {fmt(D['azar'], 3)}", 0.28, GOOD).next_to([Rr, Y(D["azar"]), 0], UP, buff=0.1).align_to([Rr, 0, 0], RIGHT))
        xt = VGroup(*[T(k, 0.32, MUTED).move_to([X(i), B - 0.4, 0]) for i, k in enumerate(K)])
        self.play(FadeIn(lines), FadeIn(ltx), FadeIn(xt), run_time=0.6)
        pts2 = VGroup()
        for i, k in enumerate(K):
            col = POINT if k in ("2", "3") else (WARN if k == "one-hot" else MUTED)
            d_ = Dot([X(i), Y(dm[k]["te"]), 0], radius=0.12, color=col)
            v_ = T(fmt(dm[k]["te"], 3), 0.26, INK).next_to(d_, UP, buff=0.12)
            p_ = T(f"{dm[k]['param']} núm.", 0.24, MUTED).move_to([X(i), B - 0.78, 0])
            pts2.add(VGroup(d_, v_, p_))
        self.mark("dims"); self.play(LaggedStart(*[FadeIn(p, shift=0.2 * UP) for p in pts2], lag_ratio=0.2), run_time=2.0)
        nt = T("2 o 3 números bastan; el one-hot, con 420, se ajusta de más", 0.34, POINT, BOLD).move_to([0, -3.65, 0])
        self.play(FadeIn(nt), run_time=0.5); self.wait(2.0)
        self.clear_but(cap)

        # ---------- 7. palabras ----------
        cap = self.swap(cap, self.caption("Las palabras también son categorías", "una nota del taller, palabra a palabra"))
        pal = ["ruido", "en", "rueda", "delantera"]
        cajas = VGroup(*[VGroup(RoundedRectangle(width=1.9, height=0.62, corner_radius=0.12, stroke_color=MEAN, stroke_width=2.5), T(w, 0.38, INK, BOLD)) for w in pal]).arrange(RIGHT, buff=0.4).move_to([0, 1.5, 0])
        for c in cajas:
            c[1].move_to(c[0])
        self.play(FadeIn(cajas, lag_ratio=0.2), run_time=1.0)
        rng = np.random.default_rng(4)      # tonos de gris decorativos: solo representan «8 números aprendidos»
        vecs = VGroup()
        for c in cajas:
            row = VGroup(*[Square(0.2, stroke_width=0.8, stroke_color=LINE, fill_color=interpolate_color(ManimColor(BG), ManimColor(MEAN), rng.uniform(0.25, 1)), fill_opacity=1) for _ in range(8)]).arrange(DOWN, buff=0.03)
            row.next_to(c, DOWN, buff=0.7)
            vecs.add(VGroup(Arrow(c.get_bottom(), row.get_top(), buff=0.08, color=MUTED, stroke_width=3, max_tip_length_to_length_ratio=0.2), row))
        self.mark("words"); self.play(LaggedStart(*[GrowFromEdge(v, UP) for v in vecs], lag_ratio=0.25), run_time=1.4)
        nt = VGroup(T("cada palabra, una fila de una tabla de embeddings", 0.34, INK, BOLD), T("así empiezan también los modelos de lenguaje", 0.32, GOOD, BOLD)).arrange(DOWN, buff=0.2).move_to([0, -3.1, 0])
        self.play(FadeIn(nt), run_time=0.6); self.wait(2.0)
        self.clear_but(cap)

        # ---------- Cierre ----------
        self.mark("legend")
        self.play(FadeOut(cap), run_time=0.6)
        e1 = Text("Embeddings", font=FONT, weight=BOLD, color=INK).scale(0.7)
        e2 = VGroup(T("unos pocos números por categoría, en lugar del one-hot", 0.42, MUTED),
                    T("una fila de una tabla, aprendida con la retropropagación", 0.42, MUTED),
                    T("lo parecido acaba en la misma dirección (coseno)", 0.42, MUTED),
                    T("códigos, rutas, camiones y palabras", 0.42, MUTED)).arrange(DOWN, buff=0.18)
        VGroup(e1, e2).arrange(DOWN, buff=0.45)
        self.mark("end_text"); self.play(Write(e1), run_time=1.0); self.play(FadeIn(e2, shift=0.2 * UP, lag_ratio=0.15), run_time=1.0)
        self.wait(3.3)
        self.mark("fade"); self.play(FadeOut(VGroup(e1, e2)), run_time=1)
        self.marks["total"] = [round(self.renderer.time, 3)]
        json.dump(self.marks, open("marks.json", "w"), indent=1)
