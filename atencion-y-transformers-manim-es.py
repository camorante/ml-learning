import json, sys
import numpy as np
from manim import *

sys.path.insert(0, "../rn/r11/py")
import atencion as AT                                      # el mismo modelo que las páginas

# Paleta oscura de la serie
BG = "#0E131B"; INK = "#E4EAF3"; MUTED = "#9BA7B9"; LINE = "#2B3545"
MEAN = "#6EA6FF"; SAMPLE = "#AE9FF3"; POINT = "#FF9A55"; GOOD = "#4CC38A"; WARN = "#F08A5D"
FONT = "Inter"
config.background_color = BG
D = json.load(open("../rn/r11/embed.json", encoding="utf-8"))
VO, TIPO, VJ = D["vocab"], D["tipo"], D["viaje"]
TC = [INK, MEAN, POINT, MUTED]
P = {k: np.array(v) for k, v in D["P"].items()}
_, cache = AT.modelo(np.array(VJ), P, 2)
ZL, _ = AT.modelo(np.array(VJ), P, 2)
PR = AT.softmax(ZL)
WH = cache[1][1][5]                                         # pesos de atención de las 2 cabezas (n × n)
IR, IT = VJ.index(VO.index("regreso")), VJ.index(VO.index("taller"))


def T(s, sc=0.45, col=INK, w=None):
    return Text(s, font=FONT, color=col, weight=w or NORMAL).scale(sc)


def pc(p, d=1):
    return f"{100 * round(p, d + 2):.{d}f}".replace(".", ",") + " %"


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

    def caja(self, tok, w=1.5, h=0.62, peso=None, q=False):
        r = RoundedRectangle(width=w, height=h, corner_radius=0.1, stroke_color=POINT if q else LINE, stroke_width=4 if q else 1.5,
                             fill_color=MEAN if peso is not None else BG, fill_opacity=(0.05 + 0.9 * peso) if peso is not None else 1)
        bar = Rectangle(width=0.07, height=h - 0.12, stroke_width=0, fill_color=TC[TIPO[tok]], fill_opacity=1).move_to(r.get_left() + RIGHT * 0.1)
        lab = T(VO[tok], 0.27, BG if (peso or 0) > 0.5 else INK, BOLD if q else None)
        g = VGroup(r, bar, lab)
        if peso is not None:
            lab.move_to(r.get_center() + UP * 0.12)
            g.add(T(pc(peso), 0.22, BG if peso > 0.5 else MUTED, BOLD).move_to(r.get_center() + DOWN * 0.15))
        else:
            lab.move_to(r)
        return g

    def rejilla(self, pesos, q, fut=None):
        g = VGroup()
        for i, t in enumerate(VJ):
            r, c = divmod(i, 7)
            tapada = fut is not None and i > q
            b = self.caja(t, peso=None if tapada else pesos[i], q=i == q)
            if tapada:
                b[0].set_stroke(LINE, 1.5).set_fill(BG, 1); b.set_opacity(0.35)
            b.move_to([-5.1 + c * 1.7, 1.25 - r * 0.8, 0])
            g.add(b)
        return g

    def construct(self):
        self.marks = {}
        title = Text("Atención y Transformers", font=FONT, weight=BOLD, color=INK).scale(1.0)
        sub = Text("la pieza que hay detrás de los LLM", font=FONT, color=MUTED).scale(0.46)
        VGroup(title, sub).arrange(DOWN, buff=0.35)
        self.mark("title"); self.play(Write(title), run_time=1.4)
        self.mark("subtitle"); self.play(FadeIn(sub, shift=0.2 * UP)); self.wait(1.4)
        self.mark("title_out"); self.play(FadeOut(title), FadeOut(sub))

        # ---------- 1. la red recurrente ----------
        cap = self.swap(None, self.caption("Una red recurrente: una sola memoria", "lee el viaje ficha a ficha y lo resume todo en un estado"))
        sel = [0, 1, 2, 3, 6, 11, 14, 17, 18]
        fila = VGroup(*[self.caja(VJ[i], w=1.3) for i in sel]).arrange(RIGHT, buff=0.12).move_to([0, 1.1, 0])
        self.play(LaggedStart(*[FadeIn(b, shift=0.2 * DOWN) for b in fila], lag_ratio=0.08), run_time=1.0)
        est = VGroup(*[Square(0.42, stroke_width=0, fill_color=POINT, fill_opacity=0.25 + 0.75 * 0.82 ** max(0, k - 1)).next_to(fila[k], DOWN, buff=0.35) for k in range(len(sel))])
        flech = VGroup(*[Arrow(est[k].get_right(), est[k + 1].get_left(), buff=0.05, stroke_width=3, color=MUTED, max_tip_length_to_length_ratio=0.35) for k in range(len(sel) - 1)])
        self.mark("rnn"); self.play(LaggedStart(*[AnimationGroup(FadeIn(est[k]), GrowArrow(flech[k]) if k < len(flech) else Wait(0)) for k in range(len(sel))], lag_ratio=0.35), run_time=3.0)
        t1 = T(f"«{VO[VJ[1]]}» tiene que sobrevivir {IR - 1} pasos dentro del estado", 0.36, INK, BOLD).move_to([0, -1.2, 0])
        C = D["comp"]["LSTM"]["dep"]
        t2 = T(f"una LSTM de 3 780 números acierta el depósito: {pc(C[0])} · {pc(C[1])} · {pc(C[2])}", 0.32, WARN, BOLD).move_to([0, -2.0, 0])
        t3 = T("(tres puntos de partida; adivinando al azar, 25 %)", 0.28, MUTED).move_to([0, -2.6, 0])
        self.play(FadeIn(t1), run_time=0.6); self.wait(0.8); self.play(FadeIn(t2), FadeIn(t3), run_time=0.6); self.wait(3.0)
        self.clear_but(cap)

        # ---------- 2. la atención ----------
        cap = self.swap(cap, self.caption("La atención: cada ficha mira a todas", "el Transformer del tema, desde «regreso»: ¿cuánto importa cada ficha?"))
        g = self.rejilla(WH[0][IR], IR, fut=True)
        self.mark("attend"); self.play(LaggedStart(*[FadeIn(b) for b in g], lag_ratio=0.03), run_time=1.4)
        pr = VGroup(T("ficha siguiente:", 0.34, MUTED), T(f"{VO[VJ[IR + 1]]} {pc(PR[IR][VJ[IR + 1]])}", 0.4, GOOD, BOLD)).arrange(RIGHT, buff=0.25).move_to([0, -1.6, 0])
        n1 = T(f"cabeza 1: {pc(WH[0][IR][1])} de la atención en «{VO[VJ[1]]}», a {IR - 1} fichas: un solo paso", 0.32, INK, BOLD).move_to([0, -2.4, 0])
        self.play(FadeIn(pr), FadeIn(n1), run_time=0.7); self.wait(2.6)
        u = VJ[IT + 1]
        g2 = self.rejilla(WH[0][IT], IT, fut=True)
        pr2 = VGroup(T("ficha siguiente:", 0.34, MUTED), T(f"{VO[u]} {pc(PR[IT][u])}", 0.4, GOOD, BOLD)).arrange(RIGHT, buff=0.25).move_to([0, -1.6, 0])
        cods = [i for i in range(IT) if TIPO[VJ[i]] == 2]
        n2 = T(f"desde «taller»: {pc(WH[0][IT][cods[-1]])} en {VO[VJ[cods[-1]]]}, el último código; {pc(WH[0][IT][cods[0]])} en {VO[VJ[cods[0]]]}", 0.32, INK, BOLD).move_to([0, -2.4, 0])
        self.mark("taller"); self.play(ReplacementTransform(g, g2), ReplacementTransform(pr, pr2), ReplacementTransform(n1, n2), run_time=1.0); self.wait(3.0)
        self.clear_but(cap)

        # ---------- 3. q, k, v ----------
        cap = self.swap(cap, self.caption("Pregunta, etiqueta y contenido", "puntuación q·k, dividir entre √d, softmax y mezclar los contenidos"))
        nm = ["salida", "Madrid", "regreso"]; sc = [-1, 2, 1]; cols = [SAMPLE, MEAN, GOOD]
        z = np.array(sc) / np.sqrt(2); w = np.exp(z) / np.exp(z).sum()
        hdr = VGroup(*[T(h, 0.3, MUTED, BOLD).move_to([x, 1.15, 0]) for h, x in (("ficha", -4.6), ("q·k", -2.6), ("÷ √2", -0.9), ("peso", 2.4))])
        filas = VGroup()
        for i in range(3):
            y = 0.45 - 0.75 * i
            bar = Rectangle(width=max(0.03, 3.6 * w[i]), height=0.36, stroke_width=0, fill_color=cols[i], fill_opacity=1).move_to([0.9, y, 0], aligned_edge=LEFT)
            filas.add(VGroup(T(nm[i], 0.36, cols[i], BOLD).move_to([-4.6, y, 0]), T(str(sc[i]).replace("-", "−"), 0.36, INK).move_to([-2.6, y, 0]),
                             T(f"{z[i]:.3f}".replace(".", ",").replace("-", "−"), 0.36, INK).move_to([-0.9, y, 0]), bar,
                             T(pc(w[i]), 0.32, INK, BOLD).next_to([4.6, y, 0], RIGHT, buff=0.1)))
        qv = T("pregunta de «regreso»: q = (−1; 2) · etiquetas k: (1; 0), (0; 1), (1; 1)", 0.3, POINT, BOLD).move_to([0, 1.75, 0])
        self.play(FadeIn(qv), FadeIn(hdr), run_time=0.6)
        self.mark("qk")
        for c in range(3):
            self.play(*[FadeIn(f[c]) for f in filas], run_time=0.5)
        self.play(*[GrowFromEdge(f[3], LEFT) for f in filas], *[FadeIn(f[4]) for f in filas], run_time=0.8)
        o = w @ np.array([[1, 0], [0, 1], [1, 1]])
        mz = T(f"salida de «regreso» = suma de peso × v = ({o[0]:.3f}; {o[1]:.3f})".replace(".", ","), 0.34, INK, BOLD).move_to([0, -2.1, 0])
        m2 = T("los pesos suman 100 %: «regreso» se lleva sobre todo el contenido de «Madrid»", 0.3, GOOD).move_to([0, -2.75, 0])
        self.play(FadeIn(mz), FadeIn(m2), run_time=0.6); self.wait(3.0)
        self.clear_but(cap)

        # ---------- 4. cabezas y máscara ----------
        cap = self.swap(cap, self.caption("Dos cabezas y la máscara causal", "cada cabeza busca otra cosa; nadie mira al futuro"))
        n = len(VJ); cell = 0.19
        mapas = VGroup()
        for h, cx in ((0, -3.1), (1, 3.1)):
            m = VGroup()
            for i in range(n):
                for j in range(i + 1):
                    m.add(Square(cell * 0.92, stroke_width=0, fill_color=MEAN, fill_opacity=0.05 + 0.95 * WH[h][i][j]).move_to([cx - n * cell / 2 + (j + 0.5) * cell, -0.6 + n * cell / 2 - (i + 0.5) * cell, 0]))
            ttl = T(f"cabeza {h + 1}", 0.36, INK, BOLD).next_to(m, UP, buff=0.2).align_to([cx - n * cell / 2, 0, 0], LEFT)
            mapas.add(VGroup(m, ttl))
        self.mark("heads"); self.play(LaggedStart(*[FadeIn(mp) for mp in mapas], lag_ratio=0.4), run_time=1.6)
        hl = VGroup(*[SurroundingRectangle(mapas[h][0][IT * (IT + 1) // 2 + cods[-1]], color=POINT, buff=0.03, stroke_width=3) for h in (0, 1)])
        nh = VGroup(T(f"desde «taller», peso en el último código: cabeza 1 {pc(WH[0][IT][cods[-1]])} · cabeza 2 {pc(WH[1][IT][cods[-1]])}", 0.3, POINT, BOLD),
                    T("el triángulo vacío es la máscara: cada ficha solo ve las anteriores", 0.28, MUTED)).arrange(DOWN, buff=0.15).move_to([0, -3.35, 0])
        self.play(Create(hl), FadeIn(nh), run_time=0.8); self.wait(3.2)
        self.clear_but(cap)

        # ---------- 5. la posición ----------
        cap = self.swap(cap, self.caption("La atención no sabe de orden", "a cada ficha se le suma una huella de su posición"))
        Xp = np.linspace(-5.5, 5.5, 300)
        ondas = VGroup(*[VMobject(stroke_color=[MEAN, SAMPLE, GOOD][k], stroke_width=3).set_points_smoothly([[x, 1.0 + 0.45 * np.sin((x + 5.5) * 3 / 2 ** k), 0] for x in Xp[::3]]) for k in range(3)])
        for k, o_ in enumerate(ondas):
            o_.shift(DOWN * 0.0 * k)
        self.play(Create(ondas), run_time=1.2)
        L = -1.5
        res = [("con posición", 0.992, GOOD), ("sin posición", 0.624, WARN), ("al azar entre los encendidos", 0.607, MUTED)]
        barras = VGroup()
        for k, (nm_, v, col) in enumerate(res):
            y = -0.4 - 0.75 * k
            barras.add(VGroup(T(nm_, 0.32, col, BOLD).move_to([L - 0.2, y, 0], aligned_edge=RIGHT),
                              Rectangle(width=5.0 * v, height=0.4, stroke_width=0, fill_color=col, fill_opacity=0.9).move_to([L, y, 0], aligned_edge=LEFT),
                              T(pc(v), 0.32, INK, BOLD).move_to([L + 5.0 * v + 0.2, y, 0], aligned_edge=LEFT)))
        tt_ = T("último código tras «taller», en 1 000 viajes de prueba", 0.3, MUTED).move_to([0, 0.25, 0])
        self.mark("pos"); self.play(FadeIn(tt_), LaggedStart(*[FadeIn(b, shift=0.2 * RIGHT) for b in barras], lag_ratio=0.3), run_time=1.4)
        self.wait(3.2)
        self.clear_but(cap)

        # ---------- 6. bloques y LLM ----------
        cap = self.swap(cap, self.caption("Un Transformer: bloques apilados", "un LLM es una pila enorme que predice la ficha siguiente"))
        piezas = ["norma", "atención", "+ atajo", "norma", "red pequeña", "+ atajo"]
        pc_ = [MUTED, POINT, GOOD, MUTED, SAMPLE, GOOD]
        blq = VGroup(*[VGroup(RoundedRectangle(width=3.0, height=0.42, corner_radius=0.08, stroke_color=pc_[k], stroke_width=2.5), T(p, 0.28, INK)) for k, p in enumerate(piezas)]).arrange(DOWN, buff=0.1)
        for b in blq:
            b[1].move_to(b[0])
        marco = SurroundingRectangle(blq, color=INK, buff=0.15, stroke_width=2).set_stroke(opacity=0.8)
        bloque = VGroup(blq, marco).move_to([-3.8, -0.85, 0])
        self.play(FadeIn(bloque, lag_ratio=0.1), run_time=1.0)
        pila = VGroup(*[RoundedRectangle(width=2.2, height=0.17, corner_radius=0.04, stroke_width=0, fill_color=MEAN, fill_opacity=0.35 + 0.5 * (k % 2)) for k in range(12)]).arrange(DOWN, buff=0.04).move_to([0.9, -0.85, 0])
        l1 = VGroup(T("× 12 bloques", 0.34, INK, BOLD), T("GPT-2 pequeño: 124 millones", 0.28, MUTED)).arrange(DOWN, buff=0.12).next_to(pila, DOWN, buff=0.25)
        pila2 = VGroup(*[Rectangle(width=2.2, height=0.022, stroke_width=0, fill_color=POINT, fill_opacity=0.4 + 0.5 * (k % 2)) for k in range(96)]).arrange(DOWN, buff=0.007).move_to([4.5, -0.85, 0])
        l2 = VGroup(T("× 96 bloques", 0.34, INK, BOLD), T("GPT-3: 175 000 millones", 0.28, MUTED)).arrange(DOWN, buff=0.12).next_to(pila2, DOWN, buff=0.25)
        lt = T("el del tema: 1 bloque, 3 972 números", 0.28, MUTED).next_to(bloque, DOWN, buff=0.25)
        self.mark("stack"); self.play(FadeIn(lt), FadeIn(pila, lag_ratio=0.1), FadeIn(l1), run_time=1.2)
        self.play(FadeIn(pila2, lag_ratio=0.02), FadeIn(l2), run_time=1.2); self.wait(1.8)
        cst = T("el precio: cada ficha con todas · 40 fichas, 820 parejas · 100 000 fichas, 5 000 millones", 0.28, WARN, BOLD).move_to([0, -3.55, 0])
        self.mark("cost"); self.play(FadeIn(cst), run_time=0.6); self.wait(2.8)
        self.clear_but(cap)

        # ---------- Cierre ----------
        self.mark("legend")
        self.play(FadeOut(cap), run_time=0.6)
        e1 = Text("Atención y Transformers", font=FONT, weight=BOLD, color=INK).scale(0.7)
        e2 = VGroup(T("cada ficha mira directamente a las demás", 0.42, MUTED),
                    T("pregunta · etiqueta → softmax → mezcla de contenidos", 0.42, MUTED),
                    T("varias cabezas, posición y máscara causal", 0.42, MUTED),
                    T("bloques apilados que predicen la ficha siguiente", 0.42, MUTED)).arrange(DOWN, buff=0.18)
        VGroup(e1, e2).arrange(DOWN, buff=0.45)
        self.mark("end_text"); self.play(Write(e1), run_time=1.0); self.play(FadeIn(e2, shift=0.2 * UP, lag_ratio=0.15), run_time=1.0)
        self.wait(3.8)
        self.mark("fade"); self.play(FadeOut(VGroup(e1, e2)), run_time=1)
        self.marks["total"] = [round(self.renderer.time, 3)]
        json.dump(self.marks, open("marks.json", "w"), indent=1)
