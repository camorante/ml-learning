import json
import numpy as np
from manim import *

# Paleta oscura de la serie ML visual
BG = "#0E131B"; INK = "#E4EAF3"; MUTED = "#9BA7B9"; LINE = "#2B3545"
MEAN = "#6EA6FF"; SAMPLE = "#AE9FF3"; POINT = "#FF9A55"; GOOD = "#4CC38A"; WARN = "#F08A5D"
FONT = "Inter"
config.background_color = BG
COLS = [MEAN, SAMPLE, POINT]
NAMES = ["En ruta", "Parado", "Cargando"]
P = np.array([[.7, .2, .1], [.4, .4, .2], [.2, .2, .6]])
PI = np.array([.5, .25, .25])
HA = np.array([[.9, .07, .03], [.15, .75, .1], [.05, .05, .9]]); HB = np.array([[.1, .25, .65], [.3, .6, .1], [.75, .25, 0]])


def viterbi(obs, pi0, A, B):
    with np.errstate(divide="ignore"):
        lA, lB = np.log(A), np.log(B)
    d = np.log(pi0) + lB[:, obs[0]]; back = []
    for o in obs[1:]:
        s = d[:, None] + lA; back.append(s.argmax(0)); d = s.max(0) + lB[:, o]
    path = [int(d.argmax())]
    for b in reversed(back): path.append(int(b[path[-1]]))
    return path[::-1]


class Markov(Scene):
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

    def construct(self):
        self.marks = {}
        title = Text("Cadenas de Markov", font=FONT, weight=BOLD, color=INK).scale(1.2)
        sub = Text("solo importa dónde estás ahora", font=FONT, color=MUTED).scale(0.5)
        VGroup(title, sub).arrange(DOWN, buff=0.35)
        self.mark("title"); self.play(Write(title), run_time=1.4)
        self.mark("subtitle"); self.play(FadeIn(sub, shift=0.2 * UP)); self.wait(1.2)
        self.mark("title_out"); self.play(FadeOut(title), FadeOut(sub))

        # ---------- 1. diagrama ----------
        cap = self.swap(None, self.caption("Un camión salta de un estado a otro", "cada hora, con las probabilidades de las flechas"))
        C = [np.array([-3.6, -0.9, 0]), np.array([0.6, 1.25, 0]), np.array([0.6, -3.0, 0])]
        R = 0.78
        nodes = VGroup(*[VGroup(Circle(radius=R, color=COLS[i], stroke_width=6).set_fill(BG, 1).move_to(C[i]),
                                Text(NAMES[i], font=FONT, weight=BOLD, color=INK).scale(0.4).move_to(C[i])) for i in range(3)]).set_z_index(3)
        arrows, labels, paths = VGroup(), VGroup(), {}
        loopdir = [LEFT, np.array([0.8, 0.6, 0]), np.array([0.8, -0.6, 0])]
        for i in range(3):
            for j in range(3):
                if i == j:
                    u = loopdir[i] / np.linalg.norm(loopdir[i])
                    rot = lambda v, a: np.array([v[0] * np.cos(a) - v[1] * np.sin(a), v[0] * np.sin(a) + v[1] * np.cos(a), 0])
                    a0, a1 = C[i] + R * rot(u, 0.5), C[i] + R * rot(u, -0.5)
                    c0, c1 = C[i] + (R + 1.0) * rot(u, 0.65), C[i] + (R + 1.0) * rot(u, -0.65)
                    pth = CubicBezier(a0, c0, c1, a1, color=MUTED, stroke_width=2 + 7 * P[i, j])
                    lp = C[i] + (R + 1.05) * u
                else:
                    v = (C[j] - C[i]) / np.linalg.norm(C[j] - C[i]); n = np.array([-v[1], v[0], 0])
                    ctrl = (C[i] + C[j]) / 2 + 0.5 * n
                    a = C[i] + R * (ctrl - C[i]) / np.linalg.norm(ctrl - C[i]); b = C[j] + (R + 0.05) * (ctrl - C[j]) / np.linalg.norm(ctrl - C[j])
                    pth = CubicBezier(a, a + (ctrl - a) * 2 / 3, b + (ctrl - b) * 2 / 3, b, color=MUTED, stroke_width=2 + 7 * P[i, j])
                    lp = (C[i] + C[j]) / 2 + 0.42 * n
                tip = ArrowTriangleFilledTip(color=MUTED, length=0.22, width=0.2)
                end = pth.point_from_proportion(1); prev = pth.point_from_proportion(0.96)
                ang = np.arctan2(*(end - prev)[[1, 0]])
                tip.rotate(ang).move_to(end - 0.08 * (end - prev) / np.linalg.norm(end - prev))
                arrows.add(VGroup(pth, tip)); paths[(i, j)] = pth
                labels.add(Text(f"{int(round(P[i, j] * 100))} %", font=FONT, weight=SEMIBOLD, color=INK).scale(0.34).move_to(lp))
        self.mark("diagram"); self.play(LaggedStart(*[GrowFromCenter(n) for n in nodes], lag_ratio=0.2), run_time=1.0)
        self.play(Create(arrows), FadeIn(labels), run_time=1.4)
        truck = VGroup(RoundedRectangle(width=0.42, height=0.24, corner_radius=0.04, stroke_width=0, fill_color=INK, fill_opacity=1),
                       RoundedRectangle(width=0.15, height=0.17, corner_radius=0.03, stroke_width=0, fill_color=INK, fill_opacity=1).shift([0.29, -0.035, 0]),
                       Dot([-0.12, -0.14, 0], radius=0.055, color=INK), Dot([0.24, -0.14, 0], radius=0.055, color=INK)).set_z_index(5)
        truck.move_to(C[0] + [0, R + 0.25, 0])
        self.play(FadeIn(truck), run_time=0.4)
        seq = [0, 0, 1, 0, 2, 2, 1, 0]
        for a, b in zip(seq[:-1], seq[1:]):
            self.mark("hop")
            self.play(MoveAlongPath(truck, paths[(a, b)]), run_time=0.75, rate_func=smooth)
            self.play(truck.animate.move_to(C[b] + [0, R + 0.25, 0]), run_time=0.2)
        self.wait(0.4)

        # ---------- 2. la tabla ----------
        cap = self.swap(cap, self.caption("Toda la cadena cabe en una tabla", "cada fila: desde dónde; cada columna: hacia dónde"))
        diag = VGroup(nodes, arrows, labels, truck)
        self.play(diag.animate.scale(0.62).move_to([-3.9, -1.0, 0]), run_time=1.0)
        cells = VGroup()
        hdr = VGroup(*[Text(n, font=FONT, color=COLS[j]).scale(0.34) for j, n in enumerate(NAMES)])
        x0, y0, cw, ch = 1.5, 0.9, 1.55, 0.8
        for j, h in enumerate(hdr): h.move_to([x0 + j * cw, y0 + 0.65, 0])
        rowl = VGroup(*[Text(n, font=FONT, color=COLS[i]).scale(0.34).move_to([x0 - 1.45, y0 - i * ch, 0]) for i, n in enumerate(NAMES)])
        for i in range(3):
            for j in range(3):
                r = RoundedRectangle(width=cw - 0.12, height=ch - 0.12, corner_radius=0.08, stroke_width=0, fill_color=COLS[j], fill_opacity=0.15 + 0.7 * P[i, j]).move_to([x0 + j * cw, y0 - i * ch, 0])
                cells.add(VGroup(r, Text(f"{int(round(P[i, j] * 100))} %", font=FONT, weight=SEMIBOLD, color=INK).scale(0.36).move_to(r)))
        self.mark("table"); self.play(FadeIn(hdr), FadeIn(rowl), LaggedStart(*[FadeIn(c, scale=0.8) for c in cells], lag_ratio=0.08), run_time=1.6)
        sumt = Text("cada fila suma 100 %", font=FONT, weight=SEMIBOLD, color=GOOD).scale(0.4).move_to([x0 + cw, y0 - 3 * ch + 0.05, 0])
        hl = SurroundingRectangle(VGroup(*cells[0:3]), color=INK, buff=0.06, corner_radius=0.1, stroke_width=3)
        self.mark("row"); self.play(Create(hl), FadeIn(sumt), run_time=0.8); self.wait(1.4)
        self.play(FadeOut(VGroup(diag, cells, hdr, rowl, hl, sumt)), run_time=0.6)

        # ---------- 3. evolución ----------
        cap = self.swap(cap, self.caption("Predecir horas más allá", "el reparto se mezcla y olvida dónde empezó"))
        ax0, ay0, H, bw, gap = -5.6, -3.4, 4.6, 0.62, 0.22
        def stack(p, k, start):
            x = ax0 + k * (bw + gap); g = VGroup(); y = ay0
            for i in range(3):
                h = p[i] * H
                if h > 1e-3:
                    g.add(Rectangle(width=bw, height=h, stroke_width=0, fill_color=COLS[i], fill_opacity=0.9).move_to([x, y + h / 2, 0]))
                y += h
            return g
        lines = VGroup(*[DashedLine([ax0 - 0.4, ay0 + v * H, 0], [ax0 + 12 * (bw + gap) + 0.3, ay0 + v * H, 0], color=INK, stroke_width=2, dash_length=0.1) for v in (0.5, 0.75)])
        legend = VGroup(*[VGroup(Square(0.22, stroke_width=0, fill_color=COLS[i], fill_opacity=1), Text(NAMES[i], font=FONT, color=INK).scale(0.32)).arrange(RIGHT, buff=0.12) for i in range(3)]).arrange(RIGHT, buff=0.45).move_to([2.6, 1.95, 0])
        for start in (0, 2):
            p = np.eye(3)[start]; bars = VGroup(); labs = VGroup()
            tag = Text("empieza " + ["en ruta", "", "cargando"][start], font=FONT, weight=SEMIBOLD, color=COLS[start]).scale(0.42).move_to([-3.7, 1.95, 0])
            self.mark("evo")
            self.play(FadeIn(tag), FadeIn(legend) if start == 0 else Wait(0.01), run_time=0.5)
            for k in range(12):
                b = stack(p, k, start); bars.add(b)
                labs.add(Text("ahora" if k == 0 else f"{k}h", font=FONT, color=MUTED).scale(0.26).move_to([ax0 + k * (bw + gap), ay0 - 0.25, 0]))
                self.play(FadeIn(b, shift=0.15 * UP), FadeIn(labs[-1]), run_time=0.22 if k else 0.4)
                p = p @ P
            self.play(Create(lines), run_time=0.6)
            fin = Text("a la larga: 50 % · 25 % · 25 %", font=FONT, weight=SEMIBOLD, color=GOOD).scale(0.42).move_to([2.6, 1.45, 0])
            self.play(FadeIn(fin), run_time=0.5); self.wait(1.0)
            self.play(FadeOut(VGroup(bars, labs, tag, lines, fin)), run_time=0.5)
        self.play(FadeOut(legend), run_time=0.3)

        # ---------- 4. tiempo hasta cargar ----------
        cap = self.swap(cap, self.caption("¿Cuánto falta para la próxima carga?", "desde en ruta: 8 horas de media, pero muy variable"))
        Q = P[:2, :2]; q = np.array([1.0, 0.0]); ex = []
        for k in range(1, 31):
            ex.append(q @ P[:2, 2]); q = q @ Q
        axl = Line([-6.2, -3.3, 0], [6.2, -3.3, 0], color=LINE, stroke_width=3)
        bars = VGroup(*[Rectangle(width=0.32, height=max(0.01, v * 38), stroke_width=0, fill_color=POINT, fill_opacity=0.85).move_to([-6.0 + k * 0.41, -3.3 + v * 19, 0]) for k, v in enumerate(ex)])
        ticks = VGroup(*[Text(f"{k}h", font=FONT, color=MUTED).scale(0.26).move_to([-6.0 + (k - 1) * 0.41, -3.55, 0]) for k in (1, 5, 10, 15, 20, 25, 30)])
        self.mark("hist"); self.play(Create(axl), FadeIn(ticks), LaggedStart(*[GrowFromEdge(b, DOWN) for b in bars], lag_ratio=0.05), run_time=2.0)
        mx = -6.0 + 7 * 0.41
        mean = DashedLine([mx, -3.3, 0], [mx, 1.2, 0], color=INK, stroke_width=3)
        mt = Text("media: 8 h", font=FONT, weight=SEMIBOLD, color=INK).scale(0.42).next_to(mean, UP, buff=0.1)
        self.mark("mean"); self.play(Create(mean), FadeIn(mt), run_time=0.8)
        tail = Text("1 de cada 10 tarda más de 17 h", font=FONT, weight=SEMIBOLD, color=WARN).scale(0.4).move_to([3.0, -0.4, 0])
        self.play(FadeIn(tail), run_time=0.5); self.wait(1.4)
        self.play(FadeOut(VGroup(axl, bars, ticks, mean, mt, tail)), run_time=0.6)

        # ---------- 5. modelo oculto ----------
        cap = self.swap(cap, self.caption("Lo que no se ve: el modelo oculto", "el GPS solo dice detenido, lento o rápido"))
        rng = np.random.default_rng(5)
        while True:
            Z, X = [0], []
            for t in range(40):
                if t: Z.append(rng.choice(3, p=HA[Z[-1]]))
                X.append(rng.choice(3, p=HB[Z[-1]]))
            V = viterbi(X, np.array([.5, .25, .25]), HA, HB)
            naive = [2 - x for x in X]
            accn = np.mean(np.array(naive) == Z); accv = np.mean(np.array(V) == Z)
            if min(np.bincount(Z, minlength=3)) >= 5 and accv - accn >= 0.15:
                break
        cw = 0.27; xs = -4.55 + np.arange(40) * cw
        gps = VGroup(*[Rectangle(width=cw - 0.05, height=[0.25, 0.6, 1.0][o], stroke_width=0, fill_color=MUTED, fill_opacity=0.85).move_to([x, 0.7 + [0.25, 0.6, 1.0][o] / 2, 0]) for x, o in zip(xs, X)])
        def row(seq, y): return VGroup(*[Rectangle(width=cw, height=0.42, stroke_width=0, fill_color=COLS[s], fill_opacity=0.9).move_to([x, y, 0]) for x, s in zip(xs, seq)])
        lab = lambda s, y: Text(s, font=FONT, color=INK).scale(0.32).move_to([-6.3, y, 0], aligned_edge=LEFT)
        hidden = VGroup(*[Rectangle(width=cw, height=0.42, stroke_width=0, fill_color=LINE, fill_opacity=1).move_to([x, -0.3, 0]) for x in xs])
        qm = Text("?", font=FONT, weight=BOLD, color=MUTED).scale(0.5).move_to([0.7, -0.3, 0])
        extra = VGroup(lab("GPS", 1.1), lab("estado", -0.3))
        self.mark("gps"); self.play(LaggedStart(*[GrowFromEdge(g, DOWN) for g in gps], lag_ratio=0.02), FadeIn(hidden), FadeIn(qm), FadeIn(extra), run_time=1.4)
        rn, rv, rt = row(naive, -1.3), row(V, -2.2), row(Z, -3.1)
        ln = VGroup(lab("ingenua", -1.3)); lv = VGroup(lab("oculto", -2.2)); lt = VGroup(lab("real", -3.1))
        for g in (ln, lv, lt): g.shift(LEFT * 0.0)
        self.mark("naive"); self.play(FadeIn(rn, lag_ratio=0.02), FadeIn(ln), run_time=0.9)
        self.mark("vit"); self.play(FadeIn(rv, lag_ratio=0.02), FadeIn(lv), run_time=0.9)
        self.mark("truth"); self.play(FadeIn(rt, lag_ratio=0.02), FadeIn(lt), FadeOut(qm), run_time=0.9)
        sc = VGroup(Text(f"regla ingenua: {accn:.0%}", font=FONT, color=INK).scale(0.38), Text(f"modelo oculto (Viterbi): {accv:.0%}", font=FONT, weight=SEMIBOLD, color=GOOD).scale(0.42)).arrange(RIGHT, buff=0.6).move_to([0.4, -3.72, 0])
        self.mark("score"); self.play(FadeIn(sc), run_time=0.6); self.wait(1.8)
        self.play(FadeOut(VGroup(gps, hidden, rn, rv, rt, ln, lv, lt, sc, extra)), run_time=0.6)

        # ---------- Cierre ----------
        self.mark("legend")
        self.play(FadeOut(cap), run_time=0.6)
        e1 = Text("Solo el presente decide el siguiente paso.", font=FONT, weight=BOLD, color=INK).scale(0.7)
        e2 = Text("Y con eso basta para ver el largo plazo.", font=FONT, color=MUTED).scale(0.48)
        VGroup(e1, e2).arrange(DOWN, buff=0.4)
        self.mark("end_text"); self.play(Write(e1), run_time=1.4); self.play(FadeIn(e2, shift=0.2 * UP), run_time=0.8)
        self.wait(2.0)
        self.mark("fade"); self.play(FadeOut(VGroup(e1, e2)), run_time=1)
        self.marks["total"] = [round(self.renderer.time, 3)]
        json.dump(self.marks, open("marks.json", "w"), indent=1)
