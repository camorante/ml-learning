import json
import sys
import numpy as np
from manim import *

# Dark palette of the series
BG = "#0E131B"; INK = "#E4EAF3"; MUTED = "#9BA7B9"; LINE = "#2B3545"
MEAN = "#6EA6FF"; SAMPLE = "#AE9FF3"; POINT = "#FF9A55"; GOOD = "#4CC38A"; WARN = "#F08A5D"
FONT = "Inter"
config.background_color = BG

sys.path.insert(0, "../../rn/r5/py")
from datos import carga_consumo, red_inicial, ETA   # noqa: E402
C, Y = carga_consumo()
CAP = json.load(open("../../rn/r5/py/capas.json", encoding="utf-8"))
COSTE = json.load(open("../../rn/r5/py/coste.json", encoding="utf-8"))

# the video training steps: the same as idea 6 of the simple page (η = 0.03)
red = red_inicial(); PASOS = []
for s in range(5):
    z = red.adelante(C)[:, 0]
    PASOS.append((red.capas[0].W.copy(), red.capas[0].b.copy(), float(red.perdida_media(C, Y)), int(np.sum((z > 0) != Y))))
    red.paso(C, Y, ETA)
sig = lambda z: 1 / (1 + np.exp(-z))


def T(s, sc=0.45, col=INK, w=None):
    return Text(s, font=FONT, color=col, weight=w or NORMAL).scale(sc)


def fmt(v, d=1):
    s = f"{abs(v):.{d}f}"
    return ("−" if v < 0 and round(abs(v), d) != 0 else "") + s


class Panel:
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

    def line(self, w, b, col, width=5):
        """Line w0·load + w1·fuel + b = 0 inside the panel (fuel use as a function of load)."""
        f = lambda c: -(w[0] * c + b) / w[1]
        return Line(self.P(self.x0, f(self.x0)), self.P(self.x1, f(self.x1)), stroke_color=col, stroke_width=width)


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

    def band(self, box):
        P = Panel(0, 25, 18, 42, box, xt=(0, 5, 10, 15, 20, 25), xf=lambda v: f"{v} t", xl="load",
                  yt=(20, 25, 30, 35, 40), yl="fuel use (L/100 km)")
        dots = VGroup(*[Dot(P.P(c, q), radius=0.075, color=POINT if y else SAMPLE) for (c, q), y in zip(C, Y)])
        return P, dots

    def rings(self, P, W, b):
        h = sig(C @ W.T + b); p = sig(10 * h.sum(1) - 5)
        return VGroup(*[Circle(radius=0.15, color=WARN, stroke_width=3).move_to(P.P(c, q))
                        for (c, q), y, pp in zip(C, Y, p) if (pp > 0.5) != (y == 1)])

    def construct(self):
        self.marks = {}
        title = Text("Backpropagation", font=FONT, weight=BOLD, color=INK).scale(0.95)
        sub = Text("the chain rule, backwards", font=FONT, color=MUTED).scale(0.46)
        VGroup(title, sub).arrange(DOWN, buff=0.35)
        self.mark("title"); self.play(Write(title), run_time=1.4)
        self.mark("subtitle"); self.play(FadeIn(sub, shift=0.2 * UP)); self.wait(1.4)
        self.mark("title_out"); self.play(FadeOut(title), FadeOut(sub))

        # ---------- 1. the badly set-up network ----------
        cap = self.swap(None, self.caption("A badly set-up network", "line B is 3.5 L too low"))
        P, dots = self.band((-5.6, 1.4, -2.9, 1.9))
        W0, b0, L0, e0 = PASOS[0]
        lA, lB = P.line(W0[0], b0[0], MEAN), P.line(W0[1], b0[1], GOOD)
        self.play(FadeIn(P.g), FadeIn(dots, lag_ratio=0.03), run_time=1.2)
        self.mark("lines"); self.play(Create(lA), Create(lB), run_time=1.0)
        la = T("A", 0.42, MEAN, BOLD).next_to(lA.get_end(), LEFT, buff=0.15).shift(0.4 * DOWN)
        lb = T("B", 0.42, GOOD, BOLD).next_to(lB.get_end(), LEFT, buff=0.15).shift(0.3 * DOWN)
        self.play(FadeIn(la), FadeIn(lb), run_time=0.4)
        rg = self.rings(P, W0, b0)
        self.mark("miss"); self.play(Create(rg, lag_ratio=0.15), run_time=1.2)
        info = VGroup(T("6 trips that burn", 0.38, INK, BOLD), T("too little get away", 0.38, INK, BOLD),
                      T(f"average loss {fmt(L0, 2)}", 0.38, POINT, BOLD)).arrange(DOWN, aligned_edge=LEFT, buff=0.15).move_to([2.4, 0.2, 0], aligned_edge=LEFT)
        self.play(FadeIn(info, shift=0.1 * RIGHT), run_time=0.7); self.wait(3.0)
        self.clear_but(cap)

        # ---------- 2. nudging each weight: the cost ----------
        cap = self.swap(cap, self.caption("Which weight is to blame?", "nudging each weight by hand costs 2 passes per weight"))
        big = COSTE[-1]
        Pc = Panel(-5, 6, 0, 2, (-5.6, 5.6, -2.0, 1.2), xt=(-5, -3, -1, 1, 3, 5),
                   xf=lambda v: {-5: "10 µs", -3: "1 ms", -1: "0.1 s", 1: "10 s", 3: "17 min", 5: "28 h"}[v], xl="time (factors of 10)")
        hdr = T(f"a network with {fmt(big['P'] / 1e6, 1)} million weights, 30 trips", 0.4, INK, BOLD).move_to([0, 1.75, 0])
        self.mark("cost"); self.play(FadeIn(Pc.g), FadeIn(hdr), run_time=0.8)
        def hbar(t, y, col):
            x1 = Pc.P(np.log10(t), 0)[0]
            return Rectangle(width=x1 - Pc.L, height=0.62, stroke_width=0, fill_color=col, fill_opacity=0.88).move_to([(Pc.L + x1) / 2, y, 0], aligned_edge=ORIGIN)
        bf = hbar(big["tf"], 0.45, SAMPLE); bb = hbar(big["tb"], -0.85, GOOD)
        tf = T(f"nudging each weight: {fmt(big['tf'] / 3600, 1)} h", 0.36, BG, BOLD).move_to(bf.get_left() + 0.2 * RIGHT, aligned_edge=LEFT)
        tb = T(f"backpropagation: {fmt(big['tb'] * 1e3, 0)} ms", 0.36, INK, BOLD).next_to(bb, RIGHT, buff=0.2)
        self.play(GrowFromEdge(bf, LEFT), run_time=1.4); self.play(FadeIn(tf), run_time=0.4)
        self.mark("fast"); self.play(GrowFromEdge(bb, LEFT), run_time=0.6); self.play(FadeIn(tb), run_time=0.4)
        nt = T(f"all the blames for the price of about {fmt(big['tb'] / big['tp'], 0)} passes", 0.38, GOOD, BOLD).move_to([0, -3.3, 0])
        self.play(FadeIn(nt), run_time=0.6); self.wait(3.0)
        self.clear_but(cap)

        # ---------- 3. backward through the network ----------
        cap = self.swap(cap, self.caption("Backward, the blame multiplies", "trip 3: 4.8 t and 22.6 L, doesn’t add up"))
        c, q = C[2]
        zA, zB = -1.1 * c + 2 * q - 53, 1.1 * c - 2 * q + 36
        hA, hB = sig(zA), sig(zB); z = 10 * hA + 10 * hB - 5; p = sig(z)
        e = p - 1; sB = hB * (1 - hB); dB = e * 10 * sB; dA = e * 10 * hA * (1 - hA)
        pos = {"c": [-5.4, 0.85, 0], "q": [-5.4, -1.65, 0], "A": [-1.6, 1.15, 0], "B": [-1.6, -1.9, 0], "o": [1.9, -0.4, 0], "L": [5.0, -0.4, 0]}
        R = {"c": 0.55, "q": 0.55, "A": 0.75, "B": 0.75, "o": 0.75, "L": 0.6}
        def edge(a, b, col=MUTED, w=3):
            u = np.array(pos[b]) - np.array(pos[a]); u = u / np.linalg.norm(u)
            return Line(np.array(pos[a]) + R[a] * u, np.array(pos[b]) - R[b] * u, stroke_color=col, stroke_width=w)
        E = {k: edge(*k) for k in [("c", "A"), ("c", "B"), ("q", "A"), ("q", "B"), ("A", "o"), ("B", "o"), ("o", "L")]}
        circ = {k: Circle(radius=R[k], color=col, stroke_width=4).move_to(pos[k]) for k, col in
                [("c", INK), ("q", INK), ("A", MEAN), ("B", GOOD), ("o", SAMPLE), ("L", POINT)]}
        val = {"c": T("4.8 t", 0.36, INK, BOLD), "q": T("22.6 L", 0.36, INK, BOLD), "A": T(f"{fmt(100 * hA, 4)}%", 0.34, INK, BOLD),
               "B": T(f"{fmt(100 * hB, 1)}%", 0.36, INK, BOLD), "o": T(f"{fmt(100 * p, 1)}%", 0.36, INK, BOLD), "L": T(fmt(-np.log(p), 2), 0.4, INK, BOLD)}
        for k in val:
            val[k].move_to(pos[k])
        lab = VGroup(T("load", 0.3, MUTED).next_to(circ["c"], UP, buff=0.1), T("fuel use", 0.3, MUTED).next_to(circ["q"], UP, buff=0.1),
                     T("A", 0.34, MEAN, BOLD).next_to(circ["A"], UP, buff=0.1), T("B", 0.34, GOOD, BOLD).next_to(circ["B"], DOWN, buff=0.1),
                     T("output", 0.3, MUTED).next_to(circ["o"], UP, buff=0.1), T("loss", 0.3, MUTED).next_to(circ["L"], UP, buff=0.1))
        self.mark("net")
        self.play(*[Create(m) for m in E.values()], *[Create(m) for m in circ.values()], FadeIn(lab), FadeIn(val["c"]), FadeIn(val["q"]), run_time=1.2)
        self.mark("fwd")
        for ks in [("A", "B"), ("o",), ("L",)]:
            self.play(*[FadeIn(val[k], scale=1.2) for k in ks], run_time=0.5)
        self.wait(0.6)
        # backward
        def blame(k, s, d=DOWN, buff=0.12):
            return T(s, 0.36, POINT, BOLD).next_to(circ[k], d, buff=buff)
        b1 = blame("o", f"p − y = {fmt(e, 2)}")
        self.mark("back"); self.play(E[("o", "L")].animate.set_color(POINT), FadeIn(b1, shift=0.1 * LEFT), run_time=0.8); self.wait(0.6)
        k1 = T("× 10", 0.34, POINT, BOLD).move_to(E[("B", "o")].get_center() + 0.35 * DOWN + 0.25 * RIGHT)
        k2 = T(f"× slope {fmt(sB, 3)}", 0.34, POINT, BOLD).next_to(circ["B"], RIGHT, buff=0.15).shift(0.55 * DOWN)
        b2 = T(f"δ = {fmt(dB, 2)}", 0.38, POINT, BOLD).next_to(circ["B"], RIGHT, buff=0.15).shift(0.05 * UP)
        self.mark("back"); self.play(E[("B", "o")].animate.set_color(POINT), FadeIn(k1), run_time=0.7)
        self.play(FadeIn(k2), FadeIn(b2), run_time=0.7); self.wait(1.0)
        bA = T("A saturated: ≈ 0", 0.34, MUTED, BOLD).next_to(circ["A"], RIGHT, buff=0.15)
        self.mark("back"); self.play(E[("A", "o")].animate.set_color(LINE), FadeIn(bA), run_time=0.7); self.wait(0.9)
        k3 = T(f"× 22.6 = {fmt(dB * q, 2)}", 0.38, POINT, BOLD).move_to(E[("q", "B")].get_center() + 0.35 * DOWN)
        self.mark("weight"); self.play(E[("q", "B")].animate.set_color(POINT).set_stroke(width=8), FadeIn(k3), run_time=0.8)
        rule = T("blame of a weight = signal in × error back", 0.4, INK, BOLD).move_to([0, -3.55, 0])
        self.play(FadeIn(rule, shift=0.1 * UP), run_time=0.6); self.wait(3.0)
        self.clear_but(cap)

        # ---------- 4. one step, and another ----------
        cap = self.swap(cap, self.caption("One step: weight − 0.03 × blame", "all 30 trips, every weight at once"))
        P, dots = self.band((-5.6, 0.9, -2.5, 1.9))
        Wk, bk, Lk, ek = PASOS[0]
        lA, lB = P.line(Wk[0], bk[0], MEAN), P.line(Wk[1], bk[1], GOOD)
        ghost = DashedLine(lB.get_start(), lB.get_end(), color=MUTED, stroke_width=3)
        rg = self.rings(P, Wk, bk)
        Pl = Panel(0, 4, 0, 1, (2.4, 6.2, -2.5, 1.2), xt=(0, 1, 2, 3, 4), xl="steps", yt=(0, 0.5, 1), yf=lambda v: fmt(v, 1), yl="average loss")
        self.play(FadeIn(P.g), FadeIn(dots), Create(lA), Create(lB), FadeIn(ghost), FadeIn(rg), FadeIn(Pl.g), run_time=1.0)
        pts = [Dot(Pl.P(0, Lk), radius=0.08, color=POINT)]
        ro = T(f"loss {fmt(Lk, 2)} · {ek} mistake{'' if ek == 1 else 's'}", 0.38, INK, BOLD).move_to([4.3, 2.05, 0])
        self.play(FadeIn(pts[0]), FadeIn(ro), run_time=0.4)
        for s in range(1, 5):
            Wk, bk, Lk, ek = PASOS[s]
            nB = P.line(Wk[1], bk[1], GOOD); nA = P.line(Wk[0], bk[0], MEAN); nr = self.rings(P, Wk, bk)
            d = Dot(Pl.P(s, Lk), radius=0.08, color=POINT); seg = Line(pts[-1].get_center(), d.get_center(), color=POINT, stroke_width=4)
            nro = T(f"loss {fmt(Lk, 2)} · {ek} mistake{'' if ek == 1 else 's'}", 0.38, INK if ek else GOOD, BOLD).move_to(ro)
            self.mark("step")
            self.play(Transform(lB, nB), Transform(lA, nA), Transform(rg, nr), Create(seg), FadeIn(d), Transform(ro, nro), run_time=0.9)
            pts.append(d); self.wait(0.9)
        nb = T("line B rises on its own: nearly all the blame was its own", 0.38, GOOD, BOLD).move_to([0, -3.6, 0])
        self.play(FadeIn(nb), run_time=0.6); self.wait(3.0)
        self.clear_but(cap)

        # ---------- 5. many layers ----------
        cap = self.swap(cap, self.caption("Many layers: the blame fades", "10 hidden layers, blame of each layer ÷ the output’s"))
        Pd = Panel(0.4, 11.6, -8, 0.3, (-5.4, 5.6, -2.9, 1.5), xt=(1, 3, 5, 7, 9, 11), xf=lambda v: "output" if v == 11 else f"layer {v}",
                   yt=(-8, -6, -4, -2, 0), yf=lambda v: "1" if v == 0 else f"10^{v}")
        def bars(k, col):
            g = VGroup()
            for j, v in enumerate(CAP[k]["g"]):
                top, bot = Pd.P(j + 1, max(-8, np.log10(v))), Pd.P(j + 1, -8)
                g.add(Rectangle(width=0.55, height=max(0.02, top[1] - bot[1]), stroke_width=0, fill_color=col, fill_opacity=0.88).move_to((top + bot) / 2))
            return g
        bs = bars("sigmoide", MEAN); br = bars("relu", GOOD)
        ls = T(f"sigmoid: layer 1 receives 1 / {fmt(1 / CAP['sigmoide']['g'][0] / 1e6, 1)} million", 0.38, MEAN, BOLD).move_to([0.2, -3.5, 0])
        self.mark("deep"); self.play(FadeIn(Pd.g), run_time=0.6)
        self.play(LaggedStart(*[GrowFromEdge(b, DOWN) for b in bs], lag_ratio=0.08), run_time=1.4); self.play(FadeIn(ls), run_time=0.5); self.wait(2.4)
        lr = T(f"ReLU: it receives {fmt(CAP['relu']['g'][0], 2)} of the output’s blame", 0.38, GOOD, BOLD).move_to(ls)
        self.mark("relu"); self.play(Transform(bs, br), Transform(ls, lr), run_time=1.2); self.wait(3.0)
        self.clear_but(cap)

        # ---------- closing ----------
        self.mark("legend")
        self.play(FadeOut(cap), run_time=0.6)
        e1 = Text("Forward to answer, backward to share out the blame", font=FONT, weight=BOLD, color=INK).scale(0.6)
        e2 = VGroup(T("at the output, the blame is p − y", 0.42, MUTED),
                    T("each piece multiplies it by its slope; paths add up", 0.42, MUTED),
                    T("blame of a weight = signal in × error back", 0.42, MUTED),
                    T("all the blames for about 2 or 3 passes, whether the network has 9 weights or millions", 0.42, MUTED)).arrange(DOWN, buff=0.18)
        VGroup(e1, e2).arrange(DOWN, buff=0.45)
        self.mark("end_text"); self.play(Write(e1), run_time=1.2); self.play(FadeIn(e2, shift=0.2 * UP, lag_ratio=0.15), run_time=1.0)
        self.wait(4.0)
        self.mark("fade"); self.play(FadeOut(VGroup(e1, e2)), run_time=1)
        self.marks["total"] = [round(self.renderer.time, 3)]
        json.dump(self.marks, open("marks.json", "w"), indent=1)
