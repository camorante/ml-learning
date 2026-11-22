import json, math
import numpy as np
from manim import *

# Dark palette of the series
BG = "#0E131B"; INK = "#E4EAF3"; MUTED = "#9BA7B9"; LINE = "#2B3545"
MEAN = "#6EA6FF"; SAMPLE = "#AE9FF3"; POINT = "#FF9A55"; GOOD = "#4CC38A"; WARN = "#F08A5D"
FONT = "Inter"
config.background_color = BG
D = json.load(open("../../rn/r9-en/embed.json", encoding="utf-8"))   # the same data as the pages
L0 = D["l0"]


def T(s, sc=0.45, col=INK, w=None):
    return Text(s, font=FONT, color=col, weight=w or NORMAL).scale(sc)


def fmt(v, d=1):
    s = f"{abs(v):.{d}f}"
    return ("−" if v < 0 and round(abs(v), d) != 0 else "") + s


def sup(n):
    return "".join("⁻" if ch == "-" else "⁰¹²³⁴⁵⁶⁷⁸⁹"[int(ch)] for ch in str(n))


def pct(p, d=0):
    return fmt(100 * round(p, d + 2), d) + "%"


class Panel:
    """Simple 2D axes: maps data (x, y) to the scene inside a box."""

    def __init__(self, x0, x1, y0, y1, box, xt=(), xf=None, xl=None, yt=(), yf=None, yl=None):
        self.x0, self.x1, self.y0, self.y1 = x0, x1, y0, y1
        self.L, self.R, self.B, self.Tp = box
        g = VGroup(Line([self.L, self.B, 0], [self.R, self.B, 0], stroke_color=MUTED, stroke_width=2))
        for v in xt:
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

    def poly(self, pts, col, w=4):
        q = [self.P(min(self.x1, max(self.x0, x)), min(self.y1, max(self.y0, y))) for x, y in pts]
        return VMobject(stroke_color=col, stroke_width=w).set_points_as_corners(q)


def senal_panel(box, n, yl="acceleration (m/s²)"):
    st = 10 if n <= 70 else 20
    return Panel(0.5, n + 0.5, -6, 4, box, xt=tuple(range(st, n + 1, st)), xf=lambda v: fmt(v / 10, 0) + " s",
                 yt=(-6, -3, 0, 3), yf=lambda v: fmt(v, 0), yl=yl)


def zona(P):
    return Polygon(P.P(0.5, -6), P.P(L0 + 0.5, -6), P.P(L0 + 0.5, 4), P.P(0.5, 4), stroke_width=0, fill_color=GOOD, fill_opacity=0.16)


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
        title = Text("Recurrent networks and LSTM", font=FONT, weight=BOLD, color=INK).scale(0.95)
        sub = Text("reading the trip step by step, with a memory", font=FONT, color=MUTED).scale(0.46)
        VGroup(title, sub).arrange(DOWN, buff=0.35)
        self.mark("title"); self.play(Write(title), run_time=1.4)
        self.mark("subtitle"); self.play(FadeIn(sub, shift=0.2 * UP)); self.wait(1.4)
        self.mark("title_out"); self.play(FadeOut(title), FadeOut(sub))

        # ---------- 1. reading step by step ----------
        cap = self.swap(None, self.caption("A network that reads step by step", "at each reading it mixes the new input with its memory"))
        cajas = VGroup(); flechas = VGroup(); xs = VGroup()
        for j in range(6):
            c = RoundedRectangle(width=1.15, height=0.62, corner_radius=0.12, stroke_color=POINT if j == 5 else LINE, stroke_width=3, fill_color=BG, fill_opacity=1).move_to([-5.0 + 2.0 * j, 1.75, 0])
            c.add(T("h" if j == 5 else f"h{'₁₂₃₄₅'[j]}", 0.4, INK, BOLD).move_to(c))
            cajas.add(c); xs.add(T("…" if j == 4 else ("x" if j == 5 else f"x{'₁₂₃₄'[j]}"), 0.36, MUTED).next_to(c, DOWN, buff=0.22))
            if j:
                flechas.add(Arrow(cajas[j - 1].get_right(), c.get_left(), buff=0.05, stroke_width=3, color=MUTED, max_tip_length_to_length_ratio=0.25))
        xs[4].set_opacity(0); cajas[4].set_opacity(0.0)
        dots = T("…", 0.6, INK, BOLD).move_to(cajas[4])
        mismo = T("the same weights at every step", 0.32, GOOD, BOLD).move_to([0, 0.75, 0])
        self.play(LaggedStart(*[FadeIn(VGroup(cajas[j], xs[j])) for j in (0, 1, 2, 3, 5)], lag_ratio=0.2), FadeIn(dots), run_time=1.4)
        self.mark("chain"); self.play(LaggedStart(*[GrowArrow(a) for a in flechas], lag_ratio=0.2), FadeIn(mismo), run_time=1.2)
        x = D["trazas"]["25"]["brusca"]; n = len(x)
        Ps = senal_panel((-6.0, 6.0, -3.35, -0.75), n)
        q = ValueTracker(1)
        self.play(FadeIn(Ps.g), FadeIn(zona(Ps)), run_time=0.6)
        tr = always_redraw(lambda: Ps.poly(list(enumerate(x[:max(2, int(q.get_value()))], 1)), INK, 3.5))
        cur = always_redraw(lambda: Line(Ps.P(int(q.get_value()), -6), Ps.P(int(q.get_value()), 4), stroke_color=POINT, stroke_width=4))
        self.add(tr, cur)
        self.mark("read"); self.play(q.animate.set_value(n), run_time=3.6, rate_func=linear)
        self.wait(0.8)
        self.clear_but(cap)

        # ---------- 2. the RNN forgets; the LSTM remembers ----------
        cap = self.swap(cap, self.caption("Was there a harsh brake at the start?", "the answer comes at the end; a 5 s gap in between"))
        x = D["trazas"]["50"]["brusca"]; n = len(x); R = D["run"]["50"]["brusca"]
        Ps = senal_panel((-6.0, 6.0, -0.25, 1.95), n)
        zl = T("start: brakes hard", 0.3, GOOD, BOLD).move_to(Ps.P(L0 / 2 + 0.5, 3.3))
        self.play(FadeIn(Ps.g), FadeIn(zona(Ps)), FadeIn(zl), run_time=0.7)
        q = ValueTracker(1)
        tr = always_redraw(lambda: Ps.poly(list(enumerate(x[:max(2, int(q.get_value()))], 1)), INK, 3.5))
        cur = always_redraw(lambda: Line(Ps.P(int(q.get_value()), -6), Ps.P(int(q.get_value()), 4), stroke_color=POINT, stroke_width=4))

        def barra(clave, nombre, col, y):
            def f():
                k = int(q.get_value()); p = R[clave]["p"][k - 1]
                fondo_b = Rectangle(width=6.0, height=0.42, stroke_color=LINE, stroke_width=2).move_to([0.9, y, 0]).align_to([-2.1, 0, 0], LEFT)
                lleno = Rectangle(width=max(0.001, 6.0 * p), height=0.42, stroke_width=0, fill_color=col, fill_opacity=0.9).align_to(fondo_b, LEFT).align_to(fondo_b, DOWN)
                mitad = DashedLine(fondo_b.get_top() + 0.1 * UP, fondo_b.get_bottom() + 0.1 * DOWN, color=MUTED, stroke_width=2).move_to(fondo_b)
                et = T(nombre, 0.34, col, BOLD).next_to(fondo_b, LEFT, buff=0.25)
                pv = T(pct(p), 0.36, INK, BOLD).next_to(fondo_b, RIGHT, buff=0.2)
                return VGroup(fondo_b, lleno, mitad, et, pv)
            return always_redraw(f)
        b1 = barra("RNN", "RNN", MUTED, -1.95); b2 = barra("LSTM", "LSTM", POINT, -2.75)
        pt = T("harsh? what it would answer now", 0.32, MUTED).move_to([0.9, -1.38, 0])
        self.add(tr, cur)
        self.play(FadeIn(b1), FadeIn(b2), FadeIn(pt), run_time=0.6)
        self.mark("start"); self.play(q.animate.set_value(L0), run_time=2.0, rate_func=linear)
        self.mark("gap"); self.play(q.animate.set_value(n), run_time=4.2, rate_func=linear)
        acc = D["acc"]; gi = D["huecos"].index(50)
        msg = T(f"on 1,000 segments: RNN {fmt(100 * acc['RNN'][gi], 1)}%, like a coin · LSTM {fmt(100 * acc['LSTM'][gi], 1)}%", 0.34, GOOD, BOLD).move_to([0, -3.55, 0])
        self.mark("verdict"); self.play(FadeIn(msg), run_time=0.5); self.wait(2.0)
        self.clear_but(cap)

        # ---------- 3. the error vanishes ----------
        cap = self.swap(cap, self.caption("Why it forgets: the error dies out on the way back", "how much of the error reaches each reading, before training"))
        er = D["eco"]["RNN|inicio|50"]; el = D["eco"]["LSTM|inicio|50"]; n = len(er)
        Pe = Panel(0.5, n + 0.5, -18, 0.5, (-5.6, 6.0, -2.85, 1.95), xt=(10, 20, 30, 40, 50, 60), xf=lambda v: fmt(v / 10, 0) + " s",
                   yt=(0, -5, -10, -15), yf=lambda v: "1" if v == 0 else "10" + sup(v), yl="log scale: each line is 100,000 times less")
        zr = Polygon(Pe.P(0.5, -18), Pe.P(L0 + 0.5, -18), Pe.P(L0 + 0.5, 0.5), Pe.P(0.5, 0.5), stroke_width=0, fill_color=GOOD, fill_opacity=0.16)
        self.play(FadeIn(Pe.g), FadeIn(zr), run_time=0.7)
        cr = Pe.poly([(n - i, math.log10(er[n - 1 - i])) for i in range(n)], MUTED, 5)
        cl = Pe.poly([(n - i, math.log10(el[n - 1 - i])) for i in range(n)], POINT, 5)
        self.mark("echo"); self.play(Create(cr), Create(cl), run_time=3.2, rate_func=linear)
        lr = T(f"RNN: {fmt(er[0] / 10 ** math.floor(math.log10(er[0])), 1)} · 10{sup(math.floor(math.log10(er[0])))}", 0.34, MUTED, BOLD).next_to(Pe.P(1, math.log10(er[0])), UR, buff=0.12).shift(0.22 * UP + 0.25 * RIGHT)
        ll = T(f"LSTM: {fmt(el[0], 3)}", 0.34, POINT, BOLD).next_to(Pe.P(1, math.log10(el[0])), DR, buff=0.12).shift(0.5 * RIGHT)
        self.play(FadeIn(lr), FadeIn(ll), run_time=0.5)
        m3 = T("in the RNN the correction never reaches the start: it cannot learn to look there", 0.32, GOOD, BOLD).move_to([0.3, -3.65, 0])
        self.play(FadeIn(m3), run_time=0.5); self.wait(1.8)
        self.clear_but(cap)

        # ---------- 4. the conveyor belt and the gates ----------
        cap = self.swap(cap, self.caption("The LSTM: a conveyor belt and three gates", "forget, input and output, each between 0 and 1"))
        y0 = 1.55
        cinta = Arrow([-6.2, y0, 0], [6.2, y0, 0], buff=0, color=POINT, stroke_width=9, max_tip_length_to_length_ratio=0.03)
        ct = T("belt c: the memory moves along by adding", 0.32, POINT, BOLD).next_to(cinta, UP, buff=0.15).align_to(cinta, LEFT).shift(0.2 * RIGHT)
        gates = VGroup()
        for xg, sym, nm in ((-2.6, "×", "forget f"), (0.0, "+", "input i"), (2.6, "→", "output o")):
            c = Circle(radius=0.32, stroke_color=INK, stroke_width=3, fill_color=BG, fill_opacity=1).move_to([xg, y0, 0])
            gates.add(VGroup(c, T(sym, 0.45, INK, BOLD).move_to(c), T(nm, 0.3, MUTED).next_to(c, DOWN, buff=0.14)))
        self.mark("belt"); self.play(GrowArrow(cinta), FadeIn(ct), run_time=1.0)
        self.mark("gates"); self.play(LaggedStart(*[FadeIn(g, scale=0.7) for g in gates], lag_ratio=0.3), run_time=1.2)
        # the one-unit LSTM on the segment with a harsh brake (gap of 25)
        U1 = D["uno"]; x = D["tr3"]["brusca"]; n = len(x)
        sg = lambda z: 1 / (1 + math.exp(-z))
        h = c = 0.0; S = []
        for v in x:
            xs_ = v / 4
            i = sg(U1["W"][0] * xs_ + U1["U"][0] * h + U1["b"][0]); f = sg(U1["W"][1] * xs_ + U1["U"][1] * h + U1["b"][1])
            g = math.tanh(U1["W"][2] * xs_ + U1["U"][2] * h + U1["b"][2]); o = sg(U1["W"][3] * xs_ + U1["U"][3] * h + U1["b"][3])
            c = f * c + i * g; h = o * math.tanh(c); S.append((i, f, g, o, c, h))
        Pc = Panel(0.5, n + 0.5, -4, 4, (-6.0, 1.6, -3.4, -0.5), xt=(10, 20, 30, 40), xf=lambda v: fmt(v / 10, 0) + " s", yt=(-4, 0, 4), yf=lambda v: fmt(v, 0), yl="a one-neuron LSTM: its belt c")
        zc = Polygon(Pc.P(0.5, -4), Pc.P(L0 + 0.5, -4), Pc.P(L0 + 0.5, 4), Pc.P(0.5, 4), stroke_width=0, fill_color=GOOD, fill_opacity=0.16)
        self.play(FadeIn(Pc.g), FadeIn(zc), run_time=0.6)
        q = ValueTracker(1)
        cc = always_redraw(lambda: Pc.poly([(k + 1, S[k][4]) for k in range(max(2, int(q.get_value())))], POINT, 5))

        def lecturas():
            k = int(q.get_value()) - 1; i, f, g, o, c_, h_ = S[k]
            filas = [("reading", f"{fmt(x[k], 2)} m/s²", INK), ("input i", fmt(i, 2), POINT), ("forget f", fmt(f, 5 if f >= 0.99995 else 4), POINT), ("output o", fmt(o, 2), POINT), ("belt c", fmt(c_, 2), POINT)]
            gr = VGroup()
            for r, (a, b_, col) in enumerate(filas):
                gr.add(T(a, 0.32, MUTED).move_to([2.4, -0.7 - 0.55 * r, 0], aligned_edge=LEFT))
                gr.add(T(b_, 0.34, col, BOLD).move_to([6.1, -0.7 - 0.55 * r, 0], aligned_edge=RIGHT))
            return gr
        tb = always_redraw(lecturas)
        self.add(cc, tb)
        self.mark("write"); self.play(q.animate.set_value(9), run_time=2.6, rate_func=linear)
        n4 = T("hard braking: the input gate opens and writes a negative value", 0.3, GOOD, BOLD).move_to([0, 0.45, 0])
        self.play(FadeIn(n4), run_time=0.4); self.wait(1.0)
        n5 = T("then it closes on its own, and the forget gate (almost 1) keeps it", 0.3, GOOD, BOLD).move_to([0, 0.45, 0])
        self.mark("keep"); self.play(FadeOut(n4), FadeIn(n5), q.animate.set_value(n), run_time=3.2, rate_func=linear)
        self.wait(1.4)
        self.clear_but(cap)

        # ---------- 5. the race as the gap grows ----------
        cap = self.swap(cap, self.caption("The longer the gap…", "accuracy on 1,000 test segments"))
        HU = D["huecos"]
        Pa = Panel(-5, 105, 0.4, 1.02, (-5.2, 5.6, -3.0, 1.9), xt=tuple(HU), xf=lambda v: str(v), xl="gap (readings)",
                   yt=(0.5, 0.75, 1.0), yf=lambda v: fmt(100 * v, 0) + "%")
        self.play(FadeIn(Pa.g), run_time=0.6)
        lines = VGroup(); labs = VGroup()
        for k, col in (("RNN", MUTED), ("LSTM", POINT), ("GRU", MEAN)):
            pts = [(g_, a) for g_, a in zip(HU, D["acc"][k])]
            lines.add(Pa.poly(pts, col, 6))
        labs.add(T("RNN", 0.36, MUTED, BOLD).next_to(Pa.P(100, D["acc"]["RNN"][-1]), RIGHT, buff=0.15))
        labs.add(T("LSTM and GRU", 0.32, POINT, BOLD).next_to(Pa.P(100, D["acc"]["LSTM"][-1]), RIGHT, buff=0.15).shift(0.15 * DOWN))
        self.mark("race"); self.play(*[Create(l_) for l_ in lines], run_time=3.0, rate_func=linear)
        self.play(FadeIn(labs), run_time=0.5)
        mon = T("50%: flipping a coin", 0.3, MUTED).next_to(Pa.P(30, 0.5), DOWN, buff=0.12)
        self.play(FadeIn(mon), run_time=0.4); self.wait(1.8)
        self.clear_but(cap)

        # ---------- 6. short memory ----------
        cap = self.swap(cap, self.caption("If a short memory is enough, both do the job", "predicting the speed 3 s ahead · mean error"))
        datos = [("\"stays the same\"", 4.42, MUTED), ("RNN", 3.19, MUTED), ("LSTM", 3.08, POINT)]
        objs = VGroup()
        for r, (nm, v, col) in enumerate(datos):
            y = 1.0 - r * 1.2
            ba = Rectangle(width=1.6 * v, height=0.5, stroke_width=0, fill_color=col, fill_opacity=0.9 if r else 0.45).move_to([-2.6, y, 0], aligned_edge=LEFT)
            objs.add(VGroup(T(nm, 0.36, col, BOLD).next_to(ba, LEFT, buff=0.25).align_to([-2.8, 0, 0], RIGHT), ba, T(fmt(v, 2) + " km/h", 0.36, INK, BOLD).next_to(ba, RIGHT, buff=0.2)))
        self.mark("short"); self.play(LaggedStart(*[AnimationGroup(FadeIn(g[0]), GrowFromEdge(g[1], LEFT), FadeIn(g[2])) for g in objs], lag_ratio=0.35), run_time=2.0)
        self.wait(2.4)
        self.clear_but(cap)

        # ---------- Closing ----------
        self.mark("legend")
        self.play(FadeOut(cap), run_time=0.6)
        e1 = Text("Recurrent networks and LSTM", font=FONT, weight=BOLD, color=INK).scale(0.62)
        e2 = VGroup(T("they read the sequence step by step, with a memory", 0.42, MUTED),
                    T("the plain RNN forgets: the error dies out on the way back", 0.42, MUTED),
                    T("the LSTM keeps on a belt what its gates decide", 0.42, MUTED),
                    T("10 s gap: RNN 49%, LSTM 99%", 0.42, MUTED)).arrange(DOWN, buff=0.18)
        VGroup(e1, e2).arrange(DOWN, buff=0.45)
        self.mark("end_text"); self.play(Write(e1), run_time=1.0); self.play(FadeIn(e2, shift=0.2 * UP, lag_ratio=0.15), run_time=1.0)
        self.wait(3.3)
        self.mark("fade"); self.play(FadeOut(VGroup(e1, e2)), run_time=1)
        self.marks["total"] = [round(self.renderer.time, 3)]
        json.dump(self.marks, open("marks.json", "w"), indent=1)
