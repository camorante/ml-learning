import json
import sys
import numpy as np
from manim import *

sys.path.insert(0, "../../rn/r6-en/py")
from carrera import correr, valle  # noqa: E402  (the same computation as the pages)

# Dark palette of the series
BG = "#0E131B"; INK = "#E4EAF3"; MUTED = "#9BA7B9"; LINE = "#2B3545"
MEAN = "#6EA6FF"; SAMPLE = "#AE9FF3"; POINT = "#FF9A55"; GOOD = "#4CC38A"; WARN = "#F08A5D"
FONT = "Inter"
config.background_color = BG
D = json.load(open("../../rn/r6-en/embed.json", encoding="utf-8"))


def T(s, sc=0.45, col=INK, w=None):
    return Text(s, font=FONT, color=col, weight=w or NORMAL).scale(sc)


def fmt(v, d=1):
    s = f"{abs(v):,.{d}f}"
    return ("−" if v < 0 and round(abs(v), d) != 0 else "") + s


class Panel:
    """Simple 2D axes: maps data (x, y) to the scene inside a box."""

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

    def poly(self, pts, col, w=4):
        q = [self.P(min(self.x1, max(self.x0, x)), min(self.y1, max(self.y0, y))) for x, y in pts]
        return VMobject(stroke_color=col, stroke_width=w).set_points_as_corners(q)


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

    def valle_panel(self, k):
        Pv = Panel(-5, 5, -1.9, 1.9, (-6.2, 6.2, -3.0, 1.7), xt=(-4, 0, 4), xf=lambda v: fmt(v, 0), xl="weight a", yl="weight b")
        el = VGroup()
        for c in (0.05, 0.3, 1, 2.5, 5, 8, 12.5):
            t = np.linspace(0, 2 * np.pi, 120)
            el.add(Pv.poly(list(zip(np.sqrt(2 * c) * np.cos(t), np.sqrt(2 * c / k) * np.sin(t))), MEAN, 2).set_stroke(opacity=0.45))
        return Pv, el

    def construct(self):
        self.marks = {}
        title = Text("Training well", font=FONT, weight=BOLD, color=INK).scale(0.95)
        sub = Text("batches, learning rate, momentum and Adam", font=FONT, color=MUTED).scale(0.46)
        VGroup(title, sub).arrange(DOWN, buff=0.35)
        self.mark("title"); self.play(Write(title), run_time=1.4)
        self.mark("subtitle"); self.play(FadeIn(sub, shift=0.2 * UP)); self.wait(1.4)
        self.mark("title_out"); self.play(FadeOut(title), FadeOut(sub))

        # ---------- 1. the loop ----------
        cap = self.swap(None, self.caption("The training loop", "600 engine readings, batches of 32: 19 steps per epoch"))
        names = ["batch", "forward", "loss", "backward", "update"]
        ang = [90 - 72 * i for i in range(5)]
        pos = [np.array([2.5 * np.cos(np.radians(a)), 2.3 * np.sin(np.radians(a)) - 0.75, 0]) for a in ang]
        boxes = VGroup(*[RoundedRectangle(width=2.3, height=0.75, corner_radius=0.18, stroke_color=MUTED, stroke_width=2.5).move_to(p) for p in pos])
        labs = VGroup(*[T(f"{i + 1} · {n}", 0.36, INK, BOLD).move_to(p) for i, (n, p) in enumerate(zip(names, pos))])
        arrs = VGroup()
        for i in range(5):
            a, b = pos[i], pos[(i + 1) % 5]
            d = (b - a) / np.linalg.norm(b - a)
            arrs.add(Arrow(a + d * 1.0, b - d * 1.0, buff=0, color=MUTED, stroke_width=3, max_tip_length_to_length_ratio=0.3))
        self.play(Create(boxes), FadeIn(labs), run_time=1.0)
        self.play(Create(arrs), run_time=0.8)
        cnt = T("epoch 1 · step 1", 0.4, POINT, BOLD).move_to([0, -0.75, 0])
        self.play(FadeIn(cnt), run_time=0.3)
        Lt = D["bucle"]["Lt"]
        for paso in range(2):
            for i in range(5):
                self.mark("stage")
                hl = boxes[i].copy().set_fill(POINT, 0.85).set_stroke(POINT)
                self.play(FadeIn(hl), run_time=0.22)
                self.play(FadeOut(hl), run_time=0.18)
            nc = T(f"epoch 1 · step {paso + 2}", 0.4, POINT, BOLD).move_to(cnt)
            self.play(Transform(cnt, nc), run_time=0.25)
        nc = VGroup(T("…and after all 600, shuffle and repeat", 0.36, MUTED),
                    T(f"after 2 epochs, the loss goes down from {fmt(D['toy']['adam|0.1|32']['tr'][0], 3)} to {fmt(Lt[-1], 3)}", 0.36, GOOD, BOLD)).arrange(DOWN, buff=0.12).move_to([0, -3.55, 0])
        self.mark("shuffle"); self.play(FadeIn(nc, shift=0.1 * UP), run_time=0.7); self.wait(2.0)
        self.clear_but(cap)

        # ---------- 2. the batch ----------
        cap = self.swap(cap, self.caption("The batch: a poll instead of a census", "2,000 trucks from topic 4 · one neuron with weight w and bias b"))
        L = D["lotes"]
        Pb = Panel(-8, 8, -6, 3, (-6.0, 6.0, -3.0, 1.7), xt=(-8, -4, 0, 4, 8), xf=lambda v: fmt(v, 0), xl="weight w", yt=(-6, -3, 0, 3), yf=lambda v: fmt(v, 0), yl="bias b")
        self.play(FadeIn(Pb.g), run_time=0.6)
        st = Dot(Pb.P(-6, -2.5), radius=0.1, color=INK); fo = Dot(Pb.P(*L["fondo"]), radius=0.12, color=GOOD)
        self.play(FadeIn(st), FadeIn(fo), run_time=0.4)
        c1 = Pb.poly(L["lotes"]["1"]["camino"], WARN, 2.5); c2 = Pb.poly(L["lotes"]["2000"]["camino"], MEAN, 4)
        t1 = T(f"batch of 1: it shakes (standard error {fmt(L['lotes']['1']['ruido'], 2)})", 0.34, WARN, BOLD).move_to([-1.0, -3.6, 0])
        t2 = T(f"batch of 2,000: smooth, but one step per epoch", 0.34, MEAN, BOLD).move_to([-1.0, -3.6, 0])
        self.mark("noisy"); self.play(Create(c1), FadeIn(t1), run_time=2.4); self.wait(0.6)
        self.mark("smooth"); self.play(Create(c2), FadeOut(t1), FadeIn(t2), run_time=1.8); self.wait(0.6)
        t3 = T("4 times the trucks, half the error: medium batches", 0.36, GOOD, BOLD).move_to([0, -3.6, 0])
        self.play(FadeOut(t2), FadeIn(t3), run_time=0.6); self.wait(1.8)
        self.clear_but(cap)

        # ---------- 3. the rate ----------
        cap = self.swap(cap, self.caption("The learning rate: how long each step is", "validation loss of the real network, 60 epochs"))
        Pt = Panel(0, 60, 0, 1, (-5.8, 5.8, -2.8, 1.6), xt=(0, 20, 40, 60), xf=lambda v: fmt(v, 0), xl="epoch", yt=(0, 0.5, 1), yf=lambda v: fmt(v, 1), yl="loss")
        self.play(FadeIn(Pt.g), run_time=0.6)
        for key, col, txt, y in (("0.001", MUTED, "0.001: never gets there", 0.85), ("0.1", GOOD, "0.1: the good one", 0.4), ("10", WARN, "10: blows up", 0.95)):
            v = D["tasa"][key]
            ln = Pt.poly([(i, min(x, 1.0)) for i, x in enumerate(v)], col, 4)
            lb = T(txt, 0.34, col, BOLD).move_to(Pt.P(46 if key != "10" else 30, y if key != "10" else 0.86) + 0.3 * UP)
            self.mark("lr"); self.play(Create(ln), FadeIn(lb), run_time=1.4); self.wait(0.8)
        self.wait(1.2)
        self.clear_but(cap)

        # ---------- 4. momentum ----------
        cap = self.swap(cap, self.caption("Momentum: a rolling ball", "narrow valley, same rate 0.019"))
        Pv, el = self.valle_panel(100)
        self.play(FadeIn(Pv.g), Create(el), run_time=0.8)
        n0, c0 = correr(valle(100), "sgd", 0.019, pasos=2000); n1, cm = correr(valle(100), "momento", 0.019, 0.8, pasos=2000)
        s1 = Pv.poly(c0[:120], MUTED, 2.5); s2 = Pv.poly(cm[:120], POINT, 4)
        l1 = T(f"no momentum: {n0} steps", 0.36, MUTED, BOLD).move_to([-3.3, -3.65, 0])
        l2 = T(f"momentum β = 0.8: {n1} steps", 0.36, POINT, BOLD).move_to([2.9, -3.65, 0])
        self.mark("zig"); self.play(Create(s1), FadeIn(l1), run_time=2.0); self.wait(0.4)
        self.mark("ball"); self.play(Create(s2), FadeIn(l2), run_time=2.0); self.wait(2.0)
        self.clear_but(cap)

        # ---------- 5. Adam ----------
        cap = self.swap(cap, self.caption("Adam: a step tailored to each weight", "b is 1,600 times steeper than a"))
        Pv, el = self.valle_panel(1600)
        self.play(FadeIn(Pv.g), Create(el), run_time=0.8)
        ns, cs = correr(valle(1600), "sgd", 1.5 / 1600, pasos=10000); na, ca = correr(valle(1600), "adam", 0.3, pasos=10000)
        s1 = Pv.poly(cs[:150], MUTED, 2.5); s2 = Pv.poly(ca[:150], POINT, 4)
        l1 = T(f"plain descent: {ns:,} steps", 0.36, MUTED, BOLD).move_to([-3.1, -3.65, 0])
        l2 = T(f"Adam: {na} steps", 0.36, POINT, BOLD).move_to([3.0, -3.65, 0])
        self.mark("sgd"); self.play(Create(s1), FadeIn(l1), run_time=1.8); self.wait(0.3)
        self.mark("adam"); self.play(Create(s2), FadeIn(l2), run_time=1.8)
        n3 = T("the same 44 steps in any valley of this kind", 0.34, GOOD, BOLD).move_to([0, 1.25, 0])
        self.play(FadeIn(n3), run_time=0.5); self.wait(2.0)
        self.clear_but(cap)

        # ---------- 6. reading the curve ----------
        cap = self.swap(cap, self.caption("Reading the curve", "a large network with only 40 readings"))
        S = D["curvas"]["sobre"]
        Pc = Panel(0, 1500, 0, 5, (-5.8, 5.8, -2.8, 1.6), xt=(0, 500, 1000, 1500), xf=lambda v: fmt(v, 0), xl="epoch", yt=(0, 1, 2, 3, 4, 5), yf=lambda v: fmt(v, 0), yl="loss")
        self.play(FadeIn(Pc.g), run_time=0.6)
        tr = Pc.poly([(5 * i, v) for i, v in enumerate(S["tr"])], MEAN, 4); va = Pc.poly([(5 * i, v) for i, v in enumerate(S["va"])], POINT, 4)
        lt = T("training: almost 0", 0.34, MEAN, BOLD).move_to(Pc.P(1150, 0.6)); lv = T("validation: goes up", 0.34, POINT, BOLD).move_to(Pc.P(1150, 4.75))
        self.mark("train"); self.play(Create(tr), FadeIn(lt), run_time=1.5)
        self.mark("val"); self.play(Create(va), FadeIn(lv), run_time=1.5)
        vl = DashedLine(Pc.P(S["mejor"], 0), Pc.P(S["mejor"], 5), color=GOOD, stroke_width=2.5)
        lm = T(f"best at epoch {S['mejor']}: it memorizes (topic 7)", 0.34, GOOD, BOLD).move_to(Pc.P(820, 1.6))
        self.play(Create(vl), FadeIn(lm), run_time=0.7); self.wait(2.2)
        self.clear_but(cap)

        # ---------- Closing ----------
        self.mark("legend")
        self.play(FadeOut(cap), run_time=0.6)
        e1 = Text("Training well", font=FONT, weight=BOLD, color=INK).scale(0.62)
        e2 = VGroup(T("scale the inputs and set validation aside", 0.42, MUTED),
                    T("medium batches: cheap and not too shaky", 0.42, MUTED),
                    T("try rates in steps of 3; Adam as a starting point", 0.42, MUTED),
                    T("watch both curves every epoch", 0.42, MUTED)).arrange(DOWN, buff=0.18)
        VGroup(e1, e2).arrange(DOWN, buff=0.45)
        self.mark("end_text"); self.play(Write(e1), run_time=1.0); self.play(FadeIn(e2, shift=0.2 * UP, lag_ratio=0.15), run_time=1.0)
        self.wait(3.3)
        self.mark("fade"); self.play(FadeOut(VGroup(e1, e2)), run_time=1)
        self.marks["total"] = [round(self.renderer.time, 3)]
        json.dump(self.marks, open("marks.json", "w"), indent=1)
