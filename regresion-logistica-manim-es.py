import json
import numpy as np
from manim import *

# Paleta oscura de la serie ML visual
BG = "#0E131B"; INK = "#E4EAF3"; MUTED = "#9BA7B9"; LINE = "#2B3545"
MEAN = "#6EA6FF"; SAMPLE = "#AE9FF3"; POINT = "#FF9A55"; GOOD = "#4CC38A"; WARN = "#F08A5D"
FONT = "Inter"
config.background_color = BG

# Horas de estudio y resultado (1 = aprobó)
EXAM = [(0.5, 0), (1.1, 0), (1.6, 0), (2.0, 0), (2.6, 0), (3.0, 0), (3.5, 0), (4.2, 0), (4.8, 0), (5.6, 0), (6.4, 0),
        (3.2, 1), (4.5, 1), (5.1, 1), (5.9, 1), (6.1, 1), (6.8, 1), (7.3, 1), (7.9, 1), (8.4, 1), (9.0, 1), (9.6, 1)]
H = np.array([e[0] for e in EXAM]); Y = np.array([e[1] for e in EXAM], float)
sig = lambda z: 1 / (1 + np.exp(-z))


def fit_logistic(x, y):
    w = np.zeros(2); X = np.c_[np.ones_like(x), x]
    for _ in range(50):
        mu = sig(X @ w); g = X.T @ (mu - y); Hs = X.T @ (X * (mu * (1 - mu))[:, None])
        w -= np.linalg.solve(Hs, g)
    return w


W = fit_logistic(H, Y)
K_BEST, M_BEST = W[1], -W[0] / W[1]
LIN = np.polyfit(H, Y, 1)
XS = np.linspace(0.05, 9.95, 300)


def nll(k, m):
    p = np.clip(sig(k * (H - m)), 1e-12, 1 - 1e-12)
    return float(-np.sum(Y * np.log(p) + (1 - Y) * np.log(1 - p)))


class Logistica(Scene):
    def caption(self, text, sub=None):
        t = Text(text, font=FONT, weight=SEMIBOLD, color=INK).scale(0.62)
        grp = VGroup(t)
        if sub:
            grp.add(Text(sub, font=FONT, color=MUTED).scale(0.42)); grp.arrange(DOWN, buff=0.18)
        return grp.to_edge(UP, buff=0.4)

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
        title = Text("Regresión logística", font=FONT, weight=BOLD, color=INK).scale(1.1)
        sub = Text("de datos a probabilidades… y de probabilidades a decisiones", font=FONT, color=MUTED).scale(0.48)
        VGroup(title, sub).arrange(DOWN, buff=0.35)
        self.mark("title"); self.play(Write(title), run_time=1.4)
        self.mark("subtitle"); self.play(FadeIn(sub, shift=0.2 * UP)); self.wait(1.2)
        self.mark("title_out"); self.play(FadeOut(title), FadeOut(sub))

        ax = Axes(x_range=[0, 10, 1], y_range=[-0.35, 1.35, 0.5], x_length=12, y_length=5.0,
                  axis_config={"color": LINE, "stroke_width": 2, "include_ticks": False}, tips=False).shift(0.6 * DOWN)
        g100 = DashedLine(ax.c2p(0, 1), ax.c2p(10, 1), color=LINE, dash_length=0.08)
        g0 = DashedLine(ax.c2p(0, 0), ax.c2p(10, 0), color=LINE, dash_length=0.08)
        l100 = Text("100 %", font=FONT, color=MUTED).scale(0.32).next_to(ax.c2p(0, 1), LEFT, buff=0.12)
        l0 = Text("0 %", font=FONT, color=MUTED).scale(0.32).next_to(ax.c2p(0, 0), LEFT, buff=0.12)
        self.mark("axes"); self.play(Create(ax), FadeIn(g100), FadeIn(g0), FadeIn(l100), FadeIn(l0), run_time=1)

        def dot(x, y):
            d = Dot(ax.c2p(x, y), radius=0.1, color=POINT if y else SAMPLE)
            d.set_stroke(BG, width=4, background=True); return d

        def curve(f, color=MEAN, width=5, lo=-0.35, hi=1.35):
            ys = np.clip(f(XS), lo, hi)
            m = VMobject(stroke_color=color, stroke_width=width); m.set_points_smoothly([ax.c2p(x, y) for x, y in zip(XS, ys)])
            return m

        cap = self.swap(None, self.caption("Preguntas de sí o no", "cada punto es un alumno: aprobó (arriba) o no (abajo)"))
        dots = VGroup(*[dot(x, y) for x, y in sorted(EXAM)])
        self.mark("points"); self.play(LaggedStart(*[GrowFromCenter(d) for d in dots], lag_ratio=0.12), run_time=2.0)
        self.wait(0.6)

        cap = self.swap(cap, self.caption("Una recta no sirve", "predice menos de 0 % y más de 100 %"))
        line = curve(lambda x: LIN[0] * x + LIN[1])
        self.mark("line"); self.play(Create(line), run_time=1.2)
        bad_top = Polygon(ax.c2p(0, 1), ax.c2p(10, 1), ax.c2p(10, 1.35), ax.c2p(0, 1.35), stroke_width=0, fill_color=WARN, fill_opacity=0.15)
        bad_bot = Polygon(ax.c2p(0, 0), ax.c2p(10, 0), ax.c2p(10, -0.35), ax.c2p(0, -0.35), stroke_width=0, fill_color=WARN, fill_opacity=0.15)
        self.mark("bad"); self.play(FadeIn(bad_top), FadeIn(bad_bot), run_time=0.8); self.bring_to_front(dots); self.wait(1.0)

        cap = self.swap(cap, self.caption("La curva S: siempre entre 0 y 1", "se llama sigmoide"))
        k, m = ValueTracker(4.0), ValueTracker(7.6)
        s_curve = curve(lambda x: sig(k.get_value() * (x - m.get_value())))
        self.mark("morph"); self.play(Transform(line, s_curve), FadeOut(bad_top), FadeOut(bad_bot), run_time=1.4)
        self.remove(line)
        live = always_redraw(lambda: curve(lambda x: sig(k.get_value() * (x - m.get_value()))))
        self.add(live); self.bring_to_front(dots); self.wait(0.6)

        cap = self.swap(cap, self.caption("La mejor curva: la que menos se sorprende", "líneas largas = el modelo estaba seguro y se equivocó"))
        res = always_redraw(lambda: VGroup(*[Line(ax.c2p(x, y), ax.c2p(x, sig(k.get_value() * (x - m.get_value()))),
                                                  color=MUTED, stroke_width=3, stroke_opacity=0.35 + 0.65 * abs(y - sig(k.get_value() * (x - m.get_value())))) for x, y in EXAM]))
        lab = Text("sorpresa total", font=FONT, color=MUTED).scale(0.4).move_to(ax.c2p(1.3, 1.22))
        val = always_redraw(lambda: Text(f"{nll(k.get_value(), m.get_value()):.1f}", font=FONT, weight=SEMIBOLD, color=INK).scale(0.62).next_to(lab, DOWN, buff=0.1))
        self.mark("surprise"); self.play(FadeIn(res), FadeIn(lab), FadeIn(val), run_time=0.8); self.bring_to_front(dots)
        self.mark("settle"); self.play(k.animate.set_value(K_BEST), m.animate.set_value(M_BEST), run_time=2.2, rate_func=smooth)
        self.wait(1.0)

        cap = self.swap(cap, self.caption("Umbral: de probabilidad a decisión", "por encima del umbral, se actúa"))
        for mm in (res, val):
            mm.clear_updaters()
        self.play(FadeOut(res), FadeOut(lab), FadeOut(val), run_time=0.5)
        t = ValueTracker(0.5)
        x_at = lambda: M_BEST + np.log(t.get_value() / (1 - t.get_value())) / K_BEST
        hline = always_redraw(lambda: DashedLine(ax.c2p(0, t.get_value()), ax.c2p(10, t.get_value()), color=INK, stroke_width=3))
        zone = always_redraw(lambda: Polygon(ax.c2p(x_at(), -0.35), ax.c2p(10, -0.35), ax.c2p(10, 1.35), ax.c2p(x_at(), 1.35),
                                             stroke_width=0, fill_color=POINT, fill_opacity=0.12))
        tlab = always_redraw(lambda: Text(f"umbral {round(t.get_value() * 100)} %", font=FONT, color=INK).scale(0.38)
                             .next_to(ax.c2p(10, t.get_value()), UP, buff=0.08).shift(LEFT * 0.9))
        self.mark("threshold"); self.play(Create(hline), FadeIn(zone), FadeIn(tlab), run_time=1.0); self.bring_to_front(live, dots)
        self.wait(1.0)
        cap = self.swap(cap, self.caption("Si equivocarse sale caro, baja el umbral", "mejor una revisión de más que una avería en ruta"))
        self.mark("lower"); self.play(t.animate.set_value(0.15), run_time=1.6, rate_func=smooth)
        self.wait(1.2)

        # ---------- 2D ----------
        self.mark("clear")
        for mm in (live, hline, zone, tlab):
            mm.clear_updaters()
        self.play(*[FadeOut(mm) for mm in [ax, g100, g0, l100, l0, live, dots, hline, zone, tlab]], run_time=0.8)
        cap = self.swap(cap, self.caption("Con dos variables: una frontera", "a un lado, probablemente sí; al otro, probablemente no"))
        rng = np.random.default_rng(14)
        P2 = []
        for _ in range(34):
            x, y = rng.uniform(-5.4, 5.4), rng.uniform(-2.7, 2.2)
            s = 0.9 * x + 1.3 * y + rng.normal(0, 1.4)
            P2.append((x, y - 0.5, int(s > 0)))
        d2 = VGroup(*[Dot([x, y, 0], radius=0.1, color=POINT if c else SAMPLE).set_stroke(BG, 4, background=True) for x, y, c in P2])
        self.mark("points2"); self.play(LaggedStart(*[GrowFromCenter(d) for d in d2], lag_ratio=0.05), run_time=1.6)
        ang = ValueTracker(-1.2)
        target = np.arctan2(-0.9, 1.3)  # dirección de la frontera 0.9x + 1.3y = 0
        XL, XR, YB, YT = -6.4, 6.4, -3.7, 2.3
        region = Rectangle(width=XR - XL, height=YT - YB).move_to([(XL + XR) / 2, (YB + YT) / 2, 0])
        c0 = np.array([0, -0.5, 0])
        def clip_t(v):
            lo, hi = -1e9, 1e9
            for comp, a, b_ in ((0, XL, XR), (1, YB, YT)):
                if abs(v[comp]) < 1e-9:
                    continue
                t1, t2 = (a - c0[comp]) / v[comp], (b_ - c0[comp]) / v[comp]
                lo, hi = max(lo, min(t1, t2)), min(hi, max(t1, t2))
            return lo, hi
        def bnd():
            a = ang.get_value(); v = np.array([np.cos(a), np.sin(a), 0]); lo, hi = clip_t(v)
            return DashedLine(c0 + v * lo, c0 + v * hi, color=INK, stroke_width=3.5)
        def half():
            a = ang.get_value(); v = np.array([np.cos(a), np.sin(a), 0]); n = np.array([-v[1], v[0], 0])
            big = Polygon(c0 - v * 12, c0 + v * 12, c0 + v * 12 + n * 12, c0 - v * 12 + n * 12)
            return Intersection(region, big, stroke_width=0, fill_color=POINT, fill_opacity=0.12)
        b = always_redraw(bnd); hz = always_redraw(half)
        self.mark("sweep"); self.play(FadeIn(b), FadeIn(hz), run_time=0.6); self.bring_to_front(d2)
        self.play(ang.animate.set_value(target), run_time=1.6, rate_func=smooth)
        self.wait(1.0)

        b.clear_updaters(); hz.clear_updaters()
        self.mark("legend")
        self.play(FadeOut(cap), run_time=0.5)
        k1 = VGroup(Dot(radius=0.1, color=POINT), Text("sí", font=FONT, color=INK).scale(0.42)).arrange(RIGHT, buff=0.2)
        k2 = VGroup(Dot(radius=0.1, color=SAMPLE), Text("no", font=FONT, color=INK).scale(0.42)).arrange(RIGHT, buff=0.2)
        k3 = VGroup(DashedLine(LEFT * 0.3, RIGHT * 0.3, color=INK, stroke_width=4), Text("frontera del 50 %", font=FONT, color=INK).scale(0.42)).arrange(RIGHT, buff=0.2)
        legend = VGroup(k1, k2, k3).arrange(RIGHT, buff=0.7).to_edge(UP, buff=0.5)
        end = Text("Datos → probabilidades → decisiones.", font=FONT, weight=SEMIBOLD, color=INK).scale(0.55).to_edge(DOWN, buff=0.35)
        bgend = BackgroundRectangle(end, color=BG, fill_opacity=0.85, buff=0.15)
        self.play(FadeIn(legend, shift=0.2 * DOWN), run_time=0.8)
        self.mark("end_text"); self.play(FadeIn(bgend), Write(end), run_time=1.5)
        self.wait(2.3)
        self.mark("fade"); self.play(*[FadeOut(mm) for mm in self.mobjects], run_time=1)
        self.marks["total"] = [round(self.renderer.time, 3)]
        json.dump(self.marks, open("marks.json", "w"), indent=1)
