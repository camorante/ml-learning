import json
import numpy as np
from manim import *

# Dark palette of the series
BG = "#0E131B"; INK = "#E4EAF3"; MUTED = "#9BA7B9"; LINE = "#2B3545"
MEAN = "#6EA6FF"; SAMPLE = "#AE9FF3"; POINT = "#FF9A55"; GOOD = "#4CC38A"; WARN = "#F08A5D"
FONT = "Inter"
config.background_color = BG
D = json.load(open("../../rn/r12-en/embed.json", encoding="utf-8"))   # the same data as the pages
TC = [MUTED, SAMPLE, POINT, GOOD, MEAN]                         # normal, stuck, spike, drift, vibration
TN = ["normal", "stuck sensor", "spike", "drift", "vibration"]
E_TE, Y_TE, E_VA = np.array(D["e_te"]), np.array(D["y_te"]), np.array(D["e_va"])


def T(s, sc=0.45, col=INK, w=None):
    return Text(s, font=FONT, color=col, weight=w or NORMAL).scale(sc)


def fmt(v, d=1):
    s = f"{abs(v):.{d}f}"
    if d == 0 and len(s) > 3:
        s = f"{int(s):,}"
    return ("−" if v < 0 and round(abs(v), d) != 0 else "") + s


def cuantil(a, q):
    s = np.sort(a); pos = (len(s) - 1) * q; i = int(np.floor(pos)); fr = pos - i
    return float(s[i] + (s[min(i + 1, len(s) - 1)] - s[i]) * fr)


def cuenta(u):
    al = E_TE > u
    fp, vp = int((al & (Y_TE == 0)).sum()), int((al & (Y_TE > 0)).sum())
    return fp, vp, 200 - vp


def separa(labs, gap=0.06):
    """Push apart overlapping labels vertically, keeping their x position."""
    labs = sorted(labs, key=lambda m: -m.get_center()[1])
    for a, b in zip(labs, labs[1:]):
        if abs(a.get_center()[0] - b.get_center()[0]) < (a.width + b.width) / 2:
            need = a.get_bottom()[1] - gap - b.get_top()[1]
            if need < 0:
                b.shift([0, need, 0])
    return labs


def traza(v, x0, x1, y0, y1, R, col, w=4):
    """64 readings drawn in the rectangle [x0, x1] × [y0, y1], with ±R at the edges."""
    pts = [[x0 + (x1 - x0) * i / 63, (y0 + y1) / 2 + (y1 - y0) / 2 * np.clip(v[i], -R, R) / R, 0] for i in range(64)]
    return VMobject(stroke_color=col, stroke_width=w).set_points_as_corners(pts)


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
        title = Text("Autoencoders", font=FONT, weight=BOLD, color=INK).scale(1.05)
        sub = Text("compress, rebuild and catch the odd ones", font=FONT, color=MUTED).scale(0.46)
        VGroup(title, sub).arrange(DOWN, buff=0.35)
        self.mark("title"); self.play(Write(title), run_time=1.4)
        self.mark("subtitle"); self.play(FadeIn(sub, shift=0.2 * UP)); self.wait(1.4)
        self.mark("title_out"); self.play(FadeOut(title), FadeOut(sub))

        # ---------- 1. the bottleneck ----------
        cap = self.swap(None, self.caption("A network that copies through a bottleneck", "64 accelerometer readings → 4 numbers → 64 readings"))
        xs, hs = [-4.6, -2.3, 0, 2.3, 4.6], [2.6, 1.6, 0.45, 1.6, 2.6]
        cy, bw = -0.35, 0.7
        fun = VGroup(*[Polygon([xs[i] + bw / 2, cy + hs[i] / 2, 0], [xs[i + 1] - bw / 2, cy + hs[i + 1] / 2, 0],
                               [xs[i + 1] - bw / 2, cy - hs[i + 1] / 2, 0], [xs[i] + bw / 2, cy - hs[i] / 2, 0],
                               stroke_width=0, fill_color=MEAN, fill_opacity=0.16) for i in range(4)])
        cajas = VGroup(*[RoundedRectangle(width=bw, height=hs[i], corner_radius=0.1, stroke_color=POINT if i == 2 else LINE, stroke_width=2.5,
                                          fill_color=POINT if i == 2 else BG, fill_opacity=0.9 if i == 2 else 1).move_to([xs[i], cy, 0]) for i in range(5)])
        nums = VGroup(*[T(n, 0.36, INK, BOLD).next_to(cajas[i], DOWN, buff=0.18) for i, n in enumerate(["64", "32", "4", "32", "64"])])
        tags = VGroup(T("encoder", 0.36, MUTED).move_to([-2.3, cy + 1.65, 0]), T("code", 0.36, POINT, BOLD).move_to([0, cy + 0.65, 0]),
                      T("decoder", 0.36, MUTED).move_to([2.3, cy + 1.65, 0]))
        self.mark("funnel"); self.play(FadeIn(fun), LaggedStart(*[FadeIn(c) for c in cajas], lag_ratio=0.15), FadeIn(nums), FadeIn(tags), run_time=1.6)
        w = D["win"][str(D["ej"]["0"][1])]
        tin = traza(w["x"], -6.6, -5.1, cy - 0.9, cy + 0.9, 2, INK, 3)
        tout = traza(w["r"], 5.1, 6.6, cy - 0.9, cy + 0.9, 2, MEAN, 3)
        self.mark("flow"); self.play(Create(tin), run_time=0.8)
        self.play(Indicate(cajas[2], color=POINT, scale_factor=1.25), run_time=0.8)
        self.play(Create(tout), run_time=0.8)
        nt = VGroup(T("no labels: the right answer is the input itself", 0.36, INK, BOLD),
                    T("whatever does not fit in 4 numbers is lost: the network keeps the essentials of normal", 0.32, GOOD)).arrange(DOWN, buff=0.2).move_to([0, -2.9, 0])
        self.play(FadeIn(nt), run_time=0.6); self.wait(3.4)
        self.clear_but(cap)

        # ---------- 2. normal is copied well; faults are not ----------
        cap = self.swap(cap, self.caption("Normal is copied well; odd is not", "trained on 3,000 normal windows only"))
        X0, X1, YC = -5.6, 3.0, -0.2
        u99 = cuantil(E_VA, 0.99)
        marco = Rectangle(width=X1 - X0, height=3.6, stroke_color=LINE, stroke_width=2).move_to([(X0 + X1) / 2, YC, 0])
        cero = Line([X0, YC, 0], [X1, YC, 0], stroke_color=LINE, stroke_width=1.5)
        ley = VGroup(VGroup(Line(ORIGIN, RIGHT * 0.4, stroke_color=INK, stroke_width=4), T("input", 0.3, INK)).arrange(RIGHT, buff=0.12),
                     VGroup(Line(ORIGIN, RIGHT * 0.4, stroke_color=MEAN, stroke_width=4), T("copy", 0.3, MEAN)).arrange(RIGHT, buff=0.12),
                     VGroup(Square(0.18, fill_color=POINT, fill_opacity=1, stroke_width=0), T("difference", 0.3, POINT)).arrange(RIGHT, buff=0.12)).arrange(RIGHT, buff=0.35).move_to([(X0 + X1) / 2, YC - 2.15, 0])
        self.play(Create(marco), FadeIn(cero), FadeIn(ley), run_time=0.6)
        prev = None
        for k, wi in ((0, 1), (2, 1), (3, 1), (1, 2), (4, 2)):
            w = D["win"][str(D["ej"][str(k)][wi])]
            R = max(2.0, float(np.ceil(max(np.abs(w["x"]).max(), np.abs(w["r"]).max()) + 0.25)))
            dif = VGroup(*[Line([X0 + (X1 - X0) * i / 63, YC, 0], [X0 + (X1 - X0) * i / 63, YC + 1.7 * np.clip(w["x"][i] - w["r"][i], -R, R) / R, 0],
                                stroke_color=POINT, stroke_width=4, stroke_opacity=0.75) for i in range(64)])
            gin, gout = traza(w["x"], X0, X1, YC - 1.7, YC + 1.7, R, INK, 3.5), traza(w["r"], X0, X1, YC - 1.7, YC + 1.7, R, MEAN, 4.5)
            al = w["e"] > u99
            info = VGroup(T(TN[k], 0.5, TC[k] if k else INK, BOLD), T("reconstruction error", 0.3, MUTED), T(fmt(w["e"], 4), 0.62, WARN if al else GOOD, BOLD),
                          T("alarm!" if al else "below the threshold", 0.36, WARN if al else GOOD, BOLD), T(f"threshold {fmt(u99, 3)}", 0.28, MUTED)).arrange(DOWN, aligned_edge=LEFT, buff=0.2).move_to([3.5, YC, 0], aligned_edge=LEFT)
            g = VGroup(dif, gin, gout, info)
            self.mark("normal" if k == 0 else "fault")
            if prev is None:
                self.play(Create(gin), run_time=0.7); self.play(Create(gout), FadeIn(dif), FadeIn(info), run_time=0.9)
            else:
                self.play(FadeOut(prev), run_time=0.3); self.play(Create(gin), Create(gout), FadeIn(dif), FadeIn(info), run_time=0.9)
            prev = g; self.wait(2.4 if k else 2.6)
        self.clear_but(cap)

        # ---------- 3. the histogram ----------
        cap = self.swap(cap, self.caption("Normal on one side, faults on the other", "error of the 2,200 test windows (log scale)"))
        L, R_, B, Tp = -5.4, 5.4, -2.6, 1.5
        lo, hi, nb = -2, np.log10(2), 46
        bwl = (hi - lo) / nb
        X = lambda e: L + (np.log10(e) - lo) / (hi - lo) * (R_ - L)
        bins = lambda m: np.bincount(np.clip(((np.log10(E_TE[m]) - lo) / bwl).astype(int), 0, nb - 1), minlength=nb)
        hN, hA = bins(Y_TE == 0), [bins(Y_TE == k) for k in range(1, 5)]
        mN, mA = hN.max(), np.sum(hA, 0).max()
        ww = (R_ - L) / nb
        eje = VGroup(Line([L, B, 0], [R_, B, 0], stroke_color=MUTED, stroke_width=2),
                     *[T(s, 0.3, MUTED).move_to([X(v), B - 0.3, 0]) for v, s in ((0.01, "0.01"), (0.1, "0.1"), (1, "1"))])
        barN = VGroup(*[Rectangle(width=ww * 0.92, height=max(1e-3, n / mN * (Tp - B)), stroke_width=0, fill_color=MUTED, fill_opacity=0.5).move_to([L + (i + 0.5) * ww, B + n / mN * (Tp - B) / 2, 0]) for i, n in enumerate(hN) if n])
        self.play(FadeIn(eje), run_time=0.4)
        self.mark("hist"); self.play(LaggedStart(*[GrowFromEdge(b, DOWN) for b in barN], lag_ratio=0.05), run_time=1.4)
        tn = T("2,000 normal", 0.34, MUTED, BOLD).move_to([X(0.037) - 0.25, Tp + 0.3, 0], aligned_edge=RIGHT)
        self.play(FadeIn(tn), run_time=0.4)
        acc = np.zeros(nb); barA = VGroup()
        for k in range(4):
            for i, n in enumerate(hA[k]):
                if n:
                    h0, h1 = acc[i] / mA * (Tp - B) * 0.85, (acc[i] + n) / mA * (Tp - B) * 0.85
                    barA.add(Rectangle(width=ww * 0.6, height=h1 - h0, stroke_width=0, fill_color=TC[k + 1], fill_opacity=0.95).move_to([L + (i + 0.5) * ww, B + (h0 + h1) / 2, 0]))
            acc += hA[k]
        leyA = VGroup(*[VGroup(Square(0.16, fill_color=TC[k], fill_opacity=1, stroke_width=0), T(TN[k], 0.28, TC[k])).arrange(RIGHT, buff=0.1) for k in range(1, 5)]).arrange(RIGHT, buff=0.3).move_to([5.4, Tp + 0.3, 0], aligned_edge=RIGHT)
        self.mark("faults"); self.play(LaggedStart(*[FadeIn(b, shift=0.1 * UP) for b in barA], lag_ratio=0.01), FadeIn(leyA), run_time=1.6)
        xu = X(u99)
        lin = DashedLine([xu, B, 0], [xu, Tp, 0], color=WARN, stroke_width=4)
        fp, vp, fn = cuenta(u99)
        tu = T(f"threshold {fmt(u99, 3)}", 0.32, WARN, BOLD).move_to([xu + 0.1, Tp + 0.3, 0], aligned_edge=LEFT)
        self.mark("thr"); self.play(Create(lin), FadeIn(tu), run_time=0.8)
        res = VGroup(T(f"{vp} of 200 faults caught", 0.4, INK, BOLD), T(f"{fp} false alerts out of 2,000 normal: the threshold is their 99th percentile", 0.32, MUTED)).arrange(DOWN, buff=0.18).move_to([0, -3.4, 0])
        self.play(FadeIn(res), run_time=0.6); self.wait(3.4)
        self.clear_but(cap)

        # ---------- 4. the cost ----------
        cap = self.swap(cap, self.caption("The dollars decide the threshold", "false alert: $80 · fault that slips through: $1,500"))
        G = D["grid"]; C = [cuenta(u) for u in G]; tot = [c[0] * 80 + c[2] * 1500 for c in C]
        L, R_, B, Tp, ym = -5.0, 4.4, -2.7, 1.6, 80000
        XU = lambda u: L + (u - G[0]) / (G[-1] - G[0]) * (R_ - L)
        YU = lambda v: B + v / ym * (Tp - B)
        ejes = VGroup(Line([L, B, 0], [R_, B, 0], stroke_color=MUTED, stroke_width=2),
                      *[T(fmt(u, 2), 0.28, MUTED).move_to([XU(u), B - 0.3, 0]) for u in (0.03, 0.04, 0.05, 0.06, 0.07, 0.08)],
                      *[VGroup(Line([L, YU(v), 0], [R_, YU(v), 0], stroke_color=LINE, stroke_width=1), T(f"${v // 1000}k", 0.26, MUTED).next_to([L, YU(v), 0], LEFT, buff=0.12)) for v in (20000, 40000, 60000, 80000)])
        curva = VMobject(stroke_color=INK, stroke_width=5).set_points_as_corners([[XU(u), YU(c), 0] for u, c in zip(G, tot)])
        fa = VMobject(stroke_color=SAMPLE, stroke_width=3).set_points_as_corners([[XU(u), YU(c[0] * 80), 0] for u, c in zip(G, C)])
        tfa = VGroup(VGroup(Line(ORIGIN, RIGHT * 0.4, stroke_color=INK, stroke_width=5), T("total cost", 0.3, INK)).arrange(RIGHT, buff=0.12),
                     VGroup(Line(ORIGIN, RIGHT * 0.4, stroke_color=SAMPLE, stroke_width=3), T("what the false alerts cost", 0.3, SAMPLE)).arrange(RIGHT, buff=0.12)).arrange(RIGHT, buff=0.45).move_to([R_, Tp + 0.3, 0], aligned_edge=RIGHT)
        self.play(FadeIn(ejes), run_time=0.5)
        self.mark("cost"); self.play(Create(fa), Create(curva), FadeIn(tfa), run_time=1.8)
        b = int(np.argmin(tot)); fp, vp, fn = cuenta(u99); c99 = fp * 80 + fn * 1500
        p99 = Dot([XU(u99), YU(c99), 0], radius=0.12, color=WARN)
        t99 = T(f"99th percentile: ${fmt(c99, 0)}", 0.32, WARN, BOLD).next_to(p99, UP + RIGHT, buff=0.12)
        self.play(FadeIn(p99), FadeIn(t99), run_time=0.6); self.wait(0.8)
        pb = Circle(radius=0.2, stroke_color=GOOD, stroke_width=5).move_to([XU(G[b]), YU(tot[b]), 0])
        tb = VGroup(T(f"threshold {fmt(G[b], 3)}: ${fmt(tot[b], 0)}", 0.36, GOOD, BOLD),
                    T(f"{C[b][0]} false alerts, only {C[b][2]} faults missed", 0.3, GOOD)).arrange(DOWN, aligned_edge=LEFT, buff=0.12).move_to([XU(0.041), YU(50000), 0], aligned_edge=LEFT)
        flb = Arrow(tb.get_left() + LEFT * 0.05 + DOWN * 0.1, pb.get_top() + UP * 0.05, color=GOOD, stroke_width=3, buff=0.05, max_tip_length_to_length_ratio=0.12)
        self.mark("best"); self.play(Create(pb), FadeIn(tb), GrowArrow(flb), run_time=0.8)
        nt = T("missing a fault costs almost 19 times more than an inspection: a lower threshold pays off", 0.32, INK, BOLD).move_to([0, -3.5, 0])
        self.play(FadeIn(nt), run_time=0.5); self.wait(3.4)
        self.clear_but(cap)

        # ---------- 5. PCA against autoencoder ----------
        cap = self.swap(cap, self.caption("Without activations it is PCA; with them, it follows curves", "error on new normal windows by the numbers in the code"))
        rs = D["tam"]["res"]; K = ["1", "2", "3", "4", "6", "8"]
        L, R_, B, Tp, ym = -4.6, 4.6, -2.5, 1.4, 0.3
        Xk = lambda i: L + (i + 0.5) * (R_ - L) / 6
        Yv = lambda v: B + v / ym * (Tp - B)
        base = VGroup(Line([L, B, 0], [R_, B, 0], stroke_color=MUTED, stroke_width=2), *[T(k, 0.34, MUTED).move_to([Xk(i), B - 0.3, 0]) for i, k in enumerate(K)],
                      DashedLine([L, Yv(0.0225), 0], [R_, Yv(0.0225), 0], color=GOOD, stroke_width=2.5),
                      T("the tremor alone: 0.0225", 0.28, GOOD).next_to([R_, Yv(0.0225), 0], UP, buff=0.08).align_to([R_, 0, 0], RIGHT))
        barsP = VGroup(*[Rectangle(width=0.42, height=Yv(rs[k]["pca"]) - B, stroke_width=0, fill_color=SAMPLE, fill_opacity=0.85).move_to([Xk(i) - 0.24, (B + Yv(rs[k]["pca"])) / 2, 0]) for i, k in enumerate(K)])
        barsA = VGroup(*[Rectangle(width=0.42, height=Yv(rs[k]["ae"]) - B, stroke_width=0, fill_color=MEAN, fill_opacity=0.95).move_to([Xk(i) + 0.24, (B + Yv(rs[k]["ae"])) / 2, 0]) for i, k in enumerate(K)])
        ley = VGroup(VGroup(Square(0.18, fill_color=SAMPLE, fill_opacity=1, stroke_width=0), T("PCA (= linear autoencoder)", 0.3, SAMPLE)).arrange(RIGHT, buff=0.12),
                     VGroup(Square(0.18, fill_color=MEAN, fill_opacity=1, stroke_width=0), T("autoencoder with tanh", 0.3, MEAN)).arrange(RIGHT, buff=0.12)).arrange(RIGHT, buff=0.5).move_to([0, Tp + 0.25, 0])
        self.play(FadeIn(base), FadeIn(ley), run_time=0.5)
        self.mark("bars"); self.play(LaggedStart(*[GrowFromEdge(b, DOWN) for b in barsP], lag_ratio=0.12), run_time=1.0)
        self.play(LaggedStart(*[GrowFromEdge(b, DOWN) for b in barsA], lag_ratio=0.12), run_time=1.0)
        nt = T(f"with 4 numbers: PCA {fmt(rs['4']['pca'], 3)} · autoencoder {fmt(rs['4']['ae'], 3)}", 0.36, INK, BOLD).move_to([0, -3.35, 0])
        self.play(FadeIn(nt), run_time=0.5); self.wait(3.2)
        self.clear_but(cap)

        # ---------- Closing ----------
        self.mark("legend")
        self.play(FadeOut(cap), run_time=0.6)
        e1 = Text("Autoencoders", font=FONT, weight=BOLD, color=INK).scale(0.7)
        e2 = VGroup(T("copy the input through a bottleneck of a few numbers", 0.42, MUTED),
                    T("trained on normal data only, with no labels", 0.42, MUTED),
                    T("whatever it copies badly is suspicious: error > threshold", 0.42, MUTED),
                    T("the threshold is tuned by what each mistake costs", 0.42, MUTED)).arrange(DOWN, buff=0.18)
        VGroup(e1, e2).arrange(DOWN, buff=0.45)
        self.mark("end_text"); self.play(Write(e1), run_time=1.0); self.play(FadeIn(e2, shift=0.2 * UP, lag_ratio=0.15), run_time=1.0)
        self.wait(3.3)
        self.mark("fade"); self.play(FadeOut(VGroup(e1, e2)), run_time=1)
        self.marks["total"] = [round(self.renderer.time, 3)]
        json.dump(self.marks, open("marks.json", "w"), indent=1)
