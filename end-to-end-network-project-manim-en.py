import json
import numpy as np
from manim import *

# Dark palette of the series
BG = "#0E131B"; INK = "#E4EAF3"; MUTED = "#9BA7B9"; LINE = "#2B3545"
MEAN = "#6EA6FF"; SAMPLE = "#AE9FF3"; POINT = "#FF9A55"; GOOD = "#4CC38A"; WARN = "#F08A5D"
FONT = "Inter"
config.background_color = BG
D = json.load(open("../../rn/r13-en/embed.json", encoding="utf-8"))   # the same data as the pages
Y = np.array(D["y"]); PR = np.array(D["p"]["red"]); N = len(Y)


def T(s, sc=0.45, col=INK, w=None):
    return Text(s, font=FONT, color=col, weight=w or NORMAL).scale(sc)


def fmt(v, d=1):
    s = f"{abs(v):,.{d}f}"
    return ("−" if v < 0 and round(abs(v), d) != 0 else "") + s


def media(nombre, q):
    """Average of the three seeds, as in the pages (q = 0 PR-AUC, 2 dollars)."""
    r = next(f for f in D["tabla"] if f["n"] == nombre)
    a = 0.0
    for m in r["m"]:
        a += m[q]
    return a / 3


def cuenta(p, u=0.1):
    vp = fp = fn = 0
    for i in range(N):
        if p[i] > u:
            if Y[i]: vp += 1
            else: fp += 1
        elif Y[i]:
            fn += 1
    return vp, fp, fn, N - vp - fp - fn


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

    def caja(self, s, col, w=3.2, h=0.9, sc=0.36):
        r = RoundedRectangle(width=w, height=h, corner_radius=0.14, stroke_color=col, stroke_width=3, fill_color=col, fill_opacity=0.12)
        return VGroup(r, T(s, sc, INK, BOLD).move_to(r))

    def construct(self):
        self.marks = {}
        title = Text("End-to-end project with networks", font=FONT, weight=BOLD, color=INK).scale(0.95)
        sub = Text("from the question to the shop, step by step", font=FONT, color=MUTED).scale(0.46)
        VGroup(title, sub).arrange(DOWN, buff=0.35)
        self.mark("title"); self.play(Write(title), run_time=1.4)
        self.mark("subtitle"); self.play(FadeIn(sub, shift=0.2 * UP)); self.wait(1.3)
        self.mark("title_out"); self.play(FadeOut(title), FadeOut(sub))

        # ---------- 1. the question and its price ----------
        cap = self.swap(None, self.caption("The question and the price of each mistake", "will this truck break down in the next 30 days?"))
        rev = VGroup(T("inspection", 0.4, MEAN, BOLD), T("$400", 0.9, MEAN, BOLD), T("mechanic and one day off the road", 0.3, MUTED)).arrange(DOWN, buff=0.2).move_to([-3.0, 0.6, 0])
        ave = VGroup(T("breakdown on the road", 0.4, POINT, BOLD), T("$4,000", 0.9, POINT, BOLD), T("tow truck, repair and lost delivery", 0.3, MUTED)).arrange(DOWN, buff=0.2).move_to([3.0, 0.6, 0])
        self.mark("costs"); self.play(FadeIn(rev, shift=0.2 * UP), run_time=0.7); self.play(FadeIn(ave, shift=0.2 * UP), run_time=0.7); self.wait(0.8)
        regla = VGroup(T("alert if  risk × $4,000  >  $400", 0.42, INK, BOLD), T("that is, if the risk is above 400 / 4,000 = 10%", 0.42, GOOD, BOLD)).arrange(DOWN, buff=0.3).move_to([0, -2.0, 0])
        self.mark("rule"); self.play(FadeIn(regla[0]), run_time=0.7); self.wait(0.6); self.play(FadeIn(regla[1]), run_time=0.7); self.wait(1.8)
        self.clear_but(cap)

        # ---------- 2. split by truck and by time ----------
        cap = self.swap(cap, self.caption("Split before you look", "by truck and by time: the exam, with new trucks and months"))
        cw, rh, x0, y0 = 0.27, 0.16, -3.9, 1.75
        cols = []
        for r in range(18):
            g = 0 if r < 11 else 1 if r < 14 else 2
            for c in range(24):
                on = (g == 0 and c < 16) or (g == 1 and 16 <= c < 20) or (g == 2 and c >= 20)
                col = [MEAN, SAMPLE, POINT][g] if on else LINE
                cols.append(Rectangle(width=cw - 0.04, height=rh - 0.035, stroke_width=0, fill_color=col, fill_opacity=0.9 if on else 0.6).move_to([x0 + c * cw, y0 - r * rh, 0]))
        rej = VGroup(*cols)
        labs = VGroup(T("learn", 0.28, MEAN, BOLD).move_to([x0 - 1.05, y0 - 5 * rh, 0]), T("choose", 0.28, SAMPLE, BOLD).move_to([x0 - 1.05, y0 - 12 * rh, 0]),
                      T("exam", 0.28, POINT, BOLD).move_to([x0 - 1.05, y0 - 15.5 * rh, 0]))
        nota = T("rows: trucks · cells: months", 0.26, MUTED).next_to(rej, DOWN, buff=0.15).align_to(rej, LEFT)
        self.mark("grid"); self.play(LaggedStart(*[FadeIn(c) for c in cols], lag_ratio=0.002), FadeIn(labs), FadeIn(nota), run_time=1.6); self.wait(0.8)
        F = D["fuga"]["random, breakdowns from the whole table"]
        pa = sum(s[0][1] for s in F) / 3; pb = sum(s[1][1] for s in F) / 3
        t1 = T("with a column computed from the whole table:", 0.3, MUTED).move_to([0, -1.75, 0])
        b1 = VGroup(T("promises", 0.32, MEAN, BOLD), T(f"${fmt(pa)}", 0.5, MEAN, BOLD)).arrange(RIGHT, buff=0.25)
        b2 = VGroup(T("exam", 0.32, POINT, BOLD), T(f"${fmt(pb)}", 0.5, POINT, BOLD)).arrange(RIGHT, buff=0.25)
        fila = VGroup(b1, T("→", 0.5, MUTED), b2).arrange(RIGHT, buff=0.5).move_to([0, -2.45, 0])
        t2 = T("per truck and month: the report lies", 0.3, WARN, BOLD).move_to([0, -3.15, 0])
        self.mark("leak"); self.play(FadeIn(t1), FadeIn(b1), run_time=0.6); self.wait(0.5); self.play(FadeIn(fila[1]), FadeIn(b2), run_time=0.6); self.play(FadeIn(t2), run_time=0.5); self.wait(1.8)
        self.clear_but(cap)

        # ---------- 3. three inputs, one network ----------
        cap = self.swap(cap, self.caption("Three inputs, one network", "one branch for each kind of data"))
        e1 = self.caja("28 days × 3 sensors", SAMPLE, 3.4).move_to([-4.3, 1.0, 0])
        e2 = self.caja("model and depot", SAMPLE, 3.4).move_to([-4.3, -0.5, 0])
        e3 = self.caja("table: age, km, breakdowns…", SAMPLE, 3.4, sc=0.32).move_to([-4.3, -2.0, 0])
        r1 = self.caja("GRU → 16", MEAN, 2.2).move_to([-0.9, 1.0, 0])
        r2 = self.caja("embeddings → 3 + 3", MEAN, 2.2, sc=0.3).move_to([-0.9, -0.5, 0])
        r3 = self.caja("scaled → 5", MEAN, 2.2).move_to([-0.9, -2.0, 0])
        jun = self.caja("27 → 16 → 1", GOOD, 2.1).move_to([2.25, -0.5, 0])
        sal = VGroup(T("risk", 0.36, POINT, BOLD), T("of breakdown", 0.3, MUTED)).arrange(DOWN, buff=0.08).move_to([4.75, -0.5, 0])
        fl = VGroup(*[Arrow(a.get_right(), b.get_left(), buff=0.08, color=MUTED, stroke_width=3, max_tip_length_to_length_ratio=0.25) for a, b in ((e1, r1), (e2, r2), (e3, r3))],
                    *[Arrow(a.get_right(), jun.get_left(), buff=0.08, color=MUTED, stroke_width=3, max_tip_length_to_length_ratio=0.15) for a in (r1, r2, r3)],
                    Arrow(jun.get_right(), sal.get_left(), buff=0.1, color=MUTED, stroke_width=3, max_tip_length_to_length_ratio=0.3))
        self.mark("inputs"); self.play(LaggedStart(FadeIn(e1), FadeIn(e2), FadeIn(e3), lag_ratio=0.3), run_time=1.0)
        self.mark("branches"); self.play(LaggedStart(GrowArrow(fl[0]), GrowArrow(fl[1]), GrowArrow(fl[2]), FadeIn(r1), FadeIn(r2), FadeIn(r3), lag_ratio=0.15), run_time=1.2)
        self.mark("head"); self.play(LaggedStart(GrowArrow(fl[3]), GrowArrow(fl[4]), GrowArrow(fl[5]), FadeIn(jun), GrowArrow(fl[6]), FadeIn(sal), lag_ratio=0.15), run_time=1.2)
        nt = T("Adam, class weights and early stopping", 0.32, MUTED).move_to([0, -3.3, 0])
        self.play(FadeIn(nt), run_time=0.5); self.wait(1.6)
        self.clear_but(cap)

        # ---------- 4. simple first ----------
        cap = self.swap(cap, self.caption("First, a simple baseline", "$ per truck and month on the exam: less is better"))
        filas = [("never alert", media("never alert", 2), MUTED), ("logistic regression", media("logistic regression", 2), SAMPLE),
                 ("tree boosting", media("tree boosting", 2), GOOD), ("neural network (3 seeds)", media("full network", 2), MEAN)]
        barras = VGroup()
        for k, (nm, v, col) in enumerate(filas):
            y = 1.25 - k * 0.95
            bar = Rectangle(width=v / 340 * 6.0, height=0.5, stroke_width=0, fill_color=col, fill_opacity=0.85).move_to([-1.6, y, 0], aligned_edge=LEFT)
            barras.add(VGroup(T(nm, 0.32, INK).move_to([-1.85, y, 0], aligned_edge=RIGHT), bar, T(f"${fmt(v)}", 0.34, INK, BOLD).next_to(bar, RIGHT, buff=0.15)))
        self.mark("bars"); self.play(LaggedStart(*[FadeIn(b, shift=0.2 * RIGHT) for b in barras], lag_ratio=0.3), run_time=1.6); self.wait(0.6)
        nt = VGroup(T("all three save a third, and the network doesn't win", 0.36, WARN, BOLD), T("the information is in the table: age and history", 0.3, MUTED)).arrange(DOWN, buff=0.18).move_to([0, -2.85, 0])
        self.mark("nowin"); self.play(FadeIn(nt[0]), run_time=0.6); self.play(FadeIn(nt[1]), run_time=0.5); self.wait(1.8)
        self.clear_but(cap)

        # ---------- 5. curves ----------
        cap = self.swap(cap, self.caption("Train and stop in time", "loss per epoch: training and validation"))
        C = D["curvas"]["full network|0"]; tr = [c[0] for c in C["curva"]]; va = [c[1] for c in C["curva"]]
        ax = Axes(x_range=[1, 70, 10], y_range=[0.93, 1.12, 0.05], x_length=9.5, y_length=4.0, tips=False,
                  axis_config={"color": LINE, "stroke_width": 2}).move_to([0.2, -0.85, 0])
        lt = T("epoch", 0.28, MUTED).next_to(ax.x_axis, DOWN, buff=0.15).align_to(ax.x_axis, RIGHT)
        ctr = ax.plot_line_graph(list(range(1, len(tr) + 1)), tr, line_color=MEAN, add_vertex_dots=False, stroke_width=4)
        cva = ax.plot_line_graph(list(range(1, len(va) + 1)), va, line_color=POINT, add_vertex_dots=False, stroke_width=4)
        ley = VGroup(T("training", 0.3, MEAN, BOLD), T("validation", 0.3, POINT, BOLD)).arrange(RIGHT, buff=0.6).move_to([1.5, 1.55, 0])
        self.play(Create(ax), FadeIn(lt), FadeIn(ley), run_time=0.6)
        self.mark("curves"); self.play(Create(ctr), Create(cva), run_time=2.4, rate_func=linear)
        b = C["mejor"]
        lin = DashedLine(ax.c2p(b, 0.93), ax.c2p(b, 1.12), color=GOOD, stroke_width=3)
        dot = Dot(ax.c2p(b, va[b - 1]), radius=0.1, color=GOOD)
        tb = T(f"best: epoch {b}; stops at {len(va)}", 0.3, GOOD, BOLD).next_to(lin, LEFT, buff=0.2).shift(UP * 1.4)
        self.mark("best"); self.play(Create(lin), FadeIn(dot), FadeIn(tb), run_time=0.7); self.wait(1.8)
        self.clear_but(cap)

        # ---------- 6. the exam ----------
        cap = self.swap(cap, self.caption("The exam, just once", "600 new trucks × 4 months, 10% threshold"))
        vp, fp, fn, vn = cuenta(PR)
        cel = [(vp, "breakdowns avoided", GOOD), (fn, "breakdowns on the road", WARN), (fp, "unnecessary inspections", SAMPLE), (vn, "healthy, left alone", MUTED)]
        mat = VGroup()
        for k, (v, nm, col) in enumerate(cel):
            r, c = divmod(k, 2)
            box = RoundedRectangle(width=2.9, height=1.45, corner_radius=0.14, stroke_color=col, stroke_width=3, fill_color=col, fill_opacity=0.12).move_to([-2.3 + 3.1 * c, 0.75 - 1.6 * r, 0])
            mat.add(VGroup(box, VGroup(T(fmt(v, 0), 0.62, col, BOLD), T(nm, 0.28, INK)).arrange(DOWN, buff=0.12).move_to(box)))
        hd = VGroup(T("alert", 0.3, MUTED).move_to([-2.3, 1.75, 0]), T("don't alert", 0.3, MUTED).move_to([0.8, 1.75, 0]),
                    T("breakdown", 0.3, MUTED).move_to([-4.4, 0.75, 0]), T("healthy", 0.3, MUTED).move_to([-4.4, -0.85, 0]))
        self.mark("matrix"); self.play(FadeIn(hd), LaggedStart(*[FadeIn(m, scale=0.9) for m in mat], lag_ratio=0.2), run_time=1.4)
        cst = ((vp + fp) * 400 + fn * 4000) / N; nun = 4000 * Y.sum() / N
        tx = VGroup(T(f"${fmt(cst)}", 0.6, MEAN, BOLD), T("per truck and month", 0.3, MUTED), T(f"don't alert: ${fmt(nun)}", 0.3, MUTED),
                    T(f"{fmt(100 * round((nun - cst) / nun, 3))}% less", 0.4, GOOD, BOLD)).arrange(DOWN, buff=0.18).move_to([4.55, 0.0, 0])
        self.mark("cost"); self.play(FadeIn(tx, shift=0.2 * LEFT), run_time=0.8); self.wait(2.0)
        self.clear_but(cap)

        # ---------- 7. is it really better? ----------
        cap = self.swap(cap, self.caption("How much to trust the difference?", "bootstrap: 2,000 exams drawing trucks"))
        h = D["hist"]["eur"]
        nl = NumberLine(x_range=[-30, 20, 10], length=9, color=LINE, include_numbers=False).move_to([0, -0.3, 0])
        nums = VGroup(*[T(fmt(v, 0), 0.3, MUTED).next_to(nl.n2p(v), DOWN, buff=0.2) for v in range(-30, 21, 10)])
        bandi = Line(nl.n2p(h["lo"]), nl.n2p(h["hi"]), color=MEAN, stroke_width=14)
        cero = Line(nl.n2p(0) + 0.6 * UP, nl.n2p(0) + 0.12 * DOWN, color=INK, stroke_width=4)
        lab = T("$ the network saves versus boosting, per truck and month", 0.3, MUTED).move_to([0, -1.4, 0])
        lo = fmt(h['lo']); lo = ("−$" + lo[1:]) if lo.startswith("−") else "$" + lo
        ic = T(f"95% interval: from {lo} to ${fmt(h['hi'])}", 0.36, MEAN, BOLD).move_to([0, 1.2, 0])
        self.play(Create(nl), FadeIn(nums), FadeIn(lab), run_time=0.6)
        self.mark("ci"); self.play(Create(bandi), FadeIn(ic), run_time=0.8); self.play(FadeIn(cero), run_time=0.4)
        nt = T("zero is inside: the advantage isn't certain, for either one", 0.34, WARN, BOLD).move_to([0, -2.6, 0])
        self.play(FadeIn(nt), run_time=0.5); self.wait(1.8)
        self.clear_but(cap)

        # ---------- 8. what slips through ----------
        cap = self.swap(cap, self.caption("Look at the mistakes", "the breakdowns that slip through"))
        vis = np.array(D["vis"]); esc = (Y == 1) & (PR <= 0.1); sin = int((esc & (vis == 0)).sum()); tot = int(esc.sum())
        pts = VGroup(*[Dot(radius=0.11, color=WARN if k < sin else MUTED) for k in range(tot)]).arrange_in_grid(rows=4, buff=0.13).move_to([0, 0.6, 0])
        self.mark("misses"); self.play(LaggedStart(*[FadeIn(p) for p in pts], lag_ratio=0.01), run_time=1.2)
        nt = VGroup(T(f"{tot} breakdowns on the road; {sin} with no signal at all in the sensors", 0.36, INK, BOLD),
                    T("no model can see them coming", 0.32, WARN, BOLD)).arrange(DOWN, buff=0.2).move_to([0, -2.0, 0])
        self.play(FadeIn(nt), run_time=0.6); self.wait(2.0)
        self.clear_but(cap)

        # ---------- 9. and afterward: monitor ----------
        cap = self.swap(cap, self.caption("And afterward: monitor and retrain", "the model is the small part"))
        nodos = [("data", [-3.6, 0.9, 0]), ("model", [0.0, 0.9, 0]), ("alerts", [3.6, 0.9, 0]), ("shop and breakdowns", [0.0, -1.5, 0])]
        cj = VGroup(*[self.caja(n, MEAN if k < 3 else GOOD, 2.6 if k < 3 else 3.2, 0.8, sc=0.36 if k < 3 else 0.32).move_to(p) for k, (n, p) in enumerate(nodos)])
        ar = VGroup(Arrow(cj[0].get_right(), cj[1].get_left(), buff=0.1, color=MUTED, stroke_width=3), Arrow(cj[1].get_right(), cj[2].get_left(), buff=0.1, color=MUTED, stroke_width=3),
                    Arrow(cj[2].get_bottom(), cj[3].get_right(), buff=0.1, color=MUTED, stroke_width=3), Arrow(cj[3].get_left(), cj[0].get_bottom(), buff=0.1, color=MUTED, stroke_width=3))
        self.mark("loop"); self.play(LaggedStart(*[FadeIn(c) for c in cj], lag_ratio=0.2), run_time=1.0); self.play(LaggedStart(*[GrowArrow(a) for a in ar], lag_ratio=0.2), run_time=1.0)
        w1 = T("autoencoder: is odd data coming in?", 0.3, POINT, BOLD).next_to(cj[0], UP, buff=0.25)
        w2 = T("retrain with the same exam", 0.3, GOOD, BOLD).move_to([0, -2.8, 0])
        self.mark("watch"); self.play(FadeIn(w1), run_time=0.5); self.play(FadeIn(w2), run_time=0.5); self.wait(1.8)
        self.clear_but(cap)

        # ---------- Closing ----------
        self.mark("legend")
        self.play(FadeOut(cap), run_time=0.6)
        e1 = Text("End-to-end project with networks", font=FONT, weight=BOLD, color=INK).scale(0.7)
        e2 = VGroup(T("the question and its price give the threshold", 0.42, MUTED),
                    T("split by truck and by time, before looking", 0.42, MUTED),
                    T("a simple baseline: here, the network doesn't beat it", 0.42, MUTED),
                    T("one exam, with its margin of error, and monitoring afterward", 0.42, MUTED)).arrange(DOWN, buff=0.18)
        VGroup(e1, e2).arrange(DOWN, buff=0.45)
        self.mark("end_text"); self.play(Write(e1), run_time=1.0); self.play(FadeIn(e2, shift=0.2 * UP, lag_ratio=0.15), run_time=1.0)
        self.wait(3.3)
        self.mark("fade"); self.play(FadeOut(VGroup(e1, e2)), run_time=1)
        self.marks["total"] = [round(self.renderer.time, 3)]
        json.dump(self.marks, open("marks.json", "w"), indent=1)
