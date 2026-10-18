import json
from manim import *

# Dark palette of the series
BG = "#0E131B"; INK = "#E4EAF3"; MUTED = "#9BA7B9"; LINE = "#2B3545"
MEAN = "#6EA6FF"; SAMPLE = "#AE9FF3"; POINT = "#FF9A55"; GOOD = "#4CC38A"; WARN = "#F08A5D"
FONT = "Inter"
config.background_color = BG


def T(s, sc=0.45, col=INK, w=None):
    return Text(s, font=FONT, color=col, weight=w or NORMAL).scale(sc)


class Condicional(Scene):
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
        title = Text("Conditional probability", font=FONT, weight=BOLD, color=INK).scale(1.05)
        sub = Text("what changes when you already know something", font=FONT, color=MUTED).scale(0.5)
        VGroup(title, sub).arrange(DOWN, buff=0.35)
        self.mark("title"); self.play(Write(title), run_time=1.2)
        self.mark("subtitle"); self.play(FadeIn(sub, shift=0.2 * UP)); self.wait(1.7)
        self.mark("title_out"); self.play(FadeOut(title), FadeOut(sub))

        # ---------- 1. a thousand trips ----------
        cap = self.swap(None, self.caption("1,000 truck trips", "left: it rained (200)  ·  orange: hard-braking alert"))
        cs = 0.17; gap = 0.6; x0 = -(50 * cs + gap) / 2; y0 = 1.45
        rain, dry = VGroup(), VGroup()
        rain_al, dry_al = VGroup(), VGroup()
        for k in range(200):
            c, r = k % 10, k // 10; a = k < 50
            sq = Square(cs * 0.8, stroke_width=0, fill_color=POINT if a else MEAN, fill_opacity=1 if a else 0.45).move_to([x0 + cs / 2 + c * cs, y0 - r * cs, 0])
            rain.add(sq); (rain_al if a else VGroup()).add(sq)
        for k in range(800):
            c, r = k % 40, k // 40; a = k < 80
            sq = Square(cs * 0.8, stroke_width=0, fill_color=POINT if a else MUTED, fill_opacity=1 if a else 0.35).move_to([x0 + 10 * cs + gap + cs / 2 + c * cs, y0 - r * cs, 0])
            dry.add(sq); (dry_al if a else VGroup()).add(sq)
        lr = T("rain", 0.4, INK, BOLD).next_to(rain, DOWN, buff=0.2)
        ld = T("no rain", 0.4, INK, BOLD).next_to(dry, DOWN, buff=0.2)
        self.mark("grid"); self.play(FadeIn(rain, lag_ratio=0.002), FadeIn(dry, lag_ratio=0.0005), FadeIn(lr), FadeIn(ld), run_time=1.6)
        tot = T("alert on 130 of 1,000 = 13%", 0.5, POINT, BOLD).move_to([0, -2.9, 0])
        self.mark("count"); self.play(FadeIn(tot, shift=0.2 * UP), run_time=0.7); self.wait(3.2)

        # ---------- 2. knowing it rained ----------
        cap = self.swap(cap, self.caption("You know it rained", "keep only those trips and count again"))
        box = SurroundingRectangle(rain, color=INK, buff=0.08, stroke_width=3)
        self.mark("keep"); self.play(dry.animate.set_opacity(0.08), Create(box), FadeOut(tot), run_time=0.9)
        t2 = T("alert on 50 of 200 = 25%", 0.5, POINT, BOLD).move_to([0, -2.9, 0])
        self.mark("recount"); self.play(FadeIn(t2, shift=0.2 * UP), run_time=0.7); self.wait(3.5)

        # ---------- 3. turning it around ----------
        cap = self.swap(cap, self.caption("Turning it around isn't the same", "now you know there was an alert: did it rain?"))
        box2 = VGroup(SurroundingRectangle(rain_al, color=INK, buff=0.06, stroke_width=3), SurroundingRectangle(dry_al, color=INK, buff=0.06, stroke_width=3))
        dim = [m.animate.set_opacity(0.08) for m in list(rain[50:]) + list(dry[80:])]
        self.mark("flip")
        self.play(dry[:80].animate.set_opacity(1), *dim, ReplacementTransform(box, box2), FadeOut(t2), run_time=1.0)
        t3 = T("rain on 50 of 130 = 38.5%", 0.5, MEAN, BOLD).move_to([0, -2.9, 0])
        self.play(FadeIn(t3, shift=0.2 * UP), run_time=0.7); self.wait(1.8)
        eq = VGroup(T("P(alert | rain) = 25%", 0.46, POINT, BOLD), T("≠", 0.6, INK, BOLD), T("P(rain | alert) = 38.5%", 0.46, MEAN, BOLD)).arrange(RIGHT, buff=0.35).move_to([0, -2.9, 0])
        self.mark("flip2"); self.play(FadeOut(t3), FadeIn(eq), run_time=0.8); self.wait(3.0)
        self.play(FadeOut(VGroup(rain, dry, lr, ld, box2, eq)), run_time=0.6)

        # ---------- 4. breakdown and alert ----------
        cap = self.swap(cap, self.caption("With rare breakdowns, even worse", "1,000 trucks, 20 breakdowns · alert on 90% of breakdowns and on 5% of healthy trucks"))
        def bar(y, frac, col, lab, num):
            bgr = Rectangle(width=8, height=0.55, stroke_width=0, fill_color=LINE, fill_opacity=1).move_to([0.9, y, 0])
            fl = Rectangle(width=8 * frac, height=0.55, stroke_width=0, fill_color=col, fill_opacity=1).align_to(bgr, LEFT).set_y(y)
            lt = T(lab, 0.38, MUTED).next_to(bgr, UP, buff=0.12).align_to(bgr, LEFT)
            nt = T(num, 0.5, col, BOLD).next_to(bgr, LEFT, buff=0.3)
            return VGroup(bgr, fl, lt, nt)
        b1 = bar(0.7, 0.9, GOOD, "of the 20 breakdowns, 18 trigger an alert", "90%")
        b2 = bar(-1.2, 18 / 67, WARN, "of the 67 alerts (18 + 49 false), 18 are breakdowns", "27%")
        self.mark("avr"); self.play(FadeIn(b1[0]), FadeIn(b1[2]), GrowFromEdge(b1[1], LEFT), FadeIn(b1[3]), run_time=0.9); self.wait(1.6)
        self.mark("avr2"); self.play(FadeIn(b2[0]), FadeIn(b2[2]), GrowFromEdge(b2[1], LEFT), FadeIn(b2[3]), run_time=0.9)
        nb = T("to turn it around you need to know how rare breakdowns are: Bayes", 0.38, INK).move_to([0, -2.6, 0])
        self.play(FadeIn(nb), run_time=0.6); self.wait(3.5)
        self.play(FadeOut(VGroup(b1, b2, nb)), run_time=0.6)

        # ---------- 5. independence ----------
        cap = self.swap(cap, self.caption("Independent: knowing one doesn't change the other", "share of trips with an alert, rain or no rain"))
        H = 3.4; base = -2.3; L0, W1, W2 = -5.0, 2.2, 7.6
        def col(x, w, p, c):
            return VGroup(Rectangle(width=w, height=H, stroke_color=MEAN, stroke_width=1.5, fill_color=MEAN, fill_opacity=0.15).move_to([x + w / 2, base + H / 2, 0]),
                          Rectangle(width=w, height=H * p, stroke_width=0, fill_color=c, fill_opacity=0.9).move_to([x + w / 2, base + H * p / 2, 0]))
        c1 = col(L0, W1, 0.25, POINT); c2 = col(L0 + W1 + 0.2, W2, 0.10, POINT)
        l1 = T("rain", 0.38, INK, BOLD).next_to(c1, DOWN, buff=0.15); l2 = T("no rain", 0.38, INK, BOLD).next_to(c2, DOWN, buff=0.15)
        p1 = T("25%", 0.5, INK, BOLD).move_to([L0 + W1 / 2, base + H * 0.25 + 0.35, 0]); p2 = T("10%", 0.5, INK, BOLD).move_to([L0 + W1 + 0.2 + W2 / 2, base + H * 0.10 + 0.35, 0])
        self.mark("ind"); self.play(FadeIn(c1), FadeIn(c2), FadeIn(l1), FadeIn(l2), FadeIn(p1), FadeIn(p2), run_time=0.9); self.wait(2.4)
        c1b = col(L0, W1, 0.10, POINT)
        p1b = T("10%", 0.5, INK, BOLD).move_to([L0 + W1 / 2, base + H * 0.10 + 0.35, 0])
        self.mark("morph"); self.play(Transform(c1, c1b), Transform(p1, p1b), run_time=1.0)
        it = T("same height: the table multiplies (20% × 10% = 2%)", 0.4, GOOD, BOLD).move_to([0, 1.85, 0])
        self.play(FadeIn(it, shift=0.2 * UP), run_time=0.6); self.wait(3.2)
        self.play(FadeOut(VGroup(c1, c2, l1, l2, p1, p2, it)), run_time=0.6)

        # ---------- 6. Simpson ----------
        cap = self.swap(cap, self.caption("Simpson's paradox", "on-time deliveries: North wins in the city and on the highway…"))
        y0b, sc = -2.4, 3.6 / 0.5     # the axis goes from 50% to 100%
        def sbar(x, v, c, lab):
            h = (v - 0.5) * sc
            return VGroup(Rectangle(width=0.9, height=h, stroke_width=0, fill_color=c, fill_opacity=0.85).move_to([x, y0b + h / 2, 0]),
                          T(f"{round(v * 100)}%", 0.36, INK, BOLD).move_to([x, y0b + h + 0.25, 0]),
                          T(lab, 0.3, MUTED).move_to([x, y0b - 0.3, 0]))
        nN = VGroup(sbar(-4.2, 0.75, MEAN, "city"), sbar(-3.0, 0.95, SAMPLE, "highway"))
        nS = VGroup(sbar(1.6, 0.70, MEAN, "city"), sbar(2.8, 0.90, SAMPLE, "highway"))
        hN = T("North", 0.45, INK, BOLD).move_to([-3.6, 1.9, 0]); hS = T("South", 0.45, INK, BOLD).move_to([2.2, 1.9, 0])
        self.mark("simp1"); self.play(FadeIn(hN), FadeIn(hS), *[FadeIn(b, shift=0.2 * UP) for b in [nN, nS]], run_time=1.0); self.wait(2.8)
        cap2 = self.caption("Simpson's paradox", "…but loses overall: it does mostly city (80%), South mostly highway")
        def tbar(x, v, c):
            h = (v - 0.5) * sc
            return VGroup(Rectangle(width=1.0, height=h, stroke_width=3, stroke_color=INK, fill_color=c, fill_opacity=0.35).move_to([x, y0b + h / 2, 0]),
                          T(f"total {round(v * 100)}%", 0.4, c, BOLD).move_to([x, y0b + h + 0.28, 0]),
                          T("total", 0.3, MUTED).move_to([x, y0b - 0.3, 0]))
        tN = tbar(-1.6, 0.79, WARN); tS = tbar(4.3, 0.86, GOOD)
        self.mark("simp2"); self.play(Transform(cap, cap2), FadeIn(tN, shift=0.2 * UP), FadeIn(tS, shift=0.2 * UP), run_time=1.0); self.wait(4.0)
        self.play(*[FadeOut(m) for m in self.mobjects if m is not cap], run_time=0.6)

        # ---------- Closing ----------
        self.mark("legend")
        self.play(FadeOut(cap), run_time=0.6)
        e1 = Text("Knowing something = keep one part and count again.", font=FONT, weight=BOLD, color=INK).scale(0.6)
        e2 = Text("And order matters: P(A | B) is not P(B | A).", font=FONT, color=MUTED).scale(0.48)
        VGroup(e1, e2).arrange(DOWN, buff=0.4)
        self.mark("end_text"); self.play(Write(e1), run_time=1.4); self.play(FadeIn(e2, shift=0.2 * UP), run_time=0.8)
        self.wait(2.8)
        self.mark("fade"); self.play(FadeOut(VGroup(e1, e2)), run_time=1)
        self.marks["total"] = [round(self.renderer.time, 3)]
        json.dump(self.marks, open("marks.json", "w"), indent=1)
