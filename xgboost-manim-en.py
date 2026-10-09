import json
import numpy as np
import xgboost as xgb
from manim import *

# Dark palette of the Visual ML series
BG = "#0E131B"; INK = "#E4EAF3"; MUTED = "#9BA7B9"; LINE = "#2B3545"
MEAN = "#6EA6FF"; SAMPLE = "#AE9FF3"; POINT = "#FF9A55"; GOOD = "#4CC38A"; WARN = "#F08A5D"
FONT = "Inter"
config.background_color = BG
EMB = json.load(open("embed-xg-en.json"))


def truth(v):
    return 22 + 0.004 * (v - 62) ** 2


class XGBoost(Scene):
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

    def clear_but(self, cap):
        self.play(*[FadeOut(m) for m in self.mobjects if m is not cap], run_time=0.6)

    def construct(self):
        self.marks = {}
        title = Text("XGBoost", font=FONT, weight=BOLD, color=INK).scale(1.5)
        sub = Text("boosting with brakes, ready for production", font=FONT, color=MUTED).scale(0.5)
        VGroup(title, sub).arrange(DOWN, buff=0.35)
        self.mark("title"); self.play(Write(title), run_time=1.2)
        self.mark("subtitle"); self.play(FadeIn(sub, shift=0.2 * UP)); self.wait(1.2)
        self.mark("title_out"); self.play(FadeOut(title), FadeOut(sub))

        # data: fuel use vs. speed
        r = np.random.default_rng(4); x = r.uniform(20, 100, 90); y = truth(x) + 1.6 * r.standard_normal(90)
        xv = r.uniform(20, 100, 300); yv = truth(xv) + 1.6 * r.standard_normal(300)
        X = lambda v: -5.8 + (v - 20) / 80 * 11.6; Y = lambda c: -3.2 + (c - 18) / 22 * 4.9
        grid = np.linspace(20, 100, 321)
        m = xgb.XGBRegressor(n_estimators=300, learning_rate=0.3, max_depth=2, tree_method="exact", base_score=float(y.mean())).fit(x.reshape(-1, 1), y)
        B = m.get_booster(); D = xgb.DMatrix(grid.reshape(-1, 1)); Dt = xgb.DMatrix(x.reshape(-1, 1)); Dv = xgb.DMatrix(xv.reshape(-1, 1))
        stage = lambda k: B.predict(D, iteration_range=(0, k))
        curve = lambda k: VMobject(color=MEAN, stroke_width=5).set_points_as_corners([[X(v), Y(c), 0] for v, c in zip(grid, stage(k))])

        # ---------- 1. each tree corrects ----------
        cap = self.swap(None, self.caption("Each tree fixes what is left", "fuel use by average speed: 90 trips"))
        axis = VGroup(Line([X(20), Y(18), 0], [X(100), Y(18), 0], color=LINE, stroke_width=3),
                      *[Text(f"{v} km/h", font=FONT, color=MUTED).scale(0.26).move_to([X(v), Y(18) - 0.28, 0]) for v in (20, 40, 60, 80, 100)])
        dots = VGroup(*[Dot([X(a), Y(b), 0], radius=0.06, color=POINT) for a, b in zip(x, y)])
        self.mark("data"); self.play(FadeIn(axis), LaggedStart(*[GrowFromCenter(d) for d in dots], lag_ratio=0.01), run_time=1.2)
        base = Line([X(20), Y(y.mean()), 0], [X(100), Y(y.mean()), 0], color=MEAN, stroke_width=5)
        lab = Text("0 trees: the mean", font=FONT, weight=SEMIBOLD, color=INK).scale(0.42).move_to([3.6, 1.9, 0])
        self.mark("base"); self.play(Create(base), FadeIn(lab), run_time=0.8)
        cur = base
        for k in (1, 3, 10, 40):
            new = curve(k); nl = Text(f"{k} tree{'s' if k > 1 else ''}", font=FONT, weight=SEMIBOLD, color=INK).scale(0.42).move_to([3.6, 1.9, 0])
            self.mark("round"); self.play(Transform(cur, new), ReplacementTransform(lab, nl), run_time=0.9); lab = nl; self.wait(0.35)
        self.wait(0.8)

        # ---------- 2. stopping in time ----------
        cap = self.swap(cap, self.caption("Stop in time", "adding more and more trees ends up memorizing noise"))
        et = [np.sqrt(np.mean((B.predict(Dt, iteration_range=(0, k)) - y) ** 2)) for k in range(1, 301)]
        ev = [np.sqrt(np.mean((B.predict(Dv, iteration_range=(0, k)) - yv) ** 2)) for k in range(1, 301)]
        best = int(np.argmin(ev))
        self.play(FadeOut(VGroup(cur, dots, lab, axis)), run_time=0.5)
        PX = lambda k: -5.8 + k / 300 * 11.6; PY = lambda e: -3.2 + e / 2.5 * 4.6
        ax2 = VGroup(Line([PX(0), PY(0), 0], [PX(300), PY(0), 0], color=MUTED, stroke_width=2), Text("trees →", font=FONT, color=MUTED).scale(0.3).move_to([PX(285), PY(0) - 0.28, 0]))
        ct = VMobject(color=POINT, stroke_width=4).set_points_as_corners([[PX(k + 1), PY(min(e, 2.5)), 0] for k, e in enumerate(et)])
        cv = DashedVMobject(VMobject(color=SAMPLE, stroke_width=4).set_points_as_corners([[PX(k + 1), PY(min(e, 2.5)), 0] for k, e in enumerate(ev)]), num_dashes=80)
        l1 = Text("trips it knows", font=FONT, color=POINT).scale(0.34).move_to([PX(250), PY(min(et[-1], 2.5)) - 0.3, 0])
        l2 = Text("new trips", font=FONT, color=SAMPLE).scale(0.34).move_to([PX(250), PY(min(ev[-1], 2.5)) + 0.3, 0])
        self.mark("curves"); self.play(FadeIn(ax2), Create(ct), Create(cv), FadeIn(l1), FadeIn(l2), run_time=2.2)
        bd = Dot([PX(best + 1), PY(ev[best]), 0], radius=0.12, color=GOOD)
        bl = Text(f"early stopping: {best + 1} trees", font=FONT, weight=SEMIBOLD, color=GOOD).scale(0.42).next_to(bd, UP, buff=0.25)
        bl.shift(RIGHT * max(0, -6.7 - bl.get_left()[0]))  # keep the label inside the frame
        self.mark("stop"); self.play(GrowFromCenter(bd), FadeIn(bl), run_time=0.7); self.wait(1.4)
        self.clear_but(cap)

        # ---------- 3. a fine for complexity ----------
        cap = self.swap(cap, self.caption("A fine for getting complicated", "only splits that improve things enough are kept"))
        def tree_curve(gamma):
            t = xgb.XGBRegressor(n_estimators=1, learning_rate=1.0, max_depth=5, gamma=gamma, tree_method="exact", base_score=float(y.mean())).fit(x.reshape(-1, 1), y)
            p = t.predict(grid.reshape(-1, 1)); n = int((np.abs(np.diff(p)) > 1e-9).sum()) + 1
            return VMobject(color=MEAN, stroke_width=5).set_points_as_corners([[X(v), Y(c), 0] for v, c in zip(grid, p)]), n
        axis.set_opacity(1); dots.set_opacity(1)
        c0, n0 = tree_curve(0)
        info = Text(f"no fine: {n0} steps", font=FONT, weight=SEMIBOLD, color=INK).scale(0.42).move_to([3.4, 1.9, 0])
        self.mark("tree"); self.play(FadeIn(axis), FadeIn(dots), Create(c0), FadeIn(info), run_time=1.2); self.wait(0.8)
        for gm in (20, 60):
            c1, n1 = tree_curve(gm)
            i1 = Text(f"fine γ = {gm}: {n1} steps", font=FONT, weight=SEMIBOLD, color=GOOD).scale(0.42).move_to([3.4, 1.9, 0])
            self.mark("prune"); self.play(Transform(c0, c1), ReplacementTransform(info, i1), run_time=1.0); info = i1; self.wait(0.9)
        self.clear_but(cap)

        # ---------- 4. missing values ----------
        cap = self.swap(cap, self.caption("If the value is missing, the model picks the side", "truck G: broken load sensor, burns 42 L"))
        node = VGroup(RoundedRectangle(width=3.6, height=0.9, corner_radius=0.2, stroke_color=INK, stroke_width=3, fill_color=BG, fill_opacity=1), Text("load < 15 t?", font=FONT, weight=SEMIBOLD, color=INK).scale(0.46)).move_to([0, 1.1, 0])
        lft = VGroup(RoundedRectangle(width=3.4, height=1.0, corner_radius=0.2, stroke_color=SAMPLE, stroke_width=3), Text("light: A, B, C", font=FONT, color=INK).scale(0.4)).move_to([-3.4, -1.0, 0])
        rgt = VGroup(RoundedRectangle(width=3.4, height=1.0, corner_radius=0.2, stroke_color=POINT, stroke_width=3), Text("heavy: D, E, F", font=FONT, color=INK).scale(0.4)).move_to([3.4, -1.0, 0])
        a1 = Arrow(node.get_bottom(), lft.get_top(), color=MUTED, buff=0.1); a2 = Arrow(node.get_bottom(), rgt.get_top(), color=MUTED, buff=0.1)
        self.mark("node"); self.play(FadeIn(node), GrowArrow(a1), GrowArrow(a2), FadeIn(lft), FadeIn(rgt), run_time=1.0)
        g = VGroup(Circle(radius=0.32, color=WARN, stroke_width=3).set_fill(BG, 1), Text("G ?", font=FONT, weight=BOLD, color=WARN).scale(0.36)).move_to([0, 2.15, 0]).set_z_index(5)
        self.play(FadeIn(g), run_time=0.4)
        tl = Text("gain 62.9", font=FONT, color=INK).scale(0.4).next_to(lft, DOWN, buff=0.25)
        self.mark("tryL"); self.play(g.animate.move_to([-3.4 - 2.15, -1.0, 0]), run_time=0.8); self.play(FadeIn(tl), run_time=0.4)
        tr = Text("gain 146.9", font=FONT, weight=BOLD, color=GOOD).scale(0.44).next_to(rgt, DOWN, buff=0.25)
        self.mark("tryR"); self.play(g.animate.move_to([3.4 + 2.15, -1.0, 0]), run_time=0.9); self.play(FadeIn(tr), run_time=0.4)
        dl = Text("learned rule: no value → with the heavy ones", font=FONT, weight=SEMIBOLD, color=GOOD).scale(0.44).move_to([0, -3.0, 0])
        self.mark("rule"); self.play(FadeIn(dl), tl.animate.set_opacity(0.35), run_time=0.6); self.wait(1.4)
        self.clear_but(cap)

        # ---------- 5. SHAP ----------
        t = EMB["trips"][4]; cols = EMB["cols"]; base0 = EMB["base"]
        cap = self.swap(cap, self.caption("Why did it say that? SHAP", f"a real trip from the model: {t['pred']:.1f} L/100 km"))
        names = {"carga_t": "load", "vel_media": "speed", "pendiente": "grade", "ralenti_pct": "idling", "temp_ext": "temperature", "urbano": "city" if t["vals"]["urbano"] else "highway"}
        order = sorted(range(len(cols)), key=lambda j: -abs(t["shap"][j]))
        vals = [base0]
        for j in order: vals.append(vals[-1] + t["shap"][j])
        lo, hi = min(vals) - 1, max(vals) + 1; SX = lambda v: -2.0 + (v - lo) / (hi - lo) * 7.5
        bl = DashedLine([SX(base0), 1.9, 0], [SX(base0), -3.3, 0], color=MUTED, stroke_width=2)
        bt = Text(f"fleet average {base0:.1f}", font=FONT, color=MUTED).scale(0.32).move_to([SX(base0), -3.55, 0])
        self.mark("shap0"); self.play(Create(bl), FadeIn(bt), run_time=0.6)
        acc = base0
        for k, j in enumerate(order):
            v = t["shap"][j]; yk = 1.5 - k * 0.72
            rect = Rectangle(width=max(0.04, abs(SX(acc + v) - SX(acc))), height=0.48, stroke_width=0, fill_color=POINT if v > 0 else MEAN, fill_opacity=0.95).move_to([(SX(acc) + SX(acc + v)) / 2, yk, 0])
            nm = Text(names[cols[j]], font=FONT, color=INK).scale(0.36).move_to([-3.2, yk, 0], aligned_edge=RIGHT)
            vt = Text(("+" if v > 0 else "−") + f"{abs(v):.1f}", font=FONT, color=INK).scale(0.32).next_to(rect, RIGHT, buff=0.1)
            self.mark("bar"); self.play(GrowFromEdge(rect, LEFT if v > 0 else RIGHT), FadeIn(nm), FadeIn(vt), run_time=0.45)
            acc += v
        fin = Text(f"= {t['pred']:.1f} L/100 km", font=FONT, weight=BOLD, color=INK).scale(0.46).move_to([SX(acc), 1.5 - len(order) * 0.72 - 0.05, 0])
        self.mark("sum"); self.play(FadeIn(fin), run_time=0.5); self.wait(1.8)
        self.clear_but(cap)

        # ---------- Closing ----------
        self.mark("legend")
        self.play(FadeOut(cap), run_time=0.6)
        e1 = Text("Small corrections, with brakes.", font=FONT, weight=BOLD, color=INK).scale(0.72)
        e2 = Text("And every prediction, with its reasons.", font=FONT, color=MUTED).scale(0.5)
        VGroup(e1, e2).arrange(DOWN, buff=0.4)
        self.mark("end_text"); self.play(Write(e1), run_time=1.4); self.play(FadeIn(e2, shift=0.2 * UP), run_time=0.8)
        self.wait(2.0)
        self.mark("fade"); self.play(FadeOut(VGroup(e1, e2)), run_time=1)
        self.marks["total"] = [round(self.renderer.time, 3)]
        json.dump(self.marks, open("marks.json", "w"), indent=1)
