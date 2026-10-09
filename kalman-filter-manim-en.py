import json
import numpy as np
from manim import *

# Dark palette of the Visual ML series
BG = "#0E131B"; INK = "#E4EAF3"; MUTED = "#9BA7B9"; LINE = "#2B3545"
MEAN = "#6EA6FF"; SAMPLE = "#AE9FF3"; POINT = "#FF9A55"; GOOD = "#4CC38A"; WARN = "#F08A5D"
FONT = "Inter"
config.background_color = BG


def kf_run(Z, q, r, v0):
    """Constant-velocity Kalman per axis (1 s step); Z with None where there is no measurement."""
    out = []
    S = [dict(x=Z[0][d], v=v0[d], P=np.array([[r, 0.0], [0.0, 4.0]])) for d in range(2)]
    F = np.array([[1, 1], [0, 1.0]]); Q = q * np.array([[1 / 3, 1 / 2], [1 / 2, 1.0]])
    for k, z in enumerate(Z):
        for d, s in enumerate(S):
            if k:
                x = F @ np.array([s["x"], s["v"]]); s["x"], s["v"] = x; s["P"] = F @ s["P"] @ F.T + Q
            if z is not None:
                P = s["P"]; Sv = P[0, 0] + r; K = P[:, 0] / Sv; y = z[d] - s["x"]
                s["x"] += K[0] * y; s["v"] += K[1] * y; s["P"] = P - np.outer(K, P[0])
        out.append((S[0]["x"], S[1]["x"], S[0]["v"], S[1]["v"], np.sqrt(S[0]["P"][0, 0])))
    return np.array(out)


class Kalman(Scene):
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

    def bell(self, mu, var, col, X, base, amp, fill=0.18, w=4):
        sd = np.sqrt(var); f = lambda x: np.exp(-0.5 * ((x - mu) / sd) ** 2)
        xs = np.linspace(mu - 3.2 * sd, mu + 3.2 * sd, 120)
        pts = [X(x) + np.array([0, base + amp(sd) * f(x), 0]) for x in xs]
        curve = VMobject(color=col, stroke_width=w).set_points_smoothly(pts)
        area = Polygon(*([X(xs[0]) + [0, base, 0]] + pts + [X(xs[-1]) + [0, base, 0]]), stroke_width=0, fill_color=col, fill_opacity=fill)
        return VGroup(area, curve)

    def construct(self):
        self.marks = {}
        title = Text("Kalman filter", font=FONT, weight=BOLD, color=INK).scale(1.2)
        sub = Text("predict, measure and correct, every second", font=FONT, color=MUTED).scale(0.5)
        VGroup(title, sub).arrange(DOWN, buff=0.35)
        self.mark("title"); self.play(Write(title), run_time=1.4)
        self.mark("subtitle"); self.play(FadeIn(sub, shift=0.2 * UP)); self.wait(1.2)
        self.mark("title_out"); self.play(FadeOut(title), FadeOut(sub))

        # ---------- 1. predict and correct (example from the technical version) ----------
        cap = self.swap(None, self.caption("Two opinions, one answer", "the prediction (purple) and the GPS (orange) combine (blue)"))
        X = lambda x: np.array([-6.0 + x * 0.115, 0, 0])               # 0..100 m → screen
        base = -2.5; amp = lambda sd: 14.0 / sd
        road = Line(X(-5) + [0, base, 0], X(102) + [0, base, 0], color=LINE, stroke_width=6)
        ticks = VGroup(*[Text(f"{m} m", font=FONT, color=MUTED).scale(0.28).move_to(X(m) + [0, base - 0.32, 0]) for m in (0, 20, 40, 60, 80, 100)])
        truck = VGroup(RoundedRectangle(width=0.5, height=0.28, corner_radius=0.05, stroke_width=0, fill_color=INK, fill_opacity=1),
                       RoundedRectangle(width=0.17, height=0.2, corner_radius=0.03, stroke_width=0, fill_color=INK, fill_opacity=1).shift([0.34, -0.04, 0]),
                       Dot([-0.14, -0.17, 0], radius=0.06, color=INK), Dot([0.28, -0.17, 0], radius=0.06, color=INK))
        cur = self.bell(0, 12.5, MEAN, X, base, amp)
        truck.move_to(X(0) + [0, base - 0.85, 0])
        self.mark("road"); self.play(Create(road), FadeIn(ticks), FadeIn(cur), FadeIn(truck), run_time=1.0)
        steps = [(20, 25, 26, 0.5, 23, 12.5), (43, 25, 39, 0.5, 41, 12.5), (61, 25, None, None, 61, 25), (81, 37.5, 85, 0.6, 83.4, 15)]
        info = None
        for k, (xp, Pp, z, K, x, P) in enumerate(steps):
            pred = self.bell(xp, Pp, SAMPLE, X, base, amp)
            self.mark("pred"); self.play(Transform(cur, pred), truck.animate.move_to(X(xp) + [0, base - 0.85, 0]), run_time=0.9)
            if z is None:
                lab = Text(f"second {k + 1}: no GPS (bridge), the doubt grows", font=FONT, weight=SEMIBOLD, color=WARN).scale(0.4).move_to([0, 1.7, 0])
                new = lab
            else:
                meas = self.bell(z, 25, POINT, X, base, amp)
                self.mark("gps"); self.play(FadeIn(meas), run_time=0.5)
                post = self.bell(x, P, MEAN, X, base, amp, fill=0.28, w=5)
                lab = Text(f"second {k + 1}: it listens to the GPS {int(K * 100)}%  →  {x:g} m", font=FONT, weight=SEMIBOLD, color=INK).scale(0.4).move_to([0, 1.7, 0])
                self.mark("fuse"); self.play(Transform(cur, post), FadeOut(meas), run_time=0.9)
                new = lab
            self.play(FadeIn(new) if info is None else ReplacementTransform(info, new), run_time=0.4)
            info = new; self.wait(0.6 if k < 3 else 1.2)
        self.play(FadeOut(VGroup(road, ticks, cur, truck, info)), run_time=0.6)

        # ---------- 2. route with noisy GPS ----------
        cap = self.swap(cap, self.caption("A noisy GPS, a clean track", "orange: GPS readings · blue: the filter"))
        T = 70
        for seed in range(200):
            rng = np.random.default_rng(seed); h, w, s = 0.0, 0.0, 8.0; pos = [np.zeros(2)]
            for k in range(1, T):
                w = 0.85 * w + rng.uniform(-0.06, 0.06); h += w; s = np.clip(s + rng.uniform(-0.4, 0.4), 4, 14)
                pos.append(pos[-1] + s * np.array([np.cos(h), np.sin(h)]))
            ext = np.ptp(np.array(pos), 0)
            if ext[0] > 2.0 * ext[1] and ext[1] > 80: break
        pos = np.array(pos); sg = 8.0
        Z = pos + sg * rng.standard_normal(pos.shape)
        hd = np.unwrap(np.arctan2(*np.diff(pos, axis=0).T[::-1]))
        k0 = min(range(18, 50), key=lambda k: abs(hd[k + 12] - hd[k]))
        tun = (k0 <= np.arange(T)) & (np.arange(T) < k0 + 12)
        E = kf_run([None if t else z for z, t in zip(Z, tun)], 0.35, sg * sg, pos[1] - pos[0])
        lo, hi = pos.min(0) - 25, pos.max(0) + 25; scale = min(12.0 / (hi[0] - lo[0]), 5.3 / (hi[1] - lo[1]))
        ctr = (lo + hi) / 2
        M = lambda p: np.array([(p[0] - ctr[0]) * scale, (p[1] - ctr[1]) * scale - 0.85, 0])
        truth = DashedVMobject(VMobject(color=MUTED, stroke_width=2.5).set_points_smoothly([M(p) for p in pos]), num_dashes=90)
        self.mark("route"); self.play(Create(truth), run_time=1.0)
        gps = VGroup(*[Dot(M(z), radius=0.05, color=POINT) for z, t in zip(Z, tun) if not t])
        t_idx = ValueTracker(1)
        trail = always_redraw(lambda: VMobject(color=MEAN, stroke_width=5).set_points_as_corners([M(E[k, :2]) for k in range(max(2, int(t_idx.get_value())))]))
        circ = always_redraw(lambda: Circle(radius=max(0.06, 2 * E[int(t_idx.get_value()) - 1, 4] * scale), color=MEAN, stroke_width=2).set_fill(MEAN, 0.18).move_to(M(E[int(t_idx.get_value()) - 1, :2])))
        self.add(trail, circ)
        idx_ok = [k for k in range(T) if not tun[k]]
        self.mark("drive")
        self.play(t_idx.animate.set_value(k0), LaggedStart(*[FadeIn(gps[i]) for i in range(len([k for k in idx_ok if k < k0]))], lag_ratio=0.5), run_time=4.0, rate_func=linear)
        self.wait(0.2)
        # ---------- 3. tunnel ----------
        tb = VMobject(stroke_color=LINE, stroke_width=50, stroke_opacity=0.9).set_points_smoothly([M(p) for p in pos[k0 - 1:k0 + 13]]).set_z_index(-1)
        tl = Text("tunnel", font=FONT, weight=BOLD, color=INK).scale(0.4).move_to(M(pos[k0 + 6]) + [0, 0.55, 0])
        cap = self.swap(cap, self.caption("No GPS in the tunnel: the doubt grows", "it keeps going at its last speed, and it knows it"))
        self.play(FadeIn(tb), FadeIn(tl), run_time=0.6)
        self.mark("tunnel"); self.play(t_idx.animate.set_value(k0 + 12), run_time=3.0, rate_func=linear)
        n0 = len([k for k in idx_ok if k < k0])
        self.mark("exit"); self.play(t_idx.animate.set_value(T), LaggedStart(*[FadeIn(gps[i]) for i in range(n0, len(gps))], lag_ratio=0.5), run_time=2.0, rate_func=linear)
        trail.clear_updaters(); circ.clear_updaters()
        eg = np.mean(np.hypot(*(Z - pos)[~tun].T)); ek = np.mean(np.hypot(*(E[:, :2] - pos)[~tun].T))
        res = VGroup(Text(f"GPS average error: {eg:.1f} m", font=FONT, color=POINT).scale(0.38), Text(f"filter average error: {ek:.1f} m", font=FONT, weight=SEMIBOLD, color=MEAN).scale(0.42)).arrange(DOWN, buff=0.12, aligned_edge=LEFT).to_corner(DR, buff=0.45)
        self.mark("result"); self.play(FadeIn(res), run_time=0.6); self.wait(1.6)
        self.play(FadeOut(VGroup(truth, gps, trail, circ, tb, tl, res)), run_time=0.6)

        # ---------- 4. speed ----------
        cap = self.swap(cap, self.caption("It estimates what nobody measures: the speed", "it stops at a traffic light and pulls away again"))
        r2 = np.random.default_rng(17); TT = 90
        V = np.array([14 - (k - 30) * 1.4 if 30 <= k < 40 else 0 if 40 <= k < 52 else (k - 52) * 14 / 12 if 52 <= k < 64 else 14 for k in range(TT)], float)
        Xp = np.cumsum(V); Zp = Xp + 5 * r2.standard_normal(TT)
        Ev = kf_run([(z, 0.0) for z in Zp], 0.4, 25.0, (14, 0))[:, 2]
        naive = np.r_[V[0], np.diff(Zp)]
        P = lambda k, v: np.array([-6.0 + k * 12.0 / TT, -3.2 + (v * 3.6 + 30) * 0.034, 0])
        axis = Line(P(0, 0), P(TT, 0), color=LINE, stroke_width=2)
        yl = VGroup(*[Text(f"{v} km/h", font=FONT, color=MUTED).scale(0.26).next_to(P(0, v / 3.6), LEFT, buff=0.1) for v in (0, 50)])
        nv = VMobject(color=POINT, stroke_width=2, stroke_opacity=0.85).set_points_as_corners([P(k, np.clip(v, -8, 30)) for k, v in enumerate(naive)])
        tv = DashedVMobject(VMobject(color=INK, stroke_width=3).set_points_as_corners([P(k, v) for k, v in enumerate(V)]), num_dashes=60)
        kv = VMobject(color=MEAN, stroke_width=6).set_points_as_corners([P(k, v) for k, v in enumerate(Ev)])
        self.mark("vel"); self.play(Create(axis), FadeIn(yl), Create(tv), run_time=0.9)
        l1 = Text("subtracting GPS positions", font=FONT, color=POINT).scale(0.36).move_to([3.6, 1.55, 0])
        self.mark("naive"); self.play(Create(nv), FadeIn(l1), run_time=1.6)
        l2 = Text("Kalman filter", font=FONT, weight=SEMIBOLD, color=MEAN).scale(0.42).move_to([3.6, 1.1, 0])
        self.mark("smooth"); self.play(Create(kv), FadeIn(l2), run_time=1.6); self.wait(1.6)
        self.play(FadeOut(VGroup(axis, yl, nv, tv, kv, l1, l2)), run_time=0.6)

        # ---------- Closing ----------
        self.mark("legend")
        self.play(FadeOut(cap), run_time=0.6)
        e1 = Text("It predicts with what it knows, corrects with what it measures.", font=FONT, weight=BOLD, color=INK).scale(0.58)
        e2 = Text("And at every moment it knows how far off it might be.", font=FONT, color=MUTED).scale(0.48)
        VGroup(e1, e2).arrange(DOWN, buff=0.4)
        self.mark("end_text"); self.play(Write(e1), run_time=1.4); self.play(FadeIn(e2, shift=0.2 * UP), run_time=0.8)
        self.wait(2.0)
        self.mark("fade"); self.play(FadeOut(VGroup(e1, e2)), run_time=1)
        self.marks["total"] = [round(self.renderer.time, 3)]
        json.dump(self.marks, open("marks.json", "w"), indent=1)
