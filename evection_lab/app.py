import json
import os
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons, RadioButtons

from . import secular, nbody, rhea, stellar, theory

BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
INK, MUTED, BAND = "#14181f", "#6c7685", "#f6e6d6"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def load_rhea_table():
    path = os.path.join(DATA_DIR, "rhea_table.json")
    if os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    return rhea.table()


def interpolate_table(tab, f):
    fs = np.array([r["f"] for r in tab])
    f = float(np.clip(f, fs[0], fs[-1]))
    out = {"f": f}
    for key in ("Mx", "n_m", "gJ2", "gExt", "fJ2", "a_km"):
        if key in tab[0]:
            out[key] = float(np.interp(f, fs, [r[key] for r in tab]))
    if "a_km" not in out:
        out["a_km"] = 527108.0 * f
    out["alpha"] = out["a_km"] / (rhea.SATURN_A_AU * rhea.AU_KM / out["Mx"])
    out["theta_deg"] = 26.73
    return out


class EvectionLab:
    MODES = ("Sweep", "N-body", "Rhea")

    def __init__(self, show=True):
        self.fig = plt.figure(figsize=(15.5, 9.2))
        try:
            self.fig.canvas.manager.set_window_title("Evection Sweep Lab")
        except Exception:
            pass
        self.fig.text(0.015, 0.965, "Evection Sweep Lab", fontsize=17, fontweight="bold", color=INK)
        self.fig.text(0.015, 0.94, "stellar mass loss sweeps the evection resonance through satellite systems",
                      fontsize=9, color=MUTED)
        self.mode = "Sweep"
        self.tracks = {}
        self.table = None
        self.last = None
        self.positions = [[0.32, 0.56, 0.31, 0.35], [0.68, 0.56, 0.30, 0.35], [0.32, 0.08, 0.31, 0.38],
                          [0.68, 0.08, 0.30, 0.38]]
        self.ax = [self.fig.add_axes(p) for p in self.positions]
        self.ax[3].axis("off")
        self.status = self.fig.text(0.015, 0.012, "", fontsize=9, color=MUTED)
        rax = self.fig.add_axes([0.015, 0.80, 0.12, 0.12])
        self.radio = RadioButtons(rax, self.MODES, active=0)
        self.radio.on_clicked(self.set_mode)
        self.widgets = {m: [] for m in self.MODES}
        self.sliders = {}
        self._build_sweep()
        self._build_nbody()
        self._build_rhea()
        bax = self.fig.add_axes([0.015, 0.06, 0.1, 0.045])
        self.run_button = Button(bax, "Run", color="#f0d8bf", hovercolor="#e8c49e")
        self.run_button.on_clicked(lambda _: self.run())
        sax = self.fig.add_axes([0.13, 0.06, 0.1, 0.045])
        self.save_button = Button(sax, "Save PNG")
        self.save_button.on_clicked(lambda _: self.save())
        self.set_mode("Sweep", run=False)
        self.run()
        if show:
            plt.show()

    def _slider(self, mode, slot, key, label, vmin, vmax, init, fmt="%.3g", step=None):
        a = self.fig.add_axes([0.105, 0.735 - slot * 0.043, 0.1, 0.022])
        s = Slider(a, label, vmin, vmax, valinit=init, valfmt=fmt, valstep=step, color="#e8c49e")
        s.label.set_fontsize(8.5)
        s.valtext.set_fontsize(8.5)
        self.sliders[(mode, key)] = s
        self.widgets[mode].append(a)
        return s

    def _checks(self, mode, labels, states, y):
        a = self.fig.add_axes([0.015, y, 0.2, 0.03 * len(labels) + 0.01])
        c = CheckButtons(a, labels, states)
        for t in c.labels:
            t.set_fontsize(8.5)
        self.widgets[mode].append(a)
        return c

    def _build_sweep(self):
        m = "Sweep"
        self._slider(m, 0, "eps", "log10 n_p/n_m", -4.0, -1.5, -2.52)
        self._slider(m, 1, "lam", "log10 Lambda", -1.0, 2.5, 0.9)
        self._slider(m, 2, "e0", "log10 e0/e*", -4.0, 0.5, -1.5)
        self._slider(m, 3, "theta", "obliquity (deg)", 0.0, 85.0, 0.0, "%.1f")
        self._slider(m, 4, "fJ2", "f_J2", 0.3, 1.0, 1.0, "%.2f")
        self._slider(m, 5, "alpha", "log10 a_m/a_p", -5.0, -2.0, -3.48)
        self._slider(m, 6, "n", "members", 1, 96, 32, "%d", 1)
        pd = self._slider(m, 7, "pd", "portrait delta/c", -6.0, 6.0, -1.5, "%.2f")
        pd.on_changed(lambda _: self._draw_portrait())
        self.sweep_checks = self._checks(m, ["circular start", "octupole tide", "capture direction"],
                                         [False, False, False], 0.25)

    def _build_nbody(self):
        m = "N-body"
        self._slider(m, 0, "eps", "log10 n_p/n_m", -2.7, -1.5, -2.0)
        self._slider(m, 1, "lam", "log10 Lambda", -0.6, 1.5, 0.0)
        self._slider(m, 2, "e0r", "e0/e*", 0.0, 1.0, 0.05, "%.2f")
        self._slider(m, 3, "phi0", "phi0 (deg)", 0.0, 360.0, 57.0, "%.0f")
        self._slider(m, 4, "rr", "R_p/a_m", 0.2, 0.6, 1.0 / 3.0, "%.2f")
        self._slider(m, 5, "mp", "log10 m_p/M", -4.0, -2.0, -3.0, "%.2f")
        self._slider(m, 6, "steps", "steps/orbit", 100, 400, 200, "%d", 10)
        self.nbody_checks = self._checks(m, ["quadrupole tide only"], [False], 0.32)

    def _build_rhea(self):
        m = "Rhea"
        self._slider(m, 0, "theta", "Saturn obliquity", 0.0, 75.0, 26.73, "%.1f")
        self._slider(m, 1, "f", "Rhea expansion f", 1.0, 1.45, 1.0, "%.3f")
        self._slider(m, 2, "e0", "free e0", 0.0, 0.003, 0.001, "%.5f")
        self._slider(m, 3, "kq", "log10 k2/Q", -5.0, -2.0, -4.0, "%.2f")
        a = self.fig.add_axes([0.015, 0.40, 0.2, 0.12])
        self.eta_radio = RadioButtons(a, ("eta = 0.300", "eta = 0.477", "eta = 0.700"), active=1)
        self.widgets[m].append(a)
        self.rhea_checks = self._checks(m, ["octupole tide"], [True], 0.33)

    def value(self, mode, key):
        return self.sliders[(mode, key)].val

    def set_mode(self, label, run=True):
        self.mode = label
        for m, axs in self.widgets.items():
            for a in axs:
                a.set_visible(m == label)
        if run:
            self.run()
        else:
            self.fig.canvas.draw_idle()

    def note(self, text):
        self.status.set_text(text)
        self.fig.canvas.draw_idle()
        try:
            self.fig.canvas.flush_events()
        except Exception:
            pass

    def reset_axis(self, i):
        a = self.ax[i]
        a.clear()
        a.set_aspect("auto")
        a.set_position(self.positions[i])
        return a

    def run(self):
        for i in range(4):
            self.reset_axis(i)
        self.ax[3].axis("off")
        t0 = time.time()
        if self.mode == "Sweep":
            self.run_sweep()
        elif self.mode == "N-body":
            self.run_nbody()
        else:
            self.run_rhea()
        self.note(f"{self.mode} finished in {time.time() - t0:.1f} s")

    def stats_text(self, lines):
        self.ax[3].text(0.0, 1.0, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=9.5,
                        transform=self.ax[3].transAxes, color=INK)

    def run_sweep(self):
        m = "Sweep"
        eps = 10 ** self.value(m, "eps")
        lam = 10 ** self.value(m, "lam")
        circ, octu, capture = self.sweep_checks.get_status()
        e0r = 0.0 if circ else 10 ** self.value(m, "e0")
        theta = np.radians(self.value(m, "theta"))
        fJ2 = self.value(m, "fJ2")
        alpha = 10 ** self.value(m, "alpha") if octu else 0.0
        n = int(self.value(m, "n"))
        es = float(theory.e_star(eps, theta, fJ2))
        c = float(theory.c_width(eps, theta))
        phis = secular.phases(n)
        self.note("integrating the orbit-averaged equations ...")
        r = secular.sweep(eps, lam, e0r * es, phis, theta=theta, fJ2=fJ2, K0=secular.window_start(e0r), K1=12.0,
                          n_out=2500, alpha=alpha, direction=-1 if capture else 1)
        self.last = {"kind": "sweep", "r": r, "eps": eps, "theta": theta, "fJ2": fJ2, "alpha": alpha, "es": es, "c": c}
        ef = r.ef / es
        a = self.ax[0]
        a.axvspan(-1, 1, color=BAND, zorder=0)
        for j in range(min(10, n)):
            a.plot(r.delta_over_c, r.e[j] / es, color=BLUE, lw=1.5 if j == 0 else 0.8, alpha=1 if j == 0 else 0.4)
        a.axhline(1, color=MUTED, lw=0.8)
        a.set_xlim(max(r.delta_over_c.min(), -7), min(r.delta_over_c.max(), 12))
        a.set_xlabel("detuning delta / c")
        a.set_ylabel("e / e*")
        a.set_title("eccentricity through the crossing", loc="left", fontsize=10)
        self._draw_portrait(draw=False)
        b = self.ax[2]
        b.plot(np.degrees(phis), ef, "o", color=BLUE, ms=5)
        lin = theory.linear_amplification(lam)
        adi = float(theory.adiabatic_map(e0r * es, es)[0] / es) if e0r > 0 else 1.0
        if e0r > 0 and not capture:
            b.axhline(adi, color=ORANGE, lw=1.2, label="adiabatic jump map")
        if e0r > 0 and lin["hi"] * e0r < 4:
            b.axhline(lin["hi"] * e0r, color=MUTED, lw=1.0, ls="--", label="Landau-Zener maximum")
        b.set_xlim(0, 360)
        b.set_xlabel("initial phase phi0 (deg)")
        b.set_ylabel("e_f / e*")
        b.set_title("outcome by initial phase", loc="left", fontsize=10)
        if b.get_legend_handles_labels()[0]:
            b.legend(fontsize=8, frameon=False)
        seed = theory.octupole_seed(alpha, lam, theta) / es if alpha else 0.0
        eff = max(e0r, seed)
        if capture:
            regime = "capture direction"
        elif eff == 0:
            regime = "no seed: stays circular"
        elif eff * np.exp(lam) < 0.3:
            regime = "linear (Landau-Zener)"
        elif lam > 2 * np.log(1 / max(eff, 1e-12)) + 4:
            regime = "adiabatic"
        else:
            regime = "transitional"
        lz = e0r * lin["rms"]
        self.stats_text([
            "ORBIT-AVERAGED SWEEP",
            "",
            f"n_p/n_m            {eps:.3e}",
            f"Lambda             {lam:.3g}",
            f"band half-width c  {c:.5f}",
            f"universal e*       {es:.4f}",
            f"initial e0         {e0r * es:.5f}  ({e0r:.3g} e*)",
            "",
            f"simulated mean     {ef.mean():.3f} e*",
            f"simulated rms      {np.sqrt(np.mean(ef ** 2)):.3f} e*",
            f"5-95% range        {np.percentile(ef, 5):.3f} - {np.percentile(ef, 95):.3f} e*",
            "",
            f"Landau-Zener rms   {('%.3f e*' % lz) if 0 < lz < 0.5 else ('saturated' if lz else '-')}",
            f"adiabatic map      {adi:.3f} e*" if not capture else "adiabatic map      n/a (capture)",
            f"octupole seed      {seed:.4f} e*" if alpha else "octupole seed      off",
            "",
            f"regime             {regime}",
        ])

    def _draw_portrait(self, draw=True):
        if not self.last or self.last.get("kind") != "sweep":
            return
        L = self.last
        r, eps, theta, fJ2, es, c = L["r"], L["eps"], L["theta"], L["fJ2"], L["es"], L["c"]
        a = self.reset_axis(1)
        dc = self.value("Sweep", "pd")
        delta = dc * c
        ft = theory.f_theta(theta)
        g_tot = 1 - 0.75 * eps * ft
        npl = g_tot / (1 + delta - 0.75 * eps * ft)
        w_j2, w_ext, eps_now = fJ2 * g_tot / npl, (1 - fJ2) * g_tot / npl, eps * npl
        F = 0.9375 * eps_now * L["alpha"] * np.sqrt(npl) * theory.f_oct(theta)
        lim = min(max(2.2, 1.1 * float(np.max(r.e / es))), 0.95 / es)
        u = np.linspace(-lim, lim, 241)
        U, V = np.meshgrid(u, u)
        E = np.hypot(U, V) * es
        P = np.arctan2(V, U)
        Z = np.where(E < 0.99, theory.hamiltonian(np.minimum(E, 0.98), P, w_j2, w_ext, eps_now, theta, F), np.nan)
        zf = Z[np.isfinite(Z)]
        levels = np.unique(np.quantile(zf, np.linspace(0.02, 0.98, 22) ** 1.6))
        a.contour(U, V, Z, levels=levels, colors=BLUE, linewidths=0.6, alpha=0.5)
        row = theory.hamiltonian(np.minimum(np.abs(u) * es, 0.98), np.where(u >= 0, 0, np.pi), w_j2, w_ext, eps_now, theta, F)
        special = [float(theory.hamiltonian(0.0, 0.0, w_j2, w_ext, eps_now, theta, F))]
        for i in range(1, len(u) - 1):
            if abs(u[i]) > 0.02 * lim and (row[i] - row[i - 1]) * (row[i + 1] - row[i]) < 0:
                special.append(float(row[i]))
        a.contour(U, V, Z, levels=sorted(set(np.round(special, 13)))[:4], colors=ORANGE, linewidths=1.4)
        sel = np.abs(r.delta_over_c - dc) < 0.6
        a.plot(r.e[0][sel] / es * np.cos(r.phi[0][sel]), r.e[0][sel] / es * np.sin(r.phi[0][sel]), ".", color=INK, ms=2.5)
        a.set_aspect("equal")
        a.set_xlim(-lim, lim)
        a.set_ylim(-lim, lim)
        a.set_xlabel("e cos(phi) / e*")
        a.set_ylabel("e sin(phi) / e*")
        a.set_title(f"phase portrait at delta/c = {dc:+.2f}", loc="left", fontsize=10)
        if draw:
            self.fig.canvas.draw_idle()

    def run_nbody(self):
        m = "N-body"
        eps = 10 ** self.value(m, "eps")
        lam = 10 ** self.value(m, "lam")
        e0r = self.value(m, "e0r")
        phi0 = np.radians(self.value(m, "phi0"))
        rr = self.value(m, "rr")
        mp = 10 ** self.value(m, "mp")
        steps = int(self.value(m, "steps"))
        quad = self.nbody_checks.get_status()[0]
        es = float(theory.e_star(eps))
        K0 = max(6.0, (1 + 1.5 * e0r) ** 2 + 5)
        sec_est, n_steps = nbody.estimated_seconds(eps, lam, steps, e0r, rr, mp)
        self.note(f"integrating {n_steps:.2e} steps (about {sec_est:.0f} s) ...")
        t0 = time.time()
        o = nbody.run(eps, lam, e0=e0r * es, phi0=phi0, r_ratio=rr, mp=mp, steps=steps, K0=K0, K1=9.0, quad_only=quad,
                      progress=lambda f: self.note(f"N-body {100 * f:.0f}% ..."))
        ms = time.time() - t0
        s = secular.sweep(eps, lam, e0r * es, [phi0], K0=K0, K1=9.0, n_out=2000, alpha=0.0 if quad else o.setup.alpha,
                          proper=False)
        self.last = {"kind": "nbody"}
        a = self.ax[0]
        a.axvspan(-1, 1, color=BAND, zorder=0)
        a.plot(o.delta_over_c, o.e / es, ".", color=BLUE, ms=1.5, label="direct N-body")
        a.plot(s.delta_over_c, s.e[0] / es, color=ORANGE, lw=1.6, label="secular theory")
        a.axhline(1, color=MUTED, lw=0.8)
        a.set_xlabel("detuning delta / c")
        a.set_ylabel("e / e*")
        a.set_title("N-body against the averaged theory", loc="left", fontsize=10)
        a.legend(fontsize=8, frameon=False, loc="upper left")
        b = self.ax[1]
        b.plot(o.delta_over_c, np.degrees(o.phi), ".", color=BLUE, ms=1.2, label="N-body")
        b.plot(s.delta_over_c, np.degrees(s.phi[0]), ".", color=ORANGE, ms=0.8, alpha=0.6, label="secular")
        b.set_ylim(-180, 180)
        b.set_xlabel("delta / c")
        b.set_ylabel("phi (deg)")
        b.set_title("resonant angle", loc="left", fontsize=10)
        b.legend(fontsize=8, frameon=False)
        inv = o.invariant()
        c = self.ax[2]
        c.plot(o.delta_over_c, inv / inv[0] - 1, color=AQUA, lw=1.5)
        c.set_xlabel("delta / c")
        c.set_ylabel("a_p (M + m_p) / initial - 1")
        c.set_title("adiabatic invariant of the planet's orbit", loc="left", fontsize=10)
        dev = o.ef / s.ef[0] - 1
        self.stats_text([
            "DIRECT THREE-BODY INTEGRATION",
            "",
            f"backend            {'numba' if nbody.HAVE_NUMBA else 'pure Python'}",
            f"steps              {o.n_steps:.3e}  ({ms:.1f} s)",
            f"derived J2         {o.setup.J2:.4f}",
            f"a_m / a_p          {o.setup.alpha:.3e}",
            f"tide               {'quadrupole only' if quad else 'full'}",
            "",
            f"N-body e_f         {o.ef / es:.4f} e*",
            f"secular e_f        {s.ef[0] / es:.4f} e*",
            f"difference         {100 * dev:.1f} %",
            f"invariant drift    {np.max(np.abs(inv / inv[0] - 1)):.2e}",
        ])

    def run_rhea(self):
        m = "Rhea"
        if self.table is None:
            self.note("loading Laplace-Lagrange table for Rhea ...")
            self.table = load_rhea_table()
        eta = (0.3, 0.477, 0.7)[["eta = 0.300", "eta = 0.477", "eta = 0.700"].index(self.eta_radio.value_selected)]
        if eta not in self.tracks:
            self.tracks[eta] = stellar.evolve(eta=eta)
        tr = self.tracks[eta]
        theta = self.value(m, "theta")
        f = self.value(m, "f")
        e0 = self.value(m, "e0")
        kq = 10 ** self.value(m, "kq")
        octu = self.rhea_checks.get_status()[0]
        p = interpolate_table(self.table, f)
        self.note("integrating Rhea's crossing ...")
        out = rhea.simulate(eta, theta, f, e0, 64, octu, params=p, track=tr)
        self.last = {"kind": "rhea"}
        a = self.ax[0]
        a.plot(tr["t"] / 1e6, tr["M"], color=BLUE, lw=1.8)
        a.set_xlabel("time since the base of the red-giant branch (Myr)")
        a.set_ylabel("solar mass (M_sun)")
        a.set_title("the Sun's mass-loss history", loc="left", fontsize=10)
        if not out["crosses"]:
            self.stats_text(["RHEA", "", "no crossing: Rhea's precession period", "exceeds Saturn's final year"])
            return
        loc, r, st = out["local"], out["result"], out["stats"]
        a.plot(loc["t"] / 1e6, p["Mx"], "o", ms=9, mfc="white", mec=ORANGE, mew=2)
        a.annotate(f"crossing at {p['Mx']:.3f} M_sun", (loc["t"] / 1e6, p["Mx"]), xytext=(-150, -20),
                   textcoords="offset points", fontsize=8.5)
        b = self.ax[1]
        for j, col in enumerate((BLUE, ORANGE, AQUA, VIOLET)):
            if j < r.e.shape[0]:
                b.plot(r.t / 1e3, r.e[j * max(1, r.e.shape[0] // 4)], color=col, lw=0.9)
        b.axhline(r.e_star, color=MUTED, lw=0.8)
        b.set_xlabel("time from exact resonance (kyr)")
        b.set_ylabel("Rhea's eccentricity")
        b.set_title("Rhea through the crossing", loc="left", fontsize=10)
        c = self.ax[2]
        c.hist(r.ef, bins=16, color=BLUE, rwidth=0.9)
        c.set_xlabel("final eccentricity")
        c.set_ylabel("initial phases")
        c.set_title("distribution over 64 initial phases" if e0 > 0 else "circular start", loc="left", fontsize=10)
        heat = rhea.tidal_heating(st["mean"], kq, f)
        tau = rhea.damping_time_yr(kq, f)
        self.stats_text([
            "RHEA AND THE DYING SUN",
            "",
            f"Sun's mass at crossing   {p['Mx']:.4f} M_sun",
            f"stellar phase            {stellar.PHASE_NAMES.get(loc['phase'], '-')}",
            f"luminosity               {loc['L']:.0f} L_sun",
            f"Saturn's year then       {29.457 / p['Mx'] ** 2:.2f} yr",
            f"mass-loss time t_M       {loc['tM']:.3e} yr",
            f"adiabaticity Lambda      {r.lam:.2f}",
            f"universal e*             {r.e_star:.4f}  (f_J2 = {r.fJ2:.3f})",
            "",
            f"Rhea's final e           {st['mean']:.4f}",
            f"5-95% range              {st['p05']:.4f} - {st['p95']:.4f}",
            f"present e                {rhea.RHEA_E_NOW}",
            "",
            f"tidal heating            {heat / 1e9:.3g} GW  (k2/Q = {kq:.1e})",
            f"damping time             {tau / 1e6:.3g} Myr",
        ])

    def save(self, path=None):
        path = path or f"evection_lab_{self.mode.lower().replace('-', '')}.png"
        self.fig.savefig(path, dpi=150)
        self.note(f"saved {path}")
        return path


def main():
    EvectionLab(show=True)


if __name__ == "__main__":
    main()
