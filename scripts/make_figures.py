import json
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from evection_lab.theory import e_star, adiabatic_map, adiabaticity
from evection_lab.stellar import evolve, local_mass_loss
from evection_lab.rhea import expansion_from_Q

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
FIGS = os.path.join(HERE, "..", "figures")

C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
C5, C6, C7, C8 = "#e87ba4", "#008300", "#4a3aa7", "#e34948"
INK, INK2, MUTED, GRID = "#14181f", "#4a5462", "#7a8492", "#e3e7ec"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "cm",
    "font.size": 9, "axes.labelsize": 9.5, "axes.titlesize": 9.5, "legend.fontsize": 8, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "axes.edgecolor": "#9aa3ad", "axes.linewidth": 0.7, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK, "text.color": INK,
    "legend.frameon": False, "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight", "lines.linewidth": 1.5,
})
_cache = {}


def D(name):
    if name not in _cache:
        with open(os.path.join(DATA, name)) as fh:
            _cache[name] = json.load(fh)
    return _cache[name]


def save(fig, name):
    os.makedirs(FIGS, exist_ok=True)
    fig.savefig(os.path.join(FIGS, f"{name}.pdf"))
    fig.savefig(os.path.join(FIGS, f"{name}.png"))
    plt.close(fig)
    print("saved", name)


def fig1():
    cross = D("solar_system_crossings.json")
    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    M = np.linspace(1.0, 0.53, 300)
    tr = evolve(eta=0.477)
    Mtip = tr["M"][tr["phase"] == 1][-1]
    Mwd = tr["M"][-1]
    cols = {"Jupiter": C1, "Saturn": C2, "Uranus": C3, "Neptune": C7}
    P0 = {"Jupiter": 11.862, "Saturn": 29.457, "Uranus": 84.011, "Neptune": 164.79}
    ax.axvspan(Mwd, 0.53, color="#f1f3f6", zorder=0)
    ax.axvline(Mtip, color=MUTED, lw=0.8)
    ax.text(Mtip - 0.004, 6.2, "RGB tip", color=MUTED, fontsize=7.5, rotation=90, va="bottom", ha="right")
    ax.text(Mwd - 0.004, 6.2, "white dwarf", color=MUTED, fontsize=7.5, rotation=90, va="bottom", ha="right")
    labels = []
    for p, col in cols.items():
        ax.plot(M, P0[p] / M ** 2, color=col, lw=2, zorder=3)
        ax.text(0.995, P0[p] * 0.93, f"{p}: $P_{{\\rm orb}}\\propto M_\\ast^{{-2}}$", color=col, fontsize=7.5, ha="left", va="top")
        for r in cross:
            if r["planet"] != p or not np.isfinite(r["P_varpi_yr"]) or r["P_varpi_yr"] > 700 or r["P_varpi_yr"] < 5:
                continue
            y = r["P_varpi_yr"]
            ax.plot([1.0, 0.53], [y, y], color=col, lw=0.8, alpha=0.55, zorder=2)
            labels.append([y, r["moon"] + (" (MMR)" if r["mmr"] else ""), col])
            if r["crosses"]:
                big = r["moon"] == "Rhea"
                ax.plot(r["M_cross"], y, "o", ms=8 if big else 5.5, mfc=col if big else "white", mec=col, mew=1.6, zorder=5)
    labels.sort(key=lambda t: t[0])
    ys = [t[0] for t in labels]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] * 1.17)
    for (y0, name, col), y in zip(labels, ys):
        ax.text(0.527, y, name, color=col, fontsize=7, ha="left", va="center", clip_on=False)
    ax.annotate("Rhea crosses at $M_\\ast=0.907\\,M_\\odot$\n(upper RGB, $\\Lambda\\approx 9$–$19$)", xy=(0.9066, 35.84),
                xytext=(0.875, 64), fontsize=8, color=INK, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    ax.set_yscale("log")
    ax.set_xlim(1.0, 0.53)
    ax.set_ylim(6, 720)
    ax.set_xlabel(r"solar mass $M_\ast/M_\odot$  (time runs to the right)")
    ax.set_ylabel("period (yr)")
    fig.legend(handles=[Line2D([], [], color=INK2, lw=2, label="planet orbital period"),
                        Line2D([], [], color=INK2, lw=0.8, alpha=0.6, label="moon free apsidal period"),
                        Line2D([], [], ls="", marker="o", mfc="white", mec=INK2, label="evection crossing")],
               loc="lower center", ncol=3, bbox_to_anchor=(0.45, -0.06))
    save(fig, "fig1_sweep")


def k_exact(e, phi, delta, eps):
    w = 1 + delta - 0.75 * eps
    s = np.sqrt(1 - e ** 2)
    return w / 3 / s ** 3 + s + 0.25 * eps * ((1 + 1.5 * e ** 2) + 7.5 * e ** 2 * np.cos(2 * phi))


def fig2():
    ts = D("timeseries.json")
    eps, es, c = ts["eps"], ts["eStar"], ts["c"]
    run = [r for r in ts["runs"] if r["Lam"] == 128][0]
    fig, axs = plt.subplots(1, 4, figsize=(7.0, 2.05), sharey=True)
    L = 2.3
    u = np.linspace(-L, L, 401)
    U, V = np.meshgrid(u, u)
    for ax, dc in zip(axs, [-3.0, -1.15, 0.0, 0.85]):
        Z = k_exact(np.hypot(U, V) * es, np.arctan2(V, U), dc * c, eps)
        ax.contour(U, V, Z, levels=np.unique(np.quantile(Z, np.linspace(0.02, 0.98, 24) ** 1.7)), colors=C1,
                   linewidths=0.55, alpha=0.55)
        zrow = k_exact(np.abs(u) * es, np.where(u >= 0, 0, np.pi), dc * c, eps)
        spec = [k_exact(0.0, 0.0, dc * c, eps)]
        for i in range(1, len(u) - 1):
            if abs(u[i]) > 0.03 and (zrow[i] - zrow[i - 1]) * (zrow[i + 1] - zrow[i]) < 0:
                spec.append(zrow[i])
        ax.contour(U, V, Z, levels=sorted(set(np.round(spec, 14))), colors=C2, linewidths=1.3)
        sel = np.abs(np.array(run["dc"]) - dc) < 0.5
        ax.plot(np.array(run["h"][3])[sel], np.array(run["k"][3])[sel], ".", color=INK, ms=2.2)
        ax.set_title(f"$\\delta/c={dc:+.2f}$")
        ax.set_aspect("equal")
        ax.set_xlim(-L, L)
        ax.set_ylim(-L, L)
        ax.set_xlabel(r"$e\cos\phi/e_\star$")
        ax.grid(False)
    axs[0].set_ylabel(r"$e\sin\phi/e_\star$")
    fig.legend(handles=[Line2D([], [], color=C1, lw=0.8, alpha=0.6, label="level curves of $K$"),
                        Line2D([], [], color=C2, lw=1.3, label="separatrices"),
                        Line2D([], [], ls="", marker=".", color=INK, label="trajectory samples ($\\Lambda=128$) near that $\\delta$")],
               loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.12))
    save(fig, "fig2_portraits")


def fig3():
    ts = D("timeseries.json")
    fig, axs = plt.subplots(2, 2, figsize=(6.8, 4.6))
    titles = {(1, 1): "(a) linear regime, $\\Lambda=1$", (8, 1): "(b) transitional, $\\Lambda=8$",
              (128, 1): "(c) adiabatic, $\\Lambda=128$", (64, -1): "(d) capture direction, $\\Lambda=64$"}
    for ax, r in zip(axs.flat, ts["runs"]):
        x = np.array(r["dc"])
        col = C1 if r["dir"] == 1 else C2
        for e in r["e"]:
            ax.plot(x, np.maximum(e, 1e-4), color=col, lw=0.6, alpha=0.35)
        ax.plot(x, np.maximum(r["e"][0], 1e-4), color=col, lw=1.4)
        ax.axvspan(-1, 1, color="#f6e6d6", zorder=0)
        ax.axhline(1, color=INK2, lw=0.8)
        ax.set_yscale("log")
        ax.set_ylim(3e-3, 5)
        ax.set_title(titles[(r["Lam"], r["dir"])], loc="left")
        ax.set_xlabel(r"detuning $\delta/c$")
        ax.set_ylabel(r"$e/e_\star$")
        ax.text(0.98, 0.06, f"mean $e_f/e_\\star={np.mean(r['ef']):.3f}$", transform=ax.transAxes, ha="right", fontsize=8)
        if r["dir"] == -1:
            ax.invert_xaxis()
    fig.tight_layout()
    save(fig, "fig3_timeseries")


def fig4():
    G = D("secular_campaign.json")["grid"]
    L, R = np.array(G["Lams"]), np.array(G["e0rs"])
    mean, p05, p95 = np.array(G["mean"]), np.array(G["p05"]), np.array(G["p95"])
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.9, 2.9))
    for r0, col in zip([1e-4, 1e-3, 1e-2, 0.1, 0.3, 1.0, 2.0], [C1, C2, C3, C4, C5, C6, C7]):
        j = int(np.argmin(np.abs(R - r0)))
        a.plot(L, mean[:, j], "o-", color=col, ms=3, lw=1.3, label=f"$e_0/e_\\star={r0:g}$")
        Ls = np.logspace(-1, 2.5, 200)
        lin = r0 * np.sqrt(2 * np.exp(2 * Ls) - 1)
        m = lin < 0.6
        a.plot(Ls[m], lin[m], "--", color=col, lw=0.9)
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_ylim(1e-4, 6)
    a.set_xlim(0.11, 300)
    a.set_xlabel(r"adiabaticity $\Lambda$")
    a.set_ylabel(r"$\langle e_f\rangle/e_\star$")
    a.axhline(1, color=INK2, lw=0.8)
    a.set_title("(a) universal transfer function", loc="left")
    a.legend(fontsize=6.8, loc="lower right", handlelength=1.4)
    a.text(14, 1.6e-4, "dashed: Landau–Zener\n$e_0\\sqrt{2e^{2\\Lambda}-1}$", fontsize=7, color=INK2)
    i = int(np.argmax(L))
    e0g = np.logspace(-3, np.log10(3.2), 200)
    b.fill_between(R, p05[i], p95[i], color=C1, alpha=0.18, lw=0, label=f"simulated 5–95% ($\\Lambda={L[i]:g}$)")
    b.plot(R, mean[i], "o", color=C1, ms=3.5, label="simulated mean")
    b.plot(e0g, adiabatic_map(e0g, 1.0), color=C2, lw=1.6, label="adiabatic jump map (theory)")
    b.plot(e0g, e0g + 2 * np.sqrt(2) / np.pi, ":", color=INK2, lw=1.1, label=r"$e_0+(2\sqrt{2}/\pi)\,e_\star$")
    b.plot(e0g, e0g, color=MUTED, lw=0.8, label="no change")
    b.set_xscale("log")
    b.set_xlim(8e-4, 3.5)
    b.set_ylim(0, 4.3)
    b.set_xlabel(r"$e_0/e_\star$")
    b.set_ylabel(r"$e_f/e_\star$")
    b.set_title("(b) adiabatic limit", loc="left")
    b.legend(fontsize=6.8, loc="upper left")
    fig.tight_layout()
    save(fig, "fig4_transfer")


def fig5():
    nbv = D("nbody_validation.json")
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.6))
    c_ = nbv["c"]
    a = axs[0]
    a.plot(c_["nb"]["dOverC"], c_["nb"]["e"], ".", color=C1, ms=1.6, label="direct N-body")
    a.plot(c_["sec"]["dOverC"], c_["sec"]["e"], color=C2, lw=1.4, label="secular (quad. + oct.)")
    a.axvspan(-1, 1, color="#f6e6d6", zorder=0)
    a.axhline(1, color=INK2, lw=0.8)
    a.set_xlim(-6, 9)
    a.set_xlabel(r"$\delta/c$")
    a.set_ylabel(r"$e/e_\star$")
    a.set_title(f"(a) one trajectory, $\\varepsilon={c_['eps']}$, $\\Lambda={c_['Lam']}$", loc="left", fontsize=8.5)
    a.legend(loc="lower right", fontsize=7)
    b = axs[1]
    Ls = [r["Lam"] for r in nbv["b"]]
    b.fill_between(Ls, [np.percentile(r["nb"], 10) for r in nbv["b"]], [np.percentile(r["nb"], 90) for r in nbv["b"]],
                   color=C1, alpha=0.15, lw=0)
    b.plot(Ls, [np.mean(r["sec"]) for r in nbv["b"]], "-", color=C2, lw=1.4, label="secular mean")
    b.plot(Ls, [np.mean(r["nb"]) for r in nbv["b"]], "o", color=C1, ms=4.5, label="N-body mean")
    b.set_xscale("log")
    b.set_xlabel(r"$\Lambda$")
    b.set_ylabel(r"$\langle e_f\rangle/e_\star$")
    b.set_title(r"(b) ensembles, $e_0=0.05\,e_\star$", loc="left", fontsize=8.5)
    b.legend(loc="upper left", fontsize=7)
    cc = axs[2]
    eps = np.array([r["eps"] for r in nbv["a"]])
    dfull = np.array([abs(r["nbRms"] / r["secRms"] - 1) for r in nbv["a"]])
    dq = np.array([abs(r["nbQuadRms"] / r["secQuadRms"] - 1) for r in nbv["a"]])
    cc.plot(eps, dfull, "o-", color=C1, ms=4.5, label="full tide vs quad.+oct.")
    cc.plot(eps, dq, "s-", color=C3, ms=4, label="quad. tide vs quad.")
    cc.plot(eps, dfull[0] * eps / eps[0], ":", color=INK2, lw=1, label=r"$\propto\varepsilon$")
    cc.plot(eps, dfull[0] * (eps / eps[0]) ** 2, "-.", color=MUTED, lw=1, label=r"$\propto\varepsilon^2$")
    cc.set_xscale("log")
    cc.set_yscale("log")
    cc.set_xlabel(r"$\varepsilon=n_p/n_m$")
    cc.set_ylabel("|rms ratio − 1|")
    cc.set_xticks(list(eps))
    cc.set_xticklabels([f"{e:g}" for e in eps])
    cc.minorticks_off()
    cc.set_title(r"(c) convergence, $\Lambda=1$", loc="left", fontsize=8.5)
    cc.legend(fontsize=6.8, loc="upper left")
    fig.tight_layout()
    save(fig, "fig5_nbody")


def fig6():
    S = D("secular_campaign.json")["seed"]
    nbv = D("nbody_validation.json")
    L, xis, ef = np.array(S["Lams"]), S["xis"], np.array(S["ef"])
    fig, ax = plt.subplots(figsize=(4.6, 3.1))
    for k, (xi, col) in enumerate(zip(xis, [C1, C2, C3, C4, C7])):
        ax.plot(L, ef[:, k], "-", color=col, lw=1.5, label=f"$\\xi={xi:g}$")
        Ls = np.logspace(-1, 0.3, 60)
        bare = 2 * xi * np.sqrt(Ls)
        ax.fill_between(Ls, bare * np.sqrt(2 * np.exp(2 * Ls) - 1), bare * (np.exp(Ls) + np.sqrt(np.exp(2 * Ls) - 1)),
                        color=col, alpha=0.15, lw=0)
    for r in nbv.get("s", []):
        ax.plot(r["Lam"], np.mean(r["nb"]), "D", color=INK, ms=4.5, mfc="white")
        ax.plot(r["Lam"], np.mean(r["sec"]), "x", color=INK, ms=4.5)
    if nbv.get("s"):
        xi_nb = nbv["s"][0]["alphaA"] / (4 * e_star(nbv["s"][0]["eps"]))
        ax.text(0.55, 0.035, f"N-body ($\\diamond$) and secular ($\\times$)\nfor $\\varepsilon=0.005$, $\\xi={xi_nb:.3f}$",
                fontsize=7, color=INK2, transform=ax.transAxes)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 2)
    ax.axhline(1, color=INK2, lw=0.8)
    ax.set_xlabel(r"adiabaticity $\Lambda$")
    ax.set_ylabel(r"$e_f/e_\star$ (initially circular)")
    ax.legend(fontsize=7, loc="upper left", title="octupole strength", title_fontsize=7)
    fig.tight_layout()
    save(fig, "fig6_octupole")


def fig7():
    camp = D("secular_campaign.json")
    RP = D("rhea_params.json")
    fig = plt.figure(figsize=(7.0, 4.8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05])
    a, b, cax = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])
    for eta, col in [(0.3, C3), (0.477, C1), (0.7, C7)]:
        tr = evolve(eta=eta)
        a.plot(tr["t"] / 1e6, tr["M"], color=col, lw=1.5, label=f"$\\eta={eta}$")
        loc = local_mass_loss(tr, RP["Mx"])
        a.plot(loc["t"] / 1e6, RP["Mx"], "o", color=col, ms=5.5, mfc="white", mew=1.5)
    a.axhline(RP["Mx"], color=C2, lw=0.8)
    a.text(30, RP["Mx"] + 0.008, "Rhea's crossing mass 0.907 $M_\\odot$", color=C2, fontsize=7.5)
    a.set_xlabel("time since base of RGB (Myr)")
    a.set_ylabel(r"$M_\odot(t)$")
    a.set_title("(a) solar mass loss", loc="left")
    a.legend(fontsize=7, loc="lower left")
    rs = camp["rheaSeries"]
    t = np.array(rs["t"]) / 1e3
    for j, e in enumerate(rs["e"]):
        b.plot(t, e, color=[C1, C2, C3, C7][j], lw=1.1)
    b.axhline(rs["eStar"], color=INK2, lw=0.8)
    b.text(t[0] + 2, rs["eStar"] * 1.04, "$e_\\star$", fontsize=8, color=INK2)
    b.set_xlabel("time from exact resonance (kyr)")
    b.set_ylabel("Rhea's eccentricity")
    b.set_title(f"(b) crossing, $\\eta=0.477$, $\\Lambda={rs['Lam']:.1f}$", loc="left")
    rows = camp["rhea"]
    for k, (eta, col) in enumerate([(0.3, C3), (0.477, C1), (0.7, C7)]):
        for e0, mk, off in [(1e-3, "o", -0.8), (0.0, "s", 0.8)]:
            sel = [r for r in rows if r["eta"] == eta and r["e0"] == e0]
            th = np.array([r["thetaDeg"] for r in sel]) + (k - 1) * 1.6 + off * 0.5
            m = np.array([r["mean"] for r in sel])
            lo = np.array([r["p05"] for r in sel])
            hi = np.array([r["p95"] for r in sel])
            cax.errorbar(th, m, yerr=[m - lo, hi - m], fmt=mk, color=col, ms=5, mfc="white" if e0 == 0 else col, capsize=2, lw=1)
    cax.axhline(0.00126, color=MUTED, lw=0.8)
    cax.text(22.6, 0.0019, "Rhea's present $e$", fontsize=7.5, color=MUTED)
    cax.set_xlim(22, 66)
    cax.set_ylim(0, 0.036)
    cax.set_xticks([26.73, 45, 60])
    cax.set_xticklabels(["26.7° (today)", "45°", "60° (Saillenfest+21)"])
    cax.set_xlabel("Saturn's obliquity at the time of crossing")
    cax.set_ylabel(r"Rhea's final $e$ (mean, 5–95%)")
    cax.set_title("(c) Monte Carlo over initial phase", loc="left")
    cax.legend(handles=[Line2D([], [], ls="", marker="o", color=C3, label="$\\eta=0.30$"),
                        Line2D([], [], ls="", marker="o", color=C1, label="$\\eta=0.477$"),
                        Line2D([], [], ls="", marker="o", color=C7, label="$\\eta=0.70$"),
                        Line2D([], [], ls="", marker="o", color=INK2, label="free $e_0=10^{-3}$"),
                        Line2D([], [], ls="", marker="s", color=INK2, mfc="white", label="initially circular")],
               ncol=5, fontsize=7, loc="upper right")
    fig.tight_layout()
    save(fig, "fig7_rhea")


def transfer_interp(lam, e0r):
    G = D("secular_campaign.json")["grid"]
    L, R, Mn = np.log(np.array(G["Lams"])), np.log(np.array(G["e0rs"])), np.array(G["mean"])
    x = np.clip(np.log(lam), L[0], L[-1])
    y = np.clip(np.log(max(e0r, 1e-5)), R[0], R[-1])
    i = int(np.clip(np.searchsorted(L, x) - 1, 0, len(L) - 2))
    j = int(np.clip(np.searchsorted(R, y) - 1, 0, len(R) - 2))
    fx = (x - L[i]) / (L[i + 1] - L[i])
    fy = (y - R[j]) / (R[j + 1] - R[j])
    v = ((1 - fx) * (1 - fy) * np.log(Mn[i, j]) + fx * (1 - fy) * np.log(Mn[i + 1, j])
         + (1 - fx) * fy * np.log(Mn[i, j + 1]) + fx * fy * np.log(Mn[i + 1, j + 1]))
    return np.exp(v)


def exomoon_curve(ap, J2=0.0147, eta=0.477, e0=1e-3, n=400):
    tr = evolve(eta=eta)
    Mwd = tr["M"][-1]
    GMj, Rj, YR = 1.26686534e17, 7.1492e7, 3.15576e7
    np0 = 2 * np.pi / ap ** 1.5
    ams = np.logspace(np.log10(2.5), np.log10(40), n)
    out = np.full((4, n), np.nan)
    for i, am in enumerate(ams):
        nm = np.sqrt(GMj / (am * Rj) ** 3) * YR
        g = 1.5 * nm * J2 / am ** 2
        if not (np0 * Mwd ** 2 < g < np0):
            continue
        Mx = np.sqrt(g / np0)
        npx = np0 * Mx ** 2
        eps = npx / nm
        tM = local_mass_loss(tr, Mx)["tM"]
        lam = adiabaticity(eps, npx * tM)
        es = e_star(eps)
        out[:, i] = [es * transfer_interp(lam, e0 / es), es, lam, Mx]
    return ams, out


def fig8():
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.9))
    for ap, col in [(3.0, C1), (5.2026, C2), (10.0, C3), (20.0, C7)]:
        ams, o = exomoon_curve(ap)
        a.plot(ams, o[0], color=col, lw=1.6, label=f"$a_p={ap:g}$ AU")
        a.plot(ams, o[1], ":", color=col, lw=1.0)
        b.plot(ams, o[2], color=col, lw=1.6)
    for name, am in [("Io", 5.90), ("Europa", 9.39), ("Ganymede", 14.97), ("Callisto", 26.33)]:
        a.axvline(am, color=GRID, lw=1.0, zorder=0)
        a.text(am * 1.03, 0.105, name, fontsize=6.8, color=MUTED, rotation=90, va="top")
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_xlim(3.5, 30)
    a.set_ylim(8e-4, 0.12)
    a.set_xlabel(r"moon orbit $a_m/R_p$")
    a.set_ylabel(r"post-crossing $e_f$ ($e_0=10^{-3}$)")
    a.set_title("(a) imprint (solid) and $e_\\star$ (dotted)", loc="left")
    a.legend(fontsize=7, loc="lower left")
    b.axhline(1, color=INK2, lw=0.8)
    b.set_xscale("log")
    b.set_yscale("log")
    b.set_xlim(3.5, 30)
    b.set_xlabel(r"moon orbit $a_m/R_p$")
    b.set_ylabel(r"adiabaticity $\Lambda$")
    b.set_title("(b) crossing adiabaticity", loc="left")
    b.text(0.03, 0.06, "jumps: crossing moves from the RGB wind\nto the red clump and the AGB superwind",
           transform=b.transAxes, fontsize=7, color=INK2)
    fig.tight_layout()
    save(fig, "fig8_exomoons")


def fig9():
    race = D("rhea_race.json")
    fig, ax = plt.subplots(figsize=(5.6, 3.3))
    for eta, col in [(0.3, C3), (0.477, C1), (0.7, C7)]:
        a = sorted([r for r in race if r["eta"] == eta and r["e0"] > 0], key=lambda r: r["f"])
        b = sorted([r for r in race if r["eta"] == eta and r["e0"] == 0], key=lambda r: r["f"])
        ax.plot([r["f"] for r in a], [r["mean"] for r in a], "o-", color=col, ms=3.2, lw=1.5,
                label=f"$\\eta={eta}$, free $e_0=10^{{-3}}$")
        ax.plot([r["f"] for r in b], [max(r["mean"], 1.2e-5) for r in b], "--", color=col, lw=1.0)
    ax.axhline(0.00126, color=MUTED, lw=0.8)
    ax.text(1.405, 0.00135, "today", fontsize=7.5, color=MUTED)
    ax.set_yscale("log")
    ax.set_ylim(1e-5, 0.06)
    ax.set_xlim(0.995, 1.45)
    ax.set_xlabel(r"Rhea's tidal expansion before the crossing, $f=a_m/a_{m,\rm now}$")
    ax.set_ylabel(r"Rhea's final eccentricity")
    ax.legend(fontsize=7, loc="lower left", title="solid: $e_0=10^{-3}$; dashed: initially circular", title_fontsize=7)
    top = ax.twiny()
    top.set_xlim(ax.get_xlim())
    top.set_xticks([expansion_from_Q(Q)[0] for Q in [300, 1000, 3000, 1e4]])
    top.set_xticklabels(["300", "$10^3$", "$3\\times10^3$", "$10^4$"])
    top.set_xlabel(r"Saturn's tidal $Q$ at Rhea's frequency (constant-$Q$, $k_2=0.39$)", fontsize=8)
    top.grid(False)
    ax.text(1.117, 0.045, "red-clump\nwindow", fontsize=7, color=INK2, ha="center")
    fig.tight_layout()
    save(fig, "fig9_race")


if __name__ == "__main__":
    for w in sys.argv[1:] or ["1", "2", "3", "4", "5", "6", "7", "8", "9"]:
        globals()["fig" + w]()
