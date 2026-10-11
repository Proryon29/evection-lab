import copy

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

from .stellar import evolve, local_mass_loss
from .theory import e_star, adiabaticity

DAY = 86400.0
YR = 365.25 * DAY

PLANETS = {
    "Mars": dict(GM=42828.37, R=3396.2, J2=1960.45e-6, J4=-18.9e-6, obl=25.19, P=1.8808,
                 moons=[("Phobos", 9376.0, 7.087e-4), ("Deimos", 23463.2, 9.6e-5)]),
    "Jupiter": dict(GM=126686531.9, R=71492.0, J2=14696.5e-6, J4=-586.6e-6, obl=3.13, P=11.862,
                    moons=[("Io", 421800.0, 5959.9), ("Europa", 671100.0, 3202.7),
                           ("Ganymede", 1070400.0, 9887.8), ("Callisto", 1882700.0, 7179.3)]),
    "Saturn": dict(GM=37931207.7, R=60330.0, J2=16290.71e-6, J4=-935.83e-6, obl=26.73, P=29.457,
                   moons=[("Mimas", 185539.0, 2.503), ("Enceladus", 237948.0, 7.211),
                          ("Tethys", 294619.0, 41.21), ("Dione", 377396.0, 73.11),
                          ("Rhea", 527108.0, 153.94), ("Titan", 1221870.0, 8978.14),
                          ("Iapetus", 3560820.0, 120.5)]),
    "Uranus": dict(GM=5793939.0, R=25559.0, J2=3510.7e-6, J4=-34.2e-6, obl=97.77, P=84.011,
                   moons=[("Miranda", 129900.0, 4.3), ("Ariel", 190900.0, 83.5),
                          ("Umbriel", 266000.0, 85.1), ("Titania", 436300.0, 226.9),
                          ("Oberon", 583500.0, 205.3)]),
    "Neptune": dict(GM=6836529.0, R=25225.0, J2=3408.4e-6, J4=-33.4e-6, obl=28.32, P=164.79,
                    moons=[("Proteus", 117647.0, 2.9)]),
}
MMR_MOONS = {"Io", "Europa", "Ganymede", "Mimas", "Enceladus", "Tethys", "Dione"}


def laplace_coefficient(s, j, alpha):
    f = lambda psi: np.cos(j * psi) / (1 - 2 * alpha * np.cos(psi) + alpha ** 2) ** s
    return quad(f, 0, 2 * np.pi, limit=200)[0] / np.pi


def secular_matrix(planet, n_p):
    GM, R, J2, J4 = planet["GM"], planet["R"], planet["J2"], planet["J4"]
    ft = 0.5 * (3 * np.cos(np.radians(planet["obl"])) ** 2 - 1)
    names = [m[0] for m in planet["moons"]]
    a = np.array([m[1] for m in planet["moons"]])
    gm = np.array([m[2] for m in planet["moons"]])
    N = len(a)
    n = np.sqrt((GM + gm) / a ** 3)
    A = np.zeros((N, N))
    dJ2 = np.zeros(N)
    for j in range(N):
        x = (R / a[j]) ** 2
        dJ2[j] = n[j] * 1.5 * J2 * x
        A[j, j] = n[j] * (1.5 * J2 * x - 9.0 / 8.0 * J2 ** 2 * x ** 2 - 15.0 / 4.0 * J4 * x ** 2)
        A[j, j] += 0.75 * n_p ** 2 / n[j] * ft
        for k in range(N):
            if k == j:
                continue
            if a[j] < a[k]:
                al = a[j] / a[k]
                alb = al
            else:
                al = a[k] / a[j]
                alb = 1.0
            mu = gm[k] / (GM + gm[j])
            A[j, j] += 0.25 * n[j] * mu * al * alb * laplace_coefficient(1.5, 1, al)
            A[j, k] = -0.25 * n[j] * mu * al * alb * laplace_coefficient(1.5, 2, al)
    return A, n, dJ2, names


def free_modes(planet, n_p):
    A, n, dJ2, names = secular_matrix(planet, n_p)
    w, V = np.linalg.eig(A)
    w, V = w.real, V.real
    norms = np.linalg.norm(V, axis=0)
    order = [int(np.argmax(np.abs(V[j, :]) / norms)) for j in range(len(names))]
    g = np.array([w[i] for i in order])
    weight = np.array([abs(V[j, order[j]]) / norms[order[j]] for j in range(len(names))])
    return g, n, dJ2, names, weight


def scaled_planet(planet_name, moon_name, factor):
    pl = copy.deepcopy(PLANETS[planet_name])
    j = [m[0] for m in pl["moons"]].index(moon_name)
    m = list(pl["moons"][j])
    m[1] *= factor
    pl["moons"][j] = tuple(m)
    return pl, j


def crossing(planet, j, M_low=0.45):
    n_p0 = 2 * np.pi / (planet["P"] * YR)
    F = lambda M: free_modes(planet, n_p0 * M ** 2)[0][j] - n_p0 * M ** 2
    if not (F(1.0) < 0 < F(M_low)):
        return None
    Mx = brentq(F, M_low, 1.0, xtol=1e-12)
    npx = n_p0 * Mx ** 2
    g, n, dJ2, _, _ = free_modes(planet, npx)
    ft = 0.5 * (3 * np.cos(np.radians(planet["obl"])) ** 2 - 1)
    solar = 0.75 * npx ** 2 / n[j] * ft
    return {"Mx": Mx, "npx": npx, "n_m": n[j], "gJ2": dJ2[j], "gExt": g[j] - solar - dJ2[j],
            "fJ2": dJ2[j] / (g[j] - solar), "eps": npx / n[j]}


def solar_system_table(etas=(0.3, 0.477, 0.7)):
    tracks = {eta: evolve(eta=eta) for eta in etas}
    M_low = min(tr["M"][-1] for tr in tracks.values())
    rows = []
    for pname, pl in PLANETS.items():
        n_p0 = 2 * np.pi / (pl["P"] * YR)
        g0, n, dJ2, names, wgt = free_modes(pl, n_p0)
        for j, name in enumerate(names):
            row = dict(planet=pname, moon=name, a_over_R=pl["moons"][j][1] / pl["R"],
                       P_varpi_yr=2 * np.pi / g0[j] / YR if g0[j] > 0 else np.nan, P_orb_yr=pl["P"],
                       weight=wgt[j], mmr=name in MMR_MOONS, fJ2=dJ2[j] / g0[j], obl=pl["obl"], crosses=False)
            cx = crossing(pl, j, M_low) if g0[j] > 0 else None
            if cx is not None:
                th = np.radians(pl["obl"])
                row.update(crosses=True, M_cross=cx["Mx"], eps=cx["eps"], fJ2_cross=cx["fJ2"],
                           e_star=float(e_star(cx["eps"], th, cx["fJ2"])))
                for eta, tr in tracks.items():
                    loc = local_mass_loss(tr, cx["Mx"])
                    row[f"tM_{eta}"] = loc["tM"]
                    row[f"phase_{eta}"] = loc["phase"]
                    row[f"Lam_{eta}"] = float(adiabaticity(cx["eps"], cx["npx"] * YR * loc["tM"], th))
            rows.append(row)
    return rows


def format_table(rows):
    lines = [f"{'planet':8s} {'moon':10s} {'a/R':>6s} {'P_varpi':>9s} {'fJ2':>5s}  crossed   M_x     eps       e*      Lambda(0.3/0.477/0.7)"]
    for r in rows:
        s = (f"{r['planet']:8s} {r['moon']:10s} {r['a_over_R']:6.2f} {r['P_varpi_yr']:9.2f} {r['fJ2']:5.2f}  "
             f"{('yes' if r['crosses'] else 'no'):>4s}{'*' if r['mmr'] else ' '}  ")
        if r["crosses"]:
            s += (f" {r['M_cross']:.4f} {r['eps']:.3e} {r['e_star']:.4f}  "
                  f"{r['Lam_0.3']:.2f}/{r['Lam_0.477']:.2f}/{r['Lam_0.7']:.2f}")
        lines.append(s)
    lines.append("* moon in a mean-motion resonance: Laplace-Lagrange frequencies are indicative only")
    return "\n".join(lines)
