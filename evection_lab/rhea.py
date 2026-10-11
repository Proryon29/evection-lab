import numpy as np

from . import secular
from .satellites import PLANETS, YR, free_modes, secular_matrix, scaled_planet, crossing
from .stellar import evolve, local_mass_loss

G_SI = 6.674e-11
AU_KM = 1.495978707e8
SATURN_A_AU = 9.5826
SATURN_R_TIDE_M = 58232e3
SATURN_K2 = 0.390
RHEA_RADIUS_M = 763.8e3
RHEA_E_NOW = 0.00126
TITAN_E = 0.0288


def parameters(f=1.0):
    pl, j = scaled_planet("Saturn", "Rhea", f)
    cx = crossing(pl, j, M_low=0.45)
    if cx is None:
        return None
    a_km = PLANETS["Saturn"]["moons"][j][1] * f
    return {"f": f, "Mx": cx["Mx"], "n_m": cx["n_m"] * YR, "gJ2": cx["gJ2"] * YR, "gExt": cx["gExt"] * YR,
            "npx": cx["npx"] * YR, "fJ2": cx["fJ2"], "theta_deg": PLANETS["Saturn"]["obl"], "a_km": a_km,
            "alpha": a_km / (SATURN_A_AU * AU_KM / cx["Mx"])}


def forced_eccentricity_by_titan():
    pl = PLANETS["Saturn"]
    n_p0 = 2 * np.pi / (pl["P"] * YR)
    A, n, _, names = secular_matrix(pl, n_p0)
    j, k = names.index("Rhea"), names.index("Titan")
    gT = sorted(np.linalg.eigvals(A).real, key=lambda x: abs(x - A[k, k]))[0]
    return abs(A[j, k] * TITAN_E / (A[j, j] - gT))


def table(f_values=None):
    f_values = np.round(np.arange(1.0, 1.4501, 0.01), 3) if f_values is None else f_values
    rows = []
    for f in f_values:
        p = parameters(float(f))
        if p is not None:
            rows.append(p)
    return rows


def stats(values):
    v = np.sort(np.asarray(values))
    q = lambda p: v[min(len(v) - 1, max(0, int(round(p * (len(v) - 1)))))]
    return {"mean": float(v.mean()), "rms": float(np.sqrt(np.mean(v ** 2))), "p05": float(q(0.05)),
            "p50": float(q(0.5)), "p95": float(q(0.95)), "min": float(v[0]), "max": float(v[-1])}


def simulate(eta=0.477, theta_deg=26.73, f=1.0, e0=1e-3, n_phases=64, octupole=True, params=None, track=None,
             n_out=2500, phase_offset=0.37):
    p = params if params is not None else parameters(f)
    tr = track if track is not None else evolve(eta=eta)
    if p is None or p["Mx"] <= tr["M"][-1]:
        return {"crosses": False, "params": p}
    loc = local_mass_loss(tr, p["Mx"])
    phis = secular.phases(n_phases, phase_offset) if e0 > 0 else np.array([0.0])
    r = secular.physical(p["n_m"], p["gJ2"], p["gExt"], np.radians(theta_deg), loc["tM"], e0, phis,
                         alpha=p["alpha"] if octupole else 0.0, K0=7.0, K1=12.0, n_out=n_out)
    return {"crosses": True, "params": p, "local": loc, "result": r, "stats": stats(r.ef)}


def expansion_from_Q(Q, k2=SATURN_K2, dt_gyr=7.6):
    pl = PLANETS["Saturn"]
    GM_S = pl["GM"] * 1e9
    rhea = [m for m in pl["moons"] if m[0] == "Rhea"][0]
    a0 = rhea[1] * 1e3
    GM_R = rhea[2] * 1e9
    n0 = np.sqrt(GM_S / a0 ** 3)
    tau0 = 1.0 / (3 * k2 / Q * GM_R / GM_S * (SATURN_R_TIDE_M / a0) ** 5 * n0)
    f = (1 + 6.5 * dt_gyr * 1e9 * YR / tau0) ** (1 / 6.5)
    return f, tau0 / YR / 1e9


def tidal_heating(e, k2_over_Q, f=1.0):
    pl = PLANETS["Saturn"]
    Ms = pl["GM"] * 1e9 / G_SI
    rhea = [m for m in pl["moons"] if m[0] == "Rhea"][0]
    a = rhea[1] * 1e3 * f
    n = np.sqrt(pl["GM"] * 1e9 / a ** 3)
    return 10.5 * k2_over_Q * G_SI * Ms ** 2 * RHEA_RADIUS_M ** 5 * n * e ** 2 / a ** 6


def damping_time_yr(k2_over_Q, f=1.0):
    pl = PLANETS["Saturn"]
    Ms = pl["GM"] * 1e9 / G_SI
    rhea = [m for m in pl["moons"] if m[0] == "Rhea"][0]
    mR = rhea[2] * 1e9 / G_SI
    a = rhea[1] * 1e3 * f
    n = np.sqrt(pl["GM"] * 1e9 / a ** 3)
    return (4 / 63) / k2_over_Q * (mR / Ms) * (a / RHEA_RADIUS_M) ** 5 / n / YR


def monte_carlo(thetas=(26.73, 45.0, 60.0), etas=(0.3, 0.477, 0.7), e0s=(0.0, 3e-4, 1e-3, 2.2e-3), n_phases=64,
                verbose=True):
    p = parameters(1.0)
    tracks = {eta: evolve(eta=eta) for eta in etas}
    rows = []
    for th in thetas:
        for eta in etas:
            for e0 in e0s:
                out = simulate(eta, th, 1.0, e0, n_phases, True, params=p, track=tracks[eta])
                r = out["result"]
                row = {"thetaDeg": th, "eta": eta, "e0": e0, "tM": out["local"]["tM"], "Lam": r.lam, "eStar": r.e_star,
                       "eps": r.eps, "fJ2": r.fJ2, **out["stats"]}
                rows.append(row)
                if verbose:
                    print(f"theta={th} eta={eta} e0={e0}: Lambda={r.lam:.2f} e*={r.e_star:.4f} "
                          f"e_f={row['mean']:.4f} [{row['p05']:.4f}, {row['p95']:.4f}]", flush=True)
    return rows


def race(etas=(0.3, 0.477, 0.7), f_step=0.02, n_phases=32, verbose=True):
    tab = table()
    tracks = {eta: evolve(eta=eta) for eta in etas}
    rows = []
    for eta in etas:
        tr = tracks[eta]
        for p in tab:
            if round(p["f"] * 100) % round(f_step * 100) != 0 and p["f"] > 1.001:
                continue
            if p["Mx"] <= tr["M"][-1]:
                continue
            for e0 in (1e-3, 0.0):
                out = simulate(eta, 26.73, p["f"], e0, n_phases, True, params=p, track=tr, n_out=2000)
                r = out["result"]
                rows.append({"eta": eta, "f": p["f"], "Mx": p["Mx"], "phase": out["local"]["phase"],
                             "L": out["local"]["L"], "tM": out["local"]["tM"], "Lam": r.lam, "eStar": r.e_star,
                             "e0": e0, **out["stats"]})
            if verbose:
                a, b = rows[-2], rows[-1]
                print(f"eta={eta} f={p['f']:.2f} Mx={p['Mx']:.3f} phase={a['phase']} Lambda={a['Lam']:.2f} "
                      f"e_f(e0=1e-3)={a['mean']:.4f} e_f(e0=0)={b['mean']:.4f}", flush=True)
    return rows
