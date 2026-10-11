import time

import numpy as np

from . import nbody, secular, rhea
from .theory import e_star, c_width, linear_amplification, adiabatic_map

PAPER_RHEA_EF = 0.02896


def _rms(a):
    a = np.asarray(a)
    return float(np.sqrt(np.mean(a ** 2)))


def t_precession():
    p = nbody.precession_test(eps=0.01, r_ratio=1 / 3, e0=0.01, phi0=0.3, steps=200, n_orbits=600)
    return p["predicted"], p["measured"], 2e-3


def t_adiabatic_invariant():
    o = nbody.run(0.01, 1.0, e0=0.02, phi0=0.5, steps=150)
    inv = o.invariant()
    return inv[0], inv[-1], 2e-3


def t_lz_rms(lam, e0r, tol):
    def f():
        es = e_star(3e-3)
        e0 = e0r * es
        r = secular.sweep(3e-3, lam, e0, secular.phases(64))
        return linear_amplification(lam)["rms"], _rms(r.ef) / e0, tol
    return f


def t_lz_max():
    es = e_star(3e-3)
    e0 = 1e-3 * es
    r = secular.sweep(3e-3, 1.0, e0, secular.phases(96))
    return linear_amplification(1.0)["hi"], float(np.max(r.ef)) / e0, 0.03


def t_jump_map():
    es = e_star(3e-3)
    r0 = 0.7
    r = secular.sweep(3e-3, 256.0, r0 * es, secular.phases(16), K0=secular.window_start(r0))
    return float(adiabatic_map(r0 * es, es)[0] / es), float(np.mean(r.ef)) / es, 0.03


def t_plateau():
    es = e_star(3e-3)
    r = secular.sweep(3e-3, 256.0, 1e-3 * es, secular.phases(16))
    return 1.0, float(np.mean(r.ef)) / es, 0.1


def t_capture():
    eps, K0, K1, lam, n_out = 3e-3, 7.0, 12.0, 64.0, 2500
    es, c = e_star(eps), c_width(eps)
    r = secular.sweep(eps, lam, 0.01 * es, secular.phases(16), K0=K0, K1=K1, direction=-1, n_out=n_out)
    tau_n = 2 * lam / (np.pi * c * c)
    g_tot = 1 - 0.75 * eps

    def centre(npl):
        tid = npl * npl * eps
        f = lambda e: g_tot / (1 - e * e) ** 2 + 0.75 * tid * np.sqrt(1 - e * e) - npl - 3.75 * tid * np.sqrt(1 - e * e)
        lo, hi = 0.0, 0.9
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if f(mid) < 0:
                lo = mid
            else:
                hi = mid
        return lo

    nt = int(np.floor(0.15 * n_out))
    ts = np.linspace(-K0 * c * tau_n, K1 * c * tau_n, n_out)[-nt:]
    exp = np.sqrt(np.mean([centre(np.exp(t / tau_n)) ** 2 for t in ts])) / es
    return exp, float(np.mean(r.ef)) / es, 0.03


SETTINGS = {"quick": False, "workers": None}


def t_nbody_quad():
    eps, lam = 0.01, 1.0
    es = e_star(eps)
    e0 = 0.05 * es
    ph = secular.phases(4 if SETTINGS["quick"] else 8)
    nb = nbody.ensemble_ef(eps, lam, ph, workers=SETTINGS["workers"], e0=e0, steps=200, quad_only=True)
    s = secular.sweep(eps, lam, e0, ph, K0=6.0, K1=9.0, proper=False)
    return _rms(s.ef) / es, _rms(nb) / es, 0.08


def t_nbody_octupole():
    eps, lam = 0.01, 2.0
    S = nbody.setup(eps, lam)
    ph = secular.phases(1 if SETTINGS["quick"] else 2)
    nb = nbody.ensemble_ef(eps, lam, ph, workers=SETTINGS["workers"], e0=0.0, steps=200)
    s = secular.sweep(eps, lam, 0.0, [0.0], K0=6.0, K1=9.0, alpha=S.alpha, proper=False)
    return float(s.ef[0]) / np.sqrt(S.c), _rms(nb) / np.sqrt(S.c), 0.15


def t_rhea_mass():
    p = rhea.parameters(1.0)
    n_p0 = 2 * np.pi / 29.457
    return p["Mx"], np.sqrt((p["gJ2"] + p["gExt"]) / n_p0), 2e-3


def t_rhea_paper():
    out = rhea.simulate(0.477, 26.73, 1.0, 1e-3, 64, True)
    return PAPER_RHEA_EF, out["stats"]["mean"], 0.03


TESTS = [
    ("Apsidal precession from J2 (N-body, no tide)", t_precession),
    ("Adiabatic orbit expansion a_p (M + m_p) = const (N-body)", t_adiabatic_invariant),
    ("Landau-Zener rms gain, Lambda = 1", t_lz_rms(1.0, 1e-3, 0.03)),
    ("Landau-Zener rms gain, Lambda = 2.5", t_lz_rms(2.5, 1e-4, 0.04)),
    ("Landau-Zener maximum gain, Lambda = 1", t_lz_max),
    ("Adiabatic jump map, e0 = 0.7 e*, Lambda = 256", t_jump_map),
    ("Universal plateau, e0 = 1e-3 e*, Lambda = 256", t_plateau),
    ("Capture direction rides the island centre (Lambda = 64)", t_capture),
    ("Quadrupole N-body vs secular (eps = 0.01)", t_nbody_quad),
    ("Circular moon seeded by the octupole (eps = 0.01)", t_nbody_octupole),
    ("Rhea crossing mass", t_rhea_mass),
    ("Rhea final e (eta = 0.477, theta = 26.7 deg, e0 = 1e-3)", t_rhea_paper),
]


def run_all(verbose=True, skip_nbody=False, quick=False, workers=None):
    SETTINGS["quick"] = quick
    SETTINGS["workers"] = workers
    if verbose:
        print(f"N-body backend: {'numba' if nbody.HAVE_NUMBA else 'pure Python (install numba for ~60x speed)'}")
    results = []
    for name, fn in TESTS:
        if skip_nbody and "N-body" in name:
            continue
        t0 = time.time()
        try:
            exp, sim, tol = fn()
            dev = abs(sim / exp - 1)
            row = {"name": name, "expected": float(exp), "simulated": float(sim), "deviation": float(dev),
                   "tolerance": tol, "passed": bool(dev <= tol), "seconds": time.time() - t0}
        except Exception as err:
            row = {"name": name, "expected": np.nan, "simulated": np.nan, "deviation": np.nan, "tolerance": np.nan,
                   "passed": False, "seconds": time.time() - t0, "error": repr(err)}
        results.append(row)
        if verbose:
            mark = "PASS" if row["passed"] else "FAIL"
            print(f"[{mark}] {name:60s} expected {row['expected']:.5g}  simulated {row['simulated']:.5g}  "
                  f"deviation {100 * row['deviation']:.2f}% (tol {100 * row['tolerance']:.1f}%)  {row['seconds']:.1f}s",
                  flush=True)
    if verbose:
        n = sum(r["passed"] for r in results)
        print(f"{n} of {len(results)} tests passed")
    return results
