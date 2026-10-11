import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from evection_lab import secular, rhea, stellar
from evection_lab.theory import e_star, adiabatic_map

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
LAMS = [0.125, 0.25, 0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 128, 256]
E0RS = [1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]


def save(out):
    with open(os.path.join(DATA, "secular_campaign.json"), "w") as fh:
        json.dump(out, fh, default=float)


def main():
    t0 = time.time()
    out = {"grid": None, "seed": None, "rhea": [], "rheaSeries": None}
    out["rhea"] = rhea.monte_carlo()
    p = rhea.parameters(1.0)
    out["alphaRhea"] = p["alpha"]
    out["params"] = p
    tr = stellar.evolve(eta=0.477)
    loc = stellar.local_mass_loss(tr, p["Mx"])
    r = secular.physical(p["n_m"], p["gJ2"], p["gExt"], np.radians(26.73), loc["tM"], 1e-3, [0.3, 1.9, 3.5, 5.1],
                         alpha=p["alpha"], n_out=3000)
    out["rheaSeries"] = {"t": r.t.tolist(), "e": r.e.tolist(), "eStar": r.e_star, "Lam": r.lam, "c": r.c, "tM": loc["tM"]}
    save(out)
    eps = 3e-3
    es = float(e_star(eps))
    xis = [3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
    seed = {"eps": eps, "Lams": LAMS, "xis": xis, "ef": []}
    for L in LAMS:
        row = [float(secular.sweep(eps, L, 0.0, [0.0], n_out=2500, alpha=4 * xi * es).ef[0] / es) for xi in xis]
        seed["ef"].append(row)
        print(f"seed Lambda={L}: {np.round(row, 3)}", flush=True)
    out["seed"] = seed
    save(out)
    ph = secular.phases(32, 0.37)
    grid = {"eps": eps, "Lams": LAMS, "e0rs": E0RS, "mean": [], "rms": [], "p05": [], "p50": [], "p95": [],
            "adiabatic": [float(adiabatic_map(r0 * es, es)[0] / es) for r0 in E0RS]}
    for L in LAMS:
        row = {k: [] for k in ("mean", "rms", "p05", "p50", "p95")}
        for r0 in E0RS:
            res = secular.sweep(eps, L, r0 * es, ph, K0=secular.window_start(r0), n_out=2500)
            st = rhea.stats(res.ef / es)
            for k in row:
                row[k].append(st[k])
        for k in row:
            grid[k].append(row[k])
        out["grid"] = grid
        save(out)
        print(f"grid Lambda={L}: {np.round(row['mean'], 3)}  [{time.time() - t0:.0f} s]", flush=True)
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
