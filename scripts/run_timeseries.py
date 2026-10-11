import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from evection_lab import secular
from evection_lab.theory import e_star, c_width

DATA = os.path.join(os.path.dirname(__file__), "..", "data")


def thin(a, n=1200):
    a = np.asarray(a)
    idx = np.round(np.linspace(0, a.shape[-1] - 1, n)).astype(int)
    return a[..., idx]


def main():
    eps = 3e-3
    es = float(e_star(eps))
    out = {"eps": eps, "eStar": es, "c": float(c_width(eps)), "runs": []}
    ph = secular.phases(24)
    for lam, direction, e0r in ((1, 1, 0.01), (8, 1, 0.01), (128, 1, 0.01), (64, -1, 0.01)):
        r = secular.sweep(eps, lam, e0r * es, ph, n_out=3000, direction=direction)
        e = thin(r.e) / es
        phi = thin(r.phi)
        out["runs"].append({"Lam": lam, "dir": direction, "e0r": e0r, "dc": thin(r.delta_over_c).tolist(), "e": e.tolist(),
                            "h": (e * np.cos(phi)).tolist(), "k": (e * np.sin(phi)).tolist(), "ef": (r.ef / es).tolist()})
        print(f"Lambda={lam} direction={direction}: mean e_f/e* = {np.mean(r.ef) / es:.3f}", flush=True)
    with open(os.path.join(DATA, "timeseries.json"), "w") as fh:
        json.dump(out, fh)


if __name__ == "__main__":
    main()
