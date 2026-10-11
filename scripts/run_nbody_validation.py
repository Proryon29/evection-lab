import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from evection_lab import nbody, secular
from evection_lab.theory import e_star, linear_amplification, adiabatic_map

DATA = os.path.join(os.path.dirname(__file__), "..", "data")


def rms(a):
    return float(np.sqrt(np.mean(np.asarray(a) ** 2)))


def k0_for(e0r):
    return max(6.0, (1 + 1.5 * e0r) ** 2 + 5)


def ensemble(eps, lam, e0r, n, workers, quad=False):
    es = float(e_star(eps))
    e0 = e0r * es
    ph = secular.phases(n)
    K0, K1 = k0_for(e0r), 9.0
    S = nbody.setup(eps, lam, K0=K0)
    nb = nbody.ensemble_ef(eps, lam, ph, workers=workers, e0=e0, steps=200, K0=K0, K1=K1) / es
    sec = secular.sweep(eps, lam, e0, ph, K0=K0, K1=K1, n_out=3000, alpha=S.alpha, proper=False).ef / es
    r = {"eps": eps, "Lam": lam, "e0r": e0r, "alphaA": S.alpha, "phases": ph.tolist(), "nb": nb.tolist(),
         "sec": sec.tolist(), "nbMean": float(nb.mean()), "secMean": float(sec.mean()), "nbRms": rms(nb), "secRms": rms(sec)}
    if quad:
        nq = nbody.ensemble_ef(eps, lam, ph, workers=workers, e0=e0, steps=200, K0=K0, K1=K1, quad_only=True) / es
        sq = secular.sweep(eps, lam, e0, ph, K0=K0, K1=K1, n_out=3000, proper=False).ef / es
        r.update({"nbQuad": nq.tolist(), "secQuad": sq.tolist(), "nbQuadRms": rms(nq), "secQuadRms": rms(sq)})
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--max-lambda", type=float, default=32.0)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    w = args.workers
    quick = args.quick
    if quick:
        args.max_lambda = min(args.max_lambda, 1.0)
    print(f"N-body backend: {'numba' if nbody.HAVE_NUMBA else 'pure Python (slow; install numba)'}", flush=True)
    out = {"a": [], "b": [], "s": [], "c": None, "d": []}
    path = os.path.join(DATA, "nbody_validation_quick.json" if quick else "nbody_validation.json")

    def save():
        with open(path, "w") as fh:
            json.dump(out, fh, default=float)

    t0 = time.time()
    for eps in ((0.02, 0.01) if quick else (0.02, 0.01, 0.005, 0.0025)):
        r = ensemble(eps, 1.0, 0.05, 4 if quick else 12, w, quad=True)
        r["linRms"] = 0.05 * float(linear_amplification(1.0)["rms"])
        out["a"].append(r)
        save()
        print(f"(a) eps={eps}: full {r['nbRms']:.4f} vs {r['secRms']:.4f} | quad {r['nbQuadRms']:.4f} vs {r['secQuadRms']:.4f}"
              f"  [{time.time() - t0:.0f} s]", flush=True)
    eps, lam = (0.01, 2.0) if quick else (0.005, 8.0)
    es = float(e_star(eps))
    S = nbody.setup(eps, lam)
    o = nbody.run(eps, lam, e0=0.05 * es, phi0=1.0, steps=200, sample_every=5)
    s = secular.sweep(eps, lam, 0.05 * es, [1.0], K0=6.0, K1=9.0, n_out=3000, alpha=S.alpha, proper=False)
    out["c"] = {"eps": eps, "Lam": lam, "e0r": 0.05, "phi0": 1.0,
                "nb": {"x": o.t.tolist(), "e": (o.e / es).tolist(), "phi": o.phi.tolist(), "dOverC": o.delta_over_c.tolist()},
                "sec": {"x": s.x.tolist(), "e": (s.e[0] / es).tolist(), "phi": s.phi[0].tolist(), "dOverC": s.delta_over_c.tolist()},
                "nbEf": o.ef / es, "secEf": float(s.ef[0]) / es}
    save()
    print(f"(c) N-body {out['c']['nbEf']:.3f} vs secular {out['c']['secEf']:.3f}", flush=True)
    eps_b = 0.02 if quick else 0.005
    for lam in (0.5, 1, 2, 4, 8, 16, 32):
        if lam > args.max_lambda:
            break
        r = ensemble(eps_b, lam, 0.05, 4 if quick else (8 if lam >= 16 else 12), w)
        out["b"].append(r)
        save()
        print(f"(b) Lambda={lam}: N-body {r['nbMean']:.3f} vs secular {r['secMean']:.3f}  [{time.time() - t0:.0f} s]", flush=True)
    for lam in (0.5, 1, 2, 4, 8, 16):
        if lam > args.max_lambda:
            break
        r = ensemble(eps_b, lam, 0.0, 2 if quick else 4, w)
        out["s"].append(r)
        save()
        print(f"(s) Lambda={lam}: N-body {r['nbMean']:.4f} vs secular {r['secMean']:.4f}", flush=True)
    for e0r in ((0.5,) if quick else (0.5, 1.0, 2.0)):
        r = ensemble(eps_b, min(16.0, args.max_lambda), e0r, 2 if quick else 4, w)
        r["adi"] = float(adiabatic_map(e0r * e_star(eps_b), e_star(eps_b))[0] / e_star(eps_b))
        out["d"].append(r)
        save()
        print(f"(d) e0/e*={e0r}: N-body {r['nbMean']:.3f} vs secular {r['secMean']:.3f} vs adiabatic map {r['adi']:.3f}", flush=True)
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
