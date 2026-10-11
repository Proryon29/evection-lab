import argparse
import json
import sys

import numpy as np

from . import __version__, theory, secular, nbody, rhea, satellites, stellar, verification


def cmd_app(args):
    from .app import main
    main()


def cmd_verify(args):
    res = verification.run_all(verbose=True, skip_nbody=args.skip_nbody, quick=args.quick, workers=args.workers)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(res, fh, indent=1)
    return 0 if all(r["passed"] for r in res) else 1


def cmd_sweep(args):
    es = float(theory.e_star(args.eps, np.radians(args.theta), args.fJ2))
    e0r = 0.0 if args.circular else args.e0
    r = secular.sweep(args.eps, args.lam, e0r * es, secular.phases(args.members), theta=np.radians(args.theta),
                      fJ2=args.fJ2, K0=secular.window_start(e0r), alpha=args.alpha,
                      direction=-1 if args.capture else 1)
    ef = r.ef / es
    lin = theory.linear_amplification(args.lam)
    print(f"c = {r.c:.5f}   e* = {es:.5f}   Lambda = {args.lam:g}")
    print(f"e_f/e*: mean {ef.mean():.4f}  rms {np.sqrt(np.mean(ef ** 2)):.4f}  "
          f"5-95% {np.percentile(ef, 5):.4f}-{np.percentile(ef, 95):.4f}")
    if e0r > 0:
        print(f"Landau-Zener rms prediction {e0r * lin['rms']:.4f}   adiabatic map {theory.adiabatic_map(e0r * es, es)[0] / es:.4f}")
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.axvspan(-1, 1, color="#f6e6d6")
        for j in range(min(12, args.members)):
            ax.plot(r.delta_over_c, r.e[j] / es, color="#2a78d6", lw=0.8)
        ax.axhline(1, color="#6c7685", lw=0.8)
        ax.set_xlabel("delta / c")
        ax.set_ylabel("e / e*")
        fig.tight_layout()
        fig.savefig(args.plot, dpi=150)
        print(f"saved {args.plot}")


def cmd_nbody(args):
    es = float(theory.e_star(args.eps))
    K0 = max(6.0, (1 + 1.5 * args.e0) ** 2 + 5)
    sec, n = nbody.estimated_seconds(args.eps, args.lam, args.steps, args.e0)
    print(f"backend {'numba' if nbody.HAVE_NUMBA else 'pure Python'}: {n:.2e} steps, about {sec:.0f} s")
    o = nbody.run(args.eps, args.lam, e0=args.e0 * es, phi0=np.radians(args.phi0), steps=args.steps, K0=K0,
                  quad_only=args.quad, progress=lambda f: print(f"\r{100 * f:5.1f}%", end="", flush=True))
    print()
    s = secular.sweep(args.eps, args.lam, args.e0 * es, [np.radians(args.phi0)], K0=K0, K1=9.0,
                      alpha=0.0 if args.quad else o.setup.alpha, proper=False)
    print(f"N-body e_f/e* = {o.ef / es:.4f}   secular e_f/e* = {s.ef[0] / es:.4f}   "
          f"difference {100 * (o.ef / s.ef[0] - 1):.2f}%")


def cmd_rhea(args):
    out = rhea.simulate(args.eta, args.theta, args.f, args.e0, args.phases, not args.no_octupole)
    if not out["crosses"]:
        print("no crossing before the white-dwarf stage")
        return
    p, loc, r, st = out["params"], out["local"], out["result"], out["stats"]
    print(f"crossing at M = {p['Mx']:.4f} M_sun ({stellar.PHASE_NAMES[loc['phase']]}), L = {loc['L']:.0f} L_sun")
    print(f"t_M = {loc['tM']:.3e} yr   Lambda = {r.lam:.2f}   e* = {r.e_star:.4f}   f_J2 = {r.fJ2:.3f}")
    print(f"final e: mean {st['mean']:.4f}   5-95% {st['p05']:.4f}-{st['p95']:.4f}")
    print(f"tidal heating {rhea.tidal_heating(st['mean'], args.kq, args.f) / 1e9:.3g} GW   "
          f"damping time {rhea.damping_time_yr(args.kq, args.f) / 1e6:.3g} Myr   (k2/Q = {args.kq:g})")


def cmd_crossings(args):
    print(satellites.format_table(satellites.solar_system_table()))


def cmd_star(args):
    tr = stellar.evolve(Mi=args.mass, eta=args.eta, etaB=args.etaB)
    s = stellar.summary(tr)
    print(f"RGB tip: M = {s['M_tip_rgb']:.3f} M_sun, L = {s['L_tip_rgb']:.0f} L_sun, R = {s['R_tip_rgb']:.0f} R_sun, "
          f"t = {s['t_rgb_Myr']:.0f} Myr after the RGB base")
    print(f"white dwarf: {s['M_wd']:.3f} M_sun")
    for M in (0.95, 0.9, 0.85, 0.8, 0.7, 0.6):
        loc = stellar.local_mass_loss(tr, M)
        if loc:
            print(f"  M = {M:.2f}: t_M = {loc['tM']:.3e} yr ({stellar.PHASE_NAMES[loc['phase']]})")


def build_parser():
    p = argparse.ArgumentParser(prog="evection-lab", description="Evection Sweep Lab")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="command")
    sub.add_parser("app", help="open the interactive lab").set_defaults(func=cmd_app)
    v = sub.add_parser("verify", help="run the verification suite")
    v.add_argument("--quick", action="store_true")
    v.add_argument("--skip-nbody", action="store_true")
    v.add_argument("--workers", type=int, default=None)
    v.add_argument("--json", default=None)
    v.set_defaults(func=cmd_verify)
    s = sub.add_parser("sweep", help="orbit-averaged sweep through the resonance")
    s.add_argument("--eps", type=float, default=3e-3)
    s.add_argument("--lam", type=float, default=8.0)
    s.add_argument("--e0", type=float, default=0.03, help="initial e in units of e*")
    s.add_argument("--theta", type=float, default=0.0)
    s.add_argument("--fJ2", type=float, default=1.0)
    s.add_argument("--alpha", type=float, default=0.0, help="a_m/a_p for the octupole term (0 = off)")
    s.add_argument("--members", type=int, default=32)
    s.add_argument("--circular", action="store_true")
    s.add_argument("--capture", action="store_true")
    s.add_argument("--plot", default=None)
    s.set_defaults(func=cmd_sweep)
    n = sub.add_parser("nbody", help="direct three-body integration with mass loss")
    n.add_argument("--eps", type=float, default=0.01)
    n.add_argument("--lam", type=float, default=1.0)
    n.add_argument("--e0", type=float, default=0.05, help="initial e in units of e*")
    n.add_argument("--phi0", type=float, default=57.0)
    n.add_argument("--steps", type=int, default=200)
    n.add_argument("--quad", action="store_true")
    n.set_defaults(func=cmd_nbody)
    r = sub.add_parser("rhea", help="simulate Rhea's crossing as the Sun dies")
    r.add_argument("--eta", type=float, default=0.477)
    r.add_argument("--theta", type=float, default=26.73)
    r.add_argument("--f", type=float, default=1.0)
    r.add_argument("--e0", type=float, default=1e-3)
    r.add_argument("--phases", type=int, default=64)
    r.add_argument("--kq", type=float, default=1e-4)
    r.add_argument("--no-octupole", action="store_true")
    r.set_defaults(func=cmd_rhea)
    sub.add_parser("crossings", help="which Solar-System moons are swept").set_defaults(func=cmd_crossings)
    st = sub.add_parser("star", help="semi-analytic mass-loss history")
    st.add_argument("--mass", type=float, default=1.0)
    st.add_argument("--eta", type=float, default=0.477)
    st.add_argument("--etaB", type=float, default=0.05)
    st.set_defaults(func=cmd_star)
    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 0
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())
