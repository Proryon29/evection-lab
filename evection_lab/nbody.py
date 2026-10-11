import math
from dataclasses import dataclass

import numpy as np

from .theory import c_width, e_star

try:
    from numba import njit
    HAVE_NUMBA = True
except ImportError:
    HAVE_NUMBA = False

    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]

        def deco(f):
            return f
        return deco


@dataclass
class NbodySetup:
    mp: float
    eps: float
    c: float
    np0: float
    np_res: float
    tau_n: float
    t_res: float
    nm: float
    am: float
    R: float
    J2: float
    g_target: float
    t_end: float
    Pm: float
    alpha: float
    e_star: float


def setup(eps, lam, r_ratio=1.0 / 3.0, mp=1e-3, K0=6.0, K1=9.0, t_end=None):
    c = c_width(eps, 0.0)
    np0 = math.sqrt(1.0 + mp)
    np_res = np0 * math.exp(-K0 * c)
    tau_n = 2.0 * lam / (math.pi * c * c) / np_res
    t_res = K0 * c * tau_n
    nm = np_res / eps
    am = (mp / (nm * nm)) ** (1.0 / 3.0)
    R = r_ratio * am
    g_target = np_res * (1.0 - 0.75 * eps)
    J2 = g_target / (1.5 * nm * r_ratio * r_ratio)
    te = t_end / np_res if t_end is not None else (K0 + K1) * c * tau_n
    ap_res = math.exp(0.5 * K0 * c)
    return NbodySetup(mp=mp, eps=eps, c=c, np0=np0, np_res=np_res, tau_n=tau_n, t_res=t_res, nm=nm, am=am, R=R,
                      J2=J2, g_target=g_target, t_end=te, Pm=2.0 * math.pi / nm, alpha=am / ap_res,
                      e_star=float(e_star(eps)))


@njit(cache=True)
def _advance(state, n_steps, h, steps_per_orbit, sample_every, mp, J2R2, tau, tide_on, loss_on, quad_only,
             nm, g_target, t_res, tau_n, out_t, out_e, out_varpi, out_phi, out_dc, out_ap, out_ep, k_out):
    Xx, Xy, Vx, Vy = state[0], state[1], state[2], state[3]
    rx, ry, vx, vy = state[4], state[5], state[6], state[7]
    t, sh, sk = state[8], state[9], state[10]
    cnt, orbit_count = int(state[11]), int(state[12])
    M0 = 1.0 + mp
    cb = 2.0 ** (1.0 / 3.0)
    w1 = 1.0 / (2.0 - cb)
    w0 = -cb / (2.0 - cb)
    n_cap = out_t.shape[0]
    for _ in range(n_steps):
        for s in range(3):
            hh = (w0 if s == 1 else w1) * h
            Xx += 0.5 * hh * Vx
            Xy += 0.5 * hh * Vy
            rx += 0.5 * hh * vx
            ry += 0.5 * hh * vy
            t += 0.5 * hh
            Mtot = M0 * math.exp(-t / (2.0 * tau)) if loss_on else M0
            Ms = Mtot - mp
            X2 = Xx * Xx + Xy * Xy
            X1 = math.sqrt(X2)
            X3 = X2 * X1
            Ax = -Mtot * Xx / X3
            Ay = -Mtot * Xy / X3
            r2 = rx * rx + ry * ry
            r1 = math.sqrt(r2)
            r3 = r2 * r1
            r5 = r3 * r2
            kk = mp / r3 + 1.5 * mp * J2R2 / r5
            ax = -kk * rx
            ay = -kk * ry
            if tide_on:
                if quad_only:
                    q = Ms / X3
                    rX = (rx * Xx + ry * Xy) / X2
                    ax += q * (3.0 * rX * Xx - rx)
                    ay += q * (3.0 * rX * Xy - ry)
                else:
                    Yx = Xx + rx
                    Yy = Xy + ry
                    Y2 = Yx * Yx + Yy * Yy
                    Y3 = Y2 * math.sqrt(Y2)
                    ax += Ms * (Xx / X3 - Yx / Y3)
                    ay += Ms * (Xy / X3 - Yy / Y3)
            Vx += hh * Ax
            Vy += hh * Ay
            vx += hh * ax
            vy += hh * ay
            Xx += 0.5 * hh * Vx
            Xy += 0.5 * hh * Vy
            rx += 0.5 * hh * vx
            ry += 0.5 * hh * vy
            t += 0.5 * hh
        r = math.sqrt(rx * rx + ry * ry)
        v2 = vx * vx + vy * vy
        rv = rx * vx + ry * vy
        sh += ((v2 - mp / r) * rx - rv * vx) / mp
        sk += ((v2 - mp / r) * ry - rv * vy) / mp
        cnt += 1
        if cnt == steps_per_orbit:
            orbit_count += 1
            if orbit_count % sample_every == 0 and k_out < n_cap:
                mex = sh / cnt
                mey = sk / cnt
                e = math.sqrt(mex * mex + mey * mey)
                vp = math.atan2(mey, mex)
                lp = math.atan2(Xy, Xx)
                Mtot = M0 * math.exp(-t / (2.0 * tau)) if loss_on else M0
                Xr = math.sqrt(Xx * Xx + Xy * Xy)
                V2 = Vx * Vx + Vy * Vy
                apo = 1.0 / (2.0 / Xr - V2 / Mtot)
                npl = math.sqrt(Mtot / (apo * apo * apo))
                XV = Xx * Vx + Xy * Vy
                epx = (V2 / Mtot - 1.0 / Xr) * Xx - XV * Vx / Mtot
                epy = (V2 / Mtot - 1.0 / Xr) * Xy - XV * Vy / Mtot
                c_now = 3.75 * npl / nm
                ph = vp - lp
                ph = math.atan2(math.sin(ph), math.cos(ph))
                out_t[k_out] = (t - t_res) / tau_n
                out_e[k_out] = e
                out_varpi[k_out] = vp
                out_phi[k_out] = ph
                out_dc[k_out] = (g_target / npl - 1.0 + 0.75 * npl / nm) / c_now
                out_ap[k_out] = apo
                out_ep[k_out] = math.sqrt(epx * epx + epy * epy)
                k_out += 1
            sh = 0.0
            sk = 0.0
            cnt = 0
    state[0] = Xx
    state[1] = Xy
    state[2] = Vx
    state[3] = Vy
    state[4] = rx
    state[5] = ry
    state[6] = vx
    state[7] = vy
    state[8] = t
    state[9] = sh
    state[10] = sk
    state[11] = cnt
    state[12] = orbit_count
    return k_out


@dataclass
class NbodyResult:
    t: np.ndarray
    e: np.ndarray
    varpi: np.ndarray
    phi: np.ndarray
    delta_over_c: np.ndarray
    ap: np.ndarray
    ep: np.ndarray
    ef: float
    setup: NbodySetup
    n_steps: int

    def invariant(self):
        S = self.setup
        Mt = (1.0 + S.mp) * np.exp(-(self.t * S.tau_n + S.t_res) / (2.0 * S.tau_n))
        return self.ap * Mt


def run(eps, lam, e0=0.0, phi0=0.0, r_ratio=1.0 / 3.0, mp=1e-3, steps=200, K0=6.0, K1=9.0, t_end=None,
        sample_every=1, tide=True, mass_loss=True, quad_only=False, tail=0.12, progress=None, chunk=200000):
    S = setup(eps, lam, r_ratio=r_ratio, mp=mp, K0=K0, K1=K1, t_end=t_end)
    J2R2 = S.J2 * S.R * S.R
    h = S.Pm / steps
    n_steps = int(math.ceil(S.t_end / h))
    lm = phi0 + 0.5 * math.pi
    vc = math.sqrt(mp / S.am + 1.5 * mp * J2R2 / S.am ** 3)
    state = np.array([1.0, 0.0, 0.0, S.np0,
                      S.am * math.cos(lm), S.am * math.sin(lm),
                      -vc * math.sin(lm) + e0 * vc * math.cos(lm), vc * math.cos(lm) + e0 * vc * math.sin(lm),
                      0.0, 0.0, 0.0, 0.0, 0.0])
    n_cap = n_steps // (steps * sample_every) + 2
    arrays = [np.zeros(n_cap) for _ in range(7)]
    k_out = 0
    done = 0
    while done < n_steps:
        m = min(chunk, n_steps - done)
        k_out = _advance(state, m, h, steps, sample_every, mp, J2R2, S.tau_n, tide, mass_loss, quad_only, S.nm,
                         S.g_target, S.t_res, S.tau_n, *arrays, k_out)
        done += m
        if progress is not None:
            progress(done / n_steps)
    t, e, varpi, phi, dc, ap, ep = (a[:k_out] for a in arrays)
    nt = max(5, int(math.floor(tail * k_out)))
    ef = float(np.sqrt(np.mean(e[-nt:] ** 2))) if k_out else float("nan")
    return NbodyResult(t=t, e=e, varpi=varpi, phi=phi, delta_over_c=dc, ap=ap, ep=ep, ef=ef, setup=S, n_steps=n_steps)


def _final_e(kwargs):
    return run(**kwargs).ef


def ensemble_ef(eps, lam, phis, workers=None, **kwargs):
    jobs = [dict(eps=eps, lam=lam, phi0=float(p), **kwargs) for p in phis]
    if workers == 1 or len(jobs) == 1:
        return np.array([_final_e(j) for j in jobs])
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return np.array(list(pool.map(_final_e, jobs)))


def estimated_seconds(eps, lam, steps=200, e0_over_estar=0.0, r_ratio=1.0 / 3.0, mp=1e-3):
    K0 = max(6.0, (1.0 + 1.5 * e0_over_estar) ** 2 + 5.0)
    S = setup(eps, lam, r_ratio=r_ratio, mp=mp, K0=K0)
    n = S.t_end / S.Pm * steps
    return n * (0.5e-6 if HAVE_NUMBA else 31e-6), n


def precession_test(eps=0.01, r_ratio=1.0 / 3.0, e0=0.01, phi0=0.3, steps=200, n_orbits=600):
    S = setup(eps, 1.0, r_ratio=r_ratio)
    o = run(eps, 1.0, e0=e0, phi0=phi0, r_ratio=r_ratio, steps=steps, t_end=n_orbits * S.Pm * S.np_res,
            tide=False, mass_loss=False)
    vp = np.unwrap(o.varpi)
    tt = o.t * S.tau_n + S.t_res
    slope = np.polyfit(tt, vp, 1)[0]
    predicted = S.nm * 1.5 * S.J2 * (S.R / S.am) ** 2
    return {"measured": slope, "predicted": predicted, "rel_err": slope / predicted - 1.0}
