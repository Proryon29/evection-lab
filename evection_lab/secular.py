from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp

from .theory import c_width, e_star, adiabaticity, f_oct, f_theta


@dataclass
class SecularModel:
    np_of: object
    nm: float
    gJ2: float
    gExt: float = 0.0
    theta: float = 0.0
    tau_e: float = np.inf
    alpha_of: object = None

    def rhs(self):
        ft = f_theta(self.theta)
        c4 = np.cos(0.5 * self.theta) ** 4
        fo = f_oct(self.theta)
        inv_tau = 0.0 if not np.isfinite(self.tau_e) else 1.0 / self.tau_e
        np_of, nm, gJ2, gExt, alpha_of = self.np_of, self.nm, self.gJ2, self.gExt, self.alpha_of

        def f(t, y):
            h = y[0::2]
            k = y[1::2]
            e2 = np.minimum(h * h + k * k, 0.999999)
            s = np.sqrt(1.0 - e2)
            npt = np_of(t)
            tid = npt * npt / nm
            W = gJ2 / (1.0 - e2) ** 2 + gExt + 0.75 * tid * s * ft - npt
            C = 3.75 * tid * s * c4
            F = 0.9375 * tid * alpha_of(t) * fo if alpha_of is not None else 0.0
            out = np.empty_like(y)
            out[0::2] = -k * (W - C) - h * inv_tau
            out[1::2] = h * (W + C) + F * s - k * inv_tau
            return out

        return f

    def initial_state(self, t0, e0s, phis, proper=True):
        e0s = np.asarray(e0s, dtype=float)
        phis = np.asarray(phis, dtype=float)
        A, hf = 1.0, 0.0
        if proper:
            ft = f_theta(self.theta)
            c4 = np.cos(0.5 * self.theta) ** 4
            npt = self.np_of(t0)
            tid = npt * npt / self.nm
            W = self.gJ2 + self.gExt + 0.75 * tid * ft - npt
            C = 3.75 * tid * c4
            F = 0.9375 * tid * self.alpha_of(t0) * f_oct(self.theta) if self.alpha_of is not None else 0.0
            if abs(W) > 1.05 * C:
                A = (abs(W - C) / abs(W + C)) ** 0.25
                hf = -F / (W + C)
        y0 = np.empty(2 * e0s.size)
        y0[0::2] = hf + e0s * A * np.cos(phis)
        y0[1::2] = e0s / A * np.sin(phis)
        return y0


def integrate(model, y0, t0, t1, n_out, rtol=1e-10, atol=1e-14):
    t_eval = np.linspace(t0, t1, n_out)
    sol = solve_ivp(model.rhs(), (t0, t1), y0, method="DOP853", t_eval=t_eval, rtol=rtol, atol=atol)
    h = sol.y[0::2]
    k = sol.y[1::2]
    return sol.t, np.sqrt(h * h + k * k), np.arctan2(k, h)


def tail_rms(e, tail):
    nt = max(10, int(np.floor(tail * e.shape[1])))
    return np.sqrt(np.mean(e[:, -nt:] ** 2, axis=1))


@dataclass
class SweepResult:
    x: np.ndarray
    delta_over_c: np.ndarray
    e: np.ndarray
    phi: np.ndarray
    ef: np.ndarray
    c: float
    tau_n: float
    e_star: float
    alpha: float = 0.0
    extra: dict = field(default_factory=dict)


def window_start(e0_over_estar):
    return max(7.0, (1.0 + 1.5 * e0_over_estar) ** 2 + 5.0)


def sweep(eps, lam, e0, phi0, theta=0.0, fJ2=1.0, K0=7.0, K1=12.0, n_out=1500, tail=0.15,
          tau_e=np.inf, alpha=0.0, direction=1, proper=True, rtol=1e-10):
    c = c_width(eps, theta)
    tau_n = 2.0 * lam / (np.pi * c * c)
    ft = f_theta(theta)
    g_tot = 1.0 - 0.75 * eps * ft
    d = -1 if direction == -1 else 1
    phis = np.atleast_1d(np.asarray(phi0, dtype=float))
    e0s = np.broadcast_to(np.asarray(e0, dtype=float), phis.shape)
    model = SecularModel(np_of=lambda t: np.exp(-d * t / tau_n), nm=1.0 / eps, gJ2=fJ2 * g_tot,
                         gExt=(1.0 - fJ2) * g_tot, theta=theta, tau_e=tau_e,
                         alpha_of=(lambda t: alpha * np.exp(-0.5 * d * t / tau_n)) if alpha else None)
    t0, t1 = -K0 * c * tau_n, K1 * c * tau_n
    y0 = model.initial_state(t0, e0s, phis, proper)
    t, e, ph = integrate(model, y0, t0, t1, n_out, rtol=rtol)
    x = t / tau_n
    dc = (g_tot * np.exp(d * x) - 1.0 + 0.75 * eps * np.exp(-d * x) * ft) / c
    return SweepResult(x=x, delta_over_c=dc, e=e, phi=ph, ef=tail_rms(e, tail), c=c, tau_n=tau_n,
                       e_star=float(e_star(eps, theta, fJ2)), alpha=alpha)


@dataclass
class PhysicalResult:
    t: np.ndarray
    e: np.ndarray
    ef: np.ndarray
    eps: float
    c: float
    fJ2: float
    npx: float
    e_star: float
    lam: float


def physical(nm, gJ2, gExt, theta, tM, e0, phi0, alpha=0.0, K0=7.0, K1=12.0, n_out=1500,
             tau_e=np.inf, proper=True, tail=0.15):
    ft = f_theta(theta)
    g = gJ2 + gExt
    npx = g
    for _ in range(20):
        npx = g + 0.75 * npx * npx / nm * ft
    eps = npx / nm
    fJ2 = gJ2 / g
    c = c_width(eps, theta)
    tau_n = tM / 2.0
    t0, t1 = -K0 * c * tau_n, K1 * c * tau_n
    phis = np.atleast_1d(np.asarray(phi0, dtype=float))
    model = SecularModel(np_of=lambda t: npx * np.exp(-t / tau_n), nm=nm, gJ2=gJ2, gExt=gExt, theta=theta,
                         tau_e=tau_e, alpha_of=(lambda t: alpha * np.exp(-0.5 * t / tau_n)) if alpha else None)
    y0 = model.initial_state(t0, np.full(phis.shape, e0), phis, proper)
    t, e, _ = integrate(model, y0, t0, t1, n_out)
    return PhysicalResult(t=t, e=e, ef=tail_rms(e, tail), eps=eps, c=c, fJ2=fJ2, npx=npx,
                          e_star=float(e_star(eps, theta, fJ2)), lam=float(adiabaticity(eps, npx * tM, theta)))


def phases(n, offset=0.5):
    return 2.0 * np.pi * (np.arange(n) + offset) / n
