import numpy as np


def c_width(eps, theta=0.0):
    return 3.75 * eps * np.cos(0.5 * theta) ** 4


def e_star(eps, theta=0.0, fJ2=1.0):
    return np.sqrt(c_width(eps, theta) / fJ2)


def adiabaticity(eps, np_tM, theta=0.0):
    c = c_width(eps, theta)
    return np.pi * c * c * np_tM / 4.0


def linear_amplification(lam):
    a2 = np.exp(np.minimum(2.0 * lam, 700.0))
    b2 = a2 - 1.0
    return {
        "mean_sq": a2 + b2,
        "rms": np.sqrt(a2 + b2),
        "lo": np.sqrt(a2) - np.sqrt(b2),
        "hi": np.sqrt(a2) + np.sqrt(b2),
    }


def separatrix_areas(q):
    q = np.asarray(q, dtype=float)
    s = 2.0 * np.sqrt(np.maximum(q - 1.0, 0.0)) + 2.0 * q * np.arcsin(1.0 / np.sqrt(q))
    return np.pi * q - s, np.pi * q + s


def adiabatic_map(e0, estar):
    e0 = np.atleast_1d(np.asarray(e0, dtype=float))
    out = np.empty_like(e0)
    for i, ee in enumerate(e0):
        target = 2.0 * np.pi * (ee / estar) ** 2
        if target <= 0.0:
            out[i] = estar
            continue
        lo, hi = 1.0, 2.0
        while separatrix_areas(hi)[0] < target:
            hi *= 2.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if separatrix_areas(mid)[0] < target:
                lo = mid
            else:
                hi = mid
        out[i] = estar * np.sqrt(separatrix_areas(0.5 * (lo + hi))[1] / (2.0 * np.pi))
    return out


def adiabatic_map_asymptote(e0, estar):
    return np.asarray(e0) + 2.0 * np.sqrt(2.0) / np.pi * estar


def f_theta(theta):
    return 0.5 * (3.0 * np.cos(theta) ** 2 - 1.0)


def f_oct(theta):
    cp = 0.5 * (1.0 + np.cos(theta))
    cm = 0.5 * (1.0 - np.cos(theta))
    return 5.0 * (cp ** 3 + 2.0 * cp * cm ** 2) - 4.0 * cp


def hamiltonian(e, phi, w_j2, w_ext, eps, theta=0.0, oct_force=0.0):
    ft = f_theta(theta)
    c4 = np.cos(0.5 * theta) ** 4
    e2 = e * e
    s = np.sqrt(1.0 - e2)
    return (w_j2 / 3.0 / s ** 3 + (1.0 - w_ext) * s
            + 0.25 * eps * (ft * (1.0 + 1.5 * e2) + 7.5 * c4 * e2 * np.cos(2.0 * phi))
            + oct_force * e * np.cos(phi))


def octupole_xi(alpha, eps, theta=0.0, fJ2=1.0):
    return alpha * f_oct(theta) / (4.0 * np.cos(0.5 * theta) ** 4 * e_star(eps, theta, fJ2))


def octupole_seed(alpha, lam, theta=0.0):
    return 0.5 * alpha * np.abs(f_oct(theta)) / np.cos(0.5 * theta) ** 4 * np.sqrt(lam)


def evection_radius(J2, R, GMp, n_p, fJ2=1.0):
    return (1.5 * J2 * R ** 2 * np.sqrt(GMp) / (fJ2 * n_p)) ** (2.0 / 7.0)


def estar_from_planet(J2, R_over_a, theta=0.0, fJ2=1.0):
    return np.sqrt(45.0 * J2 / 8.0) * R_over_a * np.cos(0.5 * theta) ** 2 / fJ2


def sweep_factor(M_initial, M_final):
    return (M_initial / M_final) ** (4.0 / 7.0)
