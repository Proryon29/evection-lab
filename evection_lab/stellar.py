import numpy as np

LSUN = 3.828e33
MSUN = 1.989e33
YR = 3.15576e7
QH = 6.0e18
XH = 0.70
CORE_RATE = LSUN * YR / (XH * QH * MSUN)

PHASE_NAMES = {1: "red-giant branch", 2: "red clump", 3: "asymptotic giant branch", 4: "white dwarf"}


def luminosity_rgb(Mc):
    return 10 ** 5.3 * Mc ** 6 / (1 + 10 ** 0.4 * Mc ** 4 + 10 ** 0.5 * Mc ** 5)


def radius_rgb(Mc):
    return 3.7e3 * Mc ** 4 / (1 + Mc ** 3 + 1.75 * Mc ** 4)


def reimers(L, R, M, eta):
    return 4e-13 * eta * L * R / M


def bloecker(L, R, M, Mi, etaB):
    return etaB * 4.83e-9 * Mi ** -2.1 * L ** 2.7 * reimers(L, R, M, 1.0)


def evolve(Mi=1.0, eta=0.477, etaB=0.05, Mc0=0.17, Mc_tip=0.47, Mc_agb0=0.505, t_hb=1.1e8,
           L_hb=45.0, R_hb=10.0, Teff_agb=3300.0, dlnmax=2e-3):
    t, M, Mc = 0.0, Mi, Mc0
    rows = []
    while Mc < Mc_tip:
        L, R = luminosity_rgb(Mc), radius_rgb(Mc)
        mdot = reimers(L, R, M, eta)
        mcdot = CORE_RATE * L
        dt = dlnmax * min(Mc / mcdot, M / mdot)
        rows.append((t, M, Mc, L, R, mdot, 1))
        t += dt
        Mc += mcdot * dt
        M -= mdot * dt
    n = 200
    for _ in range(n):
        mdot = reimers(L_hb, R_hb, M, eta)
        rows.append((t, M, Mc, L_hb, R_hb, mdot, 2))
        dt = t_hb / n
        t += dt
        M -= mdot * dt
    Mc = Mc_agb0
    while M - Mc > 1e-3:
        L = 59250.0 * (Mc - 0.495)
        R = np.sqrt(L) * (5772.0 / Teff_agb) ** 2
        mdot = max(reimers(L, R, M, eta), bloecker(L, R, M, Mi, etaB))
        mcdot = CORE_RATE * L
        dt = dlnmax * min(Mc / mcdot, (M - Mc) / (mdot + mcdot), M / mdot)
        rows.append((t, M, Mc, L, R, mdot, 3))
        t += dt
        Mc += mcdot * dt
        M -= mdot * dt
    rows.append((t, Mc, Mc, 0.0, 0.0, 0.0, 4))
    a = np.array(rows)
    return {"t": a[:, 0], "M": a[:, 1], "Mc": a[:, 2], "L": a[:, 3], "R": a[:, 4], "mdot": a[:, 5],
            "phase": a[:, 6].astype(int)}


def local_mass_loss(track, Mx):
    idx = np.where(track["M"] <= Mx)[0]
    if idx.size == 0:
        return None
    i = idx[0]
    if track["mdot"][i] <= 0:
        i -= 1
    return {"tM": track["M"][i] / track["mdot"][i], "phase": int(track["phase"][i]), "t": track["t"][i],
            "L": track["L"][i], "R": track["R"][i]}


def summary(track):
    rgb = track["phase"] == 1
    return {"M_tip_rgb": track["M"][rgb][-1], "L_tip_rgb": track["L"][rgb][-1], "R_tip_rgb": track["R"][rgb][-1],
            "t_rgb_Myr": track["t"][rgb][-1] / 1e6, "M_wd": track["M"][-1]}
