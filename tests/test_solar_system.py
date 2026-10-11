import numpy as np

from evection_lab import stellar, satellites, rhea


def test_solar_track():
    s = stellar.summary(stellar.evolve(eta=0.477))
    assert 0.75 < s["M_tip_rgb"] < 0.80
    assert 0.54 < s["M_wd"] < 0.58


def test_titan_free_precession():
    pl = satellites.PLANETS["Saturn"]
    g, n, dJ2, names, w = satellites.free_modes(pl, 2 * np.pi / (pl["P"] * satellites.YR))
    period = 2 * np.pi / g[names.index("Titan")] / satellites.YR
    assert 695 < period < 715


def test_rhea_crossing():
    p = rhea.parameters(1.0)
    assert abs(p["Mx"] - 0.9066) < 1e-3
    assert abs(p["fJ2"] - 0.927) < 5e-3


def test_rhea_outcome():
    out = rhea.simulate(0.477, 26.73, 1.0, 1e-3, 16)
    assert 0.024 < out["stats"]["mean"] < 0.032
