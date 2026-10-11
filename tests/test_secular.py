import numpy as np

from evection_lab import secular
from evection_lab.theory import e_star, linear_amplification, adiabatic_map


def test_landau_zener_gain():
    es = e_star(3e-3)
    e0 = 1e-3 * es
    r = secular.sweep(3e-3, 1.0, e0, secular.phases(64))
    gain = np.sqrt(np.mean(r.ef ** 2)) / e0
    assert abs(gain / linear_amplification(1.0)["rms"] - 1) < 0.03


def test_window_independence_with_proper_start():
    es = e_star(3e-3)
    e0 = 1e-3 * es
    g = [np.sqrt(np.mean(secular.sweep(3e-3, 1.0, e0, secular.phases(48), K0=K0).ef ** 2)) / e0 for K0 in (7, 8, 15)]
    assert np.ptp(g) / np.mean(g) < 0.01


def test_jump_map_adiabatic():
    es = e_star(3e-3)
    r = secular.sweep(3e-3, 128.0, 0.5 * es, secular.phases(8), K0=secular.window_start(0.5))
    assert abs(np.mean(r.ef) / es / (adiabatic_map(0.5 * es, es)[0] / es) - 1) < 0.03


def test_capture_versus_noncapture():
    es = e_star(3e-3)
    up = secular.sweep(3e-3, 64.0, 0.01 * es, secular.phases(8))
    down = secular.sweep(3e-3, 64.0, 0.01 * es, secular.phases(8), direction=-1)
    assert np.mean(up.ef) / es < 1.05
    assert np.mean(down.ef) / es > 2.0


def test_octupole_seeds_circular_moon():
    es = e_star(3e-3)
    quad = secular.sweep(3e-3, 4.0, 0.0, [0.0])
    octu = secular.sweep(3e-3, 4.0, 0.0, [0.0], alpha=4 * 0.01 * es)
    assert quad.ef[0] < 1e-12
    assert octu.ef[0] / es > 0.3
