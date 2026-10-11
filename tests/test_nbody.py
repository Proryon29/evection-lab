import numpy as np

from evection_lab import nbody, secular
from evection_lab.theory import e_star


def test_j2_precession():
    p = nbody.precession_test(n_orbits=300)
    assert abs(p["rel_err"]) < 2e-3


def test_adiabatic_invariant():
    o = nbody.run(0.01, 1.0, e0=0.02, phi0=0.5, steps=100)
    inv = o.invariant()
    assert np.max(np.abs(inv / inv[0] - 1)) < 1e-4


def test_nbody_matches_secular_quadrupole():
    eps, lam = 0.02, 0.5
    es = e_star(eps)
    e0 = 0.05 * es
    ph = secular.phases(4)
    nb = np.array([nbody.run(eps, lam, e0=e0, phi0=p, steps=150, quad_only=True).ef for p in ph])
    sec = secular.sweep(eps, lam, e0, ph, K0=6.0, K1=9.0, proper=False).ef
    assert abs(np.sqrt(np.mean(nb ** 2)) / np.sqrt(np.mean(sec ** 2)) - 1) < 0.25
