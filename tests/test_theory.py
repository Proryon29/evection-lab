import numpy as np

from evection_lab import theory


def test_estar_scaling():
    assert np.isclose(theory.e_star(0.01), np.sqrt(3.75 * 0.01))
    assert np.isclose(theory.e_star(0.01, fJ2=0.5) ** 2, 2 * theory.e_star(0.01) ** 2)


def test_lambda_formula():
    eps, np_tM = 3.4e-4, 1e7
    assert np.isclose(theory.adiabaticity(eps, np_tM), 225 * np.pi / 64 * eps ** 2 * np_tM)


def test_landau_zener_symplectic():
    a = theory.linear_amplification(1.3)
    assert np.isclose(a["hi"] * a["lo"], 1.0)
    assert np.isclose(a["mean_sq"], 2 * np.exp(2.6) - 1)


def test_separatrix_limits():
    A_in, A_out = theory.separatrix_areas(1.0)
    assert np.isclose(A_in, 0.0)
    assert np.isclose(A_out, 2 * np.pi)


def test_adiabatic_map_limits():
    es = 0.1
    assert np.isclose(theory.adiabatic_map(1e-8, es)[0], es, rtol=1e-3)
    big = theory.adiabatic_map(5 * es, es)[0]
    assert abs(big - theory.adiabatic_map_asymptote(5 * es, es)) < 0.02 * es


def test_obliquity_factors():
    assert np.isclose(theory.f_oct(0.0), 1.0)
    assert np.isclose(theory.f_theta(0.0), 1.0)
    assert np.isclose(theory.f_oct(np.radians(26.73)), 0.482, atol=2e-3)


def test_sweep_factor():
    assert np.isclose(theory.sweep_factor(1.0, 0.54), 1.42, atol=0.01)


def test_estar_from_planet_rhea():
    assert np.isclose(theory.estar_from_planet(0.01629, 1 / 8.737, np.radians(26.73), 0.928), 0.0354, atol=1e-3)
