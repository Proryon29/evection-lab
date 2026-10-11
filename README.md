Evection Sweep Lab
==================

An open simulator for a new effect in celestial mechanics: when a Sun-like star loses half its mass
on the red-giant and asymptotic giant branches, every surviving planet's orbit slows down
(n_p ∝ M_star²), and the evection resonance (moon apsidal precession = planet mean motion) sweeps
outward through its moons. The sweep runs in the direction in which capture is impossible, so instead
of destroying moons it leaves a universal eccentricity on each one it passes.

The lab contains everything needed to explore and verify this:

- analytic theory: the parametric band, Landau–Zener amplification, the adiabatic separatrix jump map,
  the universal imprint e*, and octupole (parallactic) seeding of circular moons
- an orbit-averaged secular integrator (SciPy DOP853) with quadrupole and octupole stellar tides,
  planetary obliquity, extra precession sources and tidal damping
- a direct three-body integrator with stellar mass loss (4th-order Yoshida symplectic scheme,
  full unexpanded tide, optional Numba acceleration)
- a semi-analytic solar mass-loss history (Reimers RGB wind, red clump, Blöcker AGB superwind)
- Laplace–Lagrange free precession frequencies for the moons of Mars, Jupiter, Saturn, Uranus and Neptune
- the Saturn–Rhea application: crossing mass, Monte Carlo outcomes, tidal heating, and the
  "tides versus wind" race
- an interactive lab window, a command-line interface and a 12-test verification suite


Install
-------

Python 3.9 or newer.

```
git clone https://github.com/Proryon29/evection-lab.git
cd evection-lab
pip install -e .
pip install numba
```

`numba` is optional. Without it the N-body integrator runs in pure Python (about 30 µs per step);
with it, it is roughly 60 times faster. Everything else needs only NumPy, SciPy and Matplotlib.


Quick start
-----------

Open the interactive lab (Sweep, N-body and Rhea modes with sliders):

```
python -m evection_lab app
```

Run the verification suite (12 tests: precession, adiabatic invariant, Landau–Zener gains,
jump map, plateau, capture direction, N-body vs theory, octupole seeding, Rhea):

```
python -m evection_lab verify
python -m evection_lab verify --quick
```

Other commands:

```
python -m evection_lab sweep --eps 3e-3 --lam 8 --e0 0.03 --plot sweep.png
python -m evection_lab sweep --eps 3e-3 --lam 64 --e0 0.01 --capture
python -m evection_lab nbody --eps 0.01 --lam 1 --e0 0.05
python -m evection_lab rhea --eta 0.477 --theta 26.73 --f 1.0 --e0 1e-3
python -m evection_lab rhea --theta 60
python -m evection_lab crossings
python -m evection_lab star --eta 0.7
```


Python API
----------

```python
import numpy as np
from evection_lab import theory, secular, nbody, rhea

eps, lam = 3e-3, 16.0
es = theory.e_star(eps)
r = secular.sweep(eps, lam, 1e-3 * es, secular.phases(32))
print(r.ef.mean() / es)

o = nbody.run(0.01, 1.0, e0=0.01, phi0=0.26)
print(o.ef)

out = rhea.simulate(eta=0.477, theta_deg=26.73, f=1.0, e0=1e-3)
print(out["stats"]["mean"], out["result"].lam)
```


Main results reproduced by the code
-----------------------------------

| Quantity | Value |
|---|---|
| Sweep of the evection radius, 1 → 0.54 M_sun | ×1.42 in semi-major axis |
| Universal imprint | e* = [(15/4)(n_p/n_m) cos⁴(θ/2) / f_J2]^½ |
| Landau–Zener gain at Λ = 1 | 3.75 simulated vs 3.71 theory |
| Adiabatic jump map at Λ = 256 | agreement 0.1–1.1 % for 0.1 ≤ e0/e* ≤ 3 |
| N-body vs secular | < 2 % for n_p/n_m ≤ 0.005 |
| Rhea crossing | Sun at 0.907 M_sun on the upper red-giant branch, Λ = 8.8–18.8 |
| Rhea after the crossing | e = 0.026–0.031 (today 0.00126) |
| Tidal heating of Rhea afterwards | 3.7 GW for k2/Q = 1e-4 |


Repository layout
-----------------

| Path | Contents |
|---|---|
| `evection_lab/theory.py` | closed-form theory |
| `evection_lab/secular.py` | orbit-averaged model and integrator |
| `evection_lab/nbody.py` | direct three-body integrator with mass loss |
| `evection_lab/stellar.py` | semi-analytic stellar mass-loss history |
| `evection_lab/satellites.py` | planet and moon data, Laplace–Lagrange theory, crossing table |
| `evection_lab/rhea.py` | Saturn–Rhea application |
| `evection_lab/verification.py` | the 12-test verification suite |
| `evection_lab/app.py` | interactive lab window |
| `evection_lab/cli.py` | command-line interface |
| `scripts/` | production campaigns, figures, paper checks, full reproduction |
| `tests/` | unit tests (`pytest`) |
| `data/` | every data file behind the paper's numbers and figures |
| `figures/` | the nine paper figures (PDF and PNG) |
| `paper/` | the paper (LaTeX source and PDF) |
| `web/index.html` | browser version of the lab (works on GitHub Pages) |


Reproducing the paper
---------------------

```
python scripts/reproduce.py
python scripts/reproduce.py --skip-nbody
python scripts/make_figures.py
python scripts/check_paper.py
```

The full N-body validation campaign takes about an hour with Numba and much longer in pure Python;
use `--skip-nbody` to rebuild everything else in a few minutes. Compile the paper with
`pdflatex evection_sweep` (twice) inside `paper/`, keeping `figures/` next to it, or upload both
folders to Overleaf.


Browser version
---------------

`web/index.html` is a self-contained version of the lab (JavaScript engine, MathJax for the equations).
Enable GitHub Pages on the repository and point it at the `web/` folder to host it.


Testing
-------

```
pip install pytest
pytest
```


Citation
--------

If you use this software, please cite it using `CITATION.cff` and the accompanying paper
"Stellar Mass Loss Sweeps the Evection Resonance Through Satellite Systems: A Universal Eccentricity
Imprint and the Fate of Rhea".


License
-------

MIT. See `LICENSE`.
