import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from evection_lab import stellar, satellites, rhea

DATA = os.path.join(os.path.dirname(__file__), "..", "data")


def dump(name, obj):
    with open(os.path.join(DATA, name), "w") as fh:
        json.dump(obj, fh, indent=1, default=float)
    print("wrote", name)


def main():
    os.makedirs(DATA, exist_ok=True)
    tracks = {}
    for eta in (0.3, 0.477, 0.7):
        tr = stellar.evolve(eta=eta)
        s = stellar.summary(tr)
        print(f"eta={eta}: RGB tip {s['M_tip_rgb']:.3f}, white dwarf {s['M_wd']:.3f}")
        keep = np.unique(np.linspace(0, tr["t"].size - 1, 1500).astype(int))
        tracks[str(eta)] = {k: np.asarray(tr[k])[keep].tolist() for k in ["t", "M", "L", "R", "mdot", "phase"]}
    dump("solar_tracks.json", tracks)
    rows = satellites.solar_system_table()
    print(satellites.format_table(rows))
    dump("solar_system_crossings.json", rows)
    p = rhea.parameters(1.0)
    for eta in (0.3, 0.477, 0.7):
        p[f"tM_{eta}"] = stellar.local_mass_loss(stellar.evolve(eta=eta), p["Mx"])["tM"]
    p["e_forced_by_Titan"] = rhea.forced_eccentricity_by_titan()
    p["Mwd"] = stellar.evolve(eta=0.477)["M"][-1]
    p["P_orb_final_yr"] = 29.457 / p["Mwd"] ** 2
    p["GM_R"] = 153.94
    p["GM_S"] = satellites.PLANETS["Saturn"]["GM"]
    p["R_S"] = satellites.PLANETS["Saturn"]["R"]
    dump("rhea_params.json", p)
    tab = rhea.table()
    for r in tab:
        print(f"f={r['f']:.2f}  Mx={r['Mx']:.4f}  fJ2={r['fJ2']:.3f}")
    dump("rhea_table.json", tab)
    mig = {f"f_at_Q{int(Q)}": rhea.expansion_from_Q(Q)[0] for Q in (300, 1000, 3000, 7000, 10000, 30000)}
    mig.update({"heat_e029_kQ1e4": rhea.tidal_heating(0.029, 1e-4), "heat_e029_kQ1e3": rhea.tidal_heating(0.029, 1e-3),
                "tau_e_kQ1e4_Myr": rhea.damping_time_yr(1e-4) / 1e6, "tau_e_kQ1e3_Myr": rhea.damping_time_yr(1e-3) / 1e6,
                "heat_now_kQ1e4": rhea.tidal_heating(rhea.RHEA_E_NOW, 1e-4)})
    dump("rhea_migration.json", mig)


if __name__ == "__main__":
    main()
