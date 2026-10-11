import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
STEPS = [
    ["prepare_data.py"],
    ["run_secular_campaign.py"],
    ["run_timeseries.py"],
    ["run_rhea_race.py"],
    ["run_nbody_validation.py"],
    ["make_figures.py"],
    ["check_paper.py"],
]


def main():
    skip_nbody = "--skip-nbody" in sys.argv
    t0 = time.time()
    for step in STEPS:
        if skip_nbody and step[0] == "run_nbody_validation.py":
            print("skipping run_nbody_validation.py")
            continue
        print(f"\n=== {step[0]} ===", flush=True)
        code = subprocess.call([sys.executable, os.path.join(HERE, step[0])] + step[1:])
        if code != 0:
            print(f"{step[0]} failed with exit code {code}")
            return code
    print(f"\nall steps finished in {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
