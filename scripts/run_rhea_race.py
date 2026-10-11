import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from evection_lab import rhea

DATA = os.path.join(os.path.dirname(__file__), "..", "data")


def main():
    rows = rhea.race()
    with open(os.path.join(DATA, "rhea_race.json"), "w") as fh:
        json.dump(rows, fh, default=float)
    print("wrote rhea_race.json")


if __name__ == "__main__":
    main()
