import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TEX = os.path.join(HERE, "..", "paper", "evection_sweep.tex")
FIGS = os.path.join(HERE, "..", "figures")


def main():
    with open(TEX, encoding="utf-8") as fh:
        src = re.sub(r"(^|[^\\])%.*$", r"\1", fh.read(), flags=re.M)
    depth = low = 0
    for ch in re.sub(r"\\[{}]", "", src):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            low = min(low, depth)
    stack, mismatched = [], []
    for kind, env in re.findall(r"\\(begin|end)\{([^}]+)\}", src):
        if kind == "begin":
            stack.append(env)
        elif not stack or stack.pop() != env:
            mismatched.append(env)
    labels = set(re.findall(r"\\label\{([^}]+)\}", src))
    refs = set(re.findall(r"\\(?:eq)?ref\{([^}]+)\}", src))
    keys = set(re.findall(r"\\bibitem\[[^\]]*\]\{([^}]+)\}", src))
    cites = {c.strip() for group in re.findall(r"\\cite(?:p|t|alt)?\{([^}]+)\}", src) for c in group.split(",")}
    figs = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", src)
    missing = [f for f in figs if not os.path.exists(os.path.join(FIGS, f))]
    non_ascii = sorted({ch for ch in src if ord(ch) > 127})
    report = {
        "brace balance": depth, "negative depth": low < 0, "unclosed environments": stack,
        "mismatched environments": mismatched, "undefined references": sorted(refs - labels),
        "undefined citations": sorted(cites - keys), "uncited entries": sorted(keys - cites),
        "missing figures": missing, "non-ASCII characters": non_ascii,
    }
    for k, v in report.items():
        print(f"{k:24s} {v}")
    ok = depth == 0 and low >= 0 and not stack and not mismatched and not (refs - labels) and not (cites - keys) \
        and not missing and not non_ascii
    print("paper OK" if ok else "paper has problems")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
