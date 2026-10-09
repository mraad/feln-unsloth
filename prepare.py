"""FELN.json -> Unsloth decision rows: which layers a NorthSea question uses.

One choice question. Each option is the primary layer (layers[0], the one returned)
plus the unordered set of secondary layers, so 3 layers give 12 options.

    python3 prepare.py ~/Documents/ArcGIS/Projects/NorthSea/FELN.json data/
"""

import json
import random
import re
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

LAYERS = ("Wells", "Pipelines", "Discoveries")
SEED = 3407

INSTRUCTIONS = (
    "Which North Sea layers does this map query need? The first layer is the one whose "
    "features are returned; any other layers only filter it by distance or containment. "
    "Wells are typed dry, oil, gas, oil/gas, gas/condensate, shows, water or salt. "
    "Pipelines are typed oil, gas, condensate, injection or unknown. "
    "Discoveries are typed oil, gas, oil/gas, gas/condensate, condensate or unknown, "
    "and have field labels."
)


def label(layers: list[str]) -> str:
    return "+".join([layers[0], *sorted(layers[1:])])


def options() -> dict[str, str]:
    out = {}
    for primary in LAYERS:
        rest = [x for x in LAYERS if x != primary]
        for n in range(len(rest) + 1):
            for sec in combinations(rest, n):
                key = label([primary, *sec])
                if not sec:
                    out[key] = f"Return {primary.lower()} only; no other layer is involved."
                else:
                    by = " and ".join(s.lower() for s in sorted(sec))
                    out[key] = f"Return {primary.lower()}, constrained by {by}."
    return out


CRITERIA = options()
QUESTIONS = {"layers": {"type": "choice", "instructions": INSTRUCTIONS, "criteria": CRITERIA}}


def to_row(rec: dict) -> dict:
    return {"state": rec["text"], "questions": QUESTIONS, "gold": {"layers": label(rec["meta"]["layers"])}}


def split(rows: list[dict]) -> dict[str, list[dict]]:
    """Stratified 80/10/10 by label."""
    by = defaultdict(list)
    for r in rows:
        by[r["gold"]["layers"]].append(r)
    out = {"train": [], "val": [], "test": []}
    rng = random.Random(SEED)
    for key in sorted(by):
        group = by[key]
        rng.shuffle(group)
        n = len(group)
        a, b = round(n * 0.8), round(n * 0.9)
        out["train"] += group[:a]
        out["val"] += group[a:b]
        out["test"] += group[b:]
    for part in out.values():
        rng.shuffle(part)
    return out


_KW = {"Wells": r"\bwell", "Pipelines": r"\bpipe", "Discoveries": r"\bdiscover"}


def keyword_guess(text: str) -> str:
    """Layer names mentioned in the text; first mention is primary. Fallback: Wells."""
    t = text.lower()
    hits = {k: m.start() for k, p in _KW.items() if (m := re.search(p, t))}
    if not hits:
        return "Wells"
    return label(sorted(hits, key=hits.__getitem__))


def main(src: str, out: str) -> None:
    recs = json.loads(Path(src).read_text())
    rows = [to_row(r) for r in recs]
    assert len(CRITERIA) == 12 and all(r["gold"]["layers"] in CRITERIA for r in rows)
    assert len({r["state"] for r in rows}) == len(rows), "duplicate question text"
    parts = split(rows)
    assert sum(map(len, parts.values())) == len(rows)
    assert not ({r["state"] for r in parts["train"]} & {r["state"] for r in parts["test"]})

    Path(out).mkdir(parents=True, exist_ok=True)
    for name, part in parts.items():
        with open(Path(out) / f"{name}.jsonl", "w") as f:
            for r in part:
                f.write(json.dumps(r) + "\n")
        print(f"{name}: {len(part)}")

    test = parts["test"]
    prior = Counter(r["gold"]["layers"] for r in parts["train"]).most_common(1)[0][0]
    for name, guess in (("prior", lambda _: prior), ("keyword", lambda r: keyword_guess(r["state"]))):
        acc = sum(guess(r) == r["gold"]["layers"] for r in test) / len(test)
        print(f"baseline {name} test acc: {acc:.3f}")


if __name__ == "__main__":
    assert label(["Wells", "Pipelines", "Discoveries"]) == "Wells+Discoveries+Pipelines"
    assert keyword_guess("Find pipelines near wells") == "Pipelines+Wells"
    main(*sys.argv[1:3])
