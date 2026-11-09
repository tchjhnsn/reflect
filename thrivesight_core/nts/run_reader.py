"""Run the reader prompt over a file of entries and print proposals (and agreement, if labelled).

  python -m thrivesight_core.nts.run_reader --model <model id> [entries.json]

Needs ANTHROPIC_API_KEY. person_labels in the file is a list of prime ids per entry,
filled in by the person; entries without labels are read but not scored.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .agreement import agreement
from .reader import read_entry

HERE = Path(__file__).parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("entries", nargs="?", default=str(HERE / "fixtures" / "sample_entries.json"))
    ap.add_argument("--model", required=True)
    args = ap.parse_args()

    import anthropic  # imported here so the rest of the package works without it

    client = anthropic.Anthropic()
    entries = json.loads(Path(args.entries).read_text())["entries"]
    model_sets, person_sets, out = {}, {}, []
    for e in entries:
        proposals, refused = read_entry(e["text"], client, args.model)
        model_sets[e["id"]] = {p.scope.steps[0] for p in proposals}
        if e.get("person_labels") is not None:
            person_sets[e["id"]] = set(e["person_labels"])
        out.append({"id": e["id"], "proposals": [
            {"scope": str(p.scope), "amplitude": p.amplitude, "quote": p.quote, "note": p.note} for p in proposals
        ], "refused": refused})
    print(json.dumps({"readings": out, "agreement": agreement(model_sets, person_sets) if person_sets else None}, indent=2))


if __name__ == "__main__":
    main()
