"""Run the reader prompt over a file of entries and print proposals (and agreement, if labelled).

  python -m thrivesight_core.nts.run_reader --model <model id> [entries.json]
  python -m thrivesight_core.nts.run_reader --ollama qwen3.8 [entries.json]

--model uses the Anthropic API and needs ANTHROPIC_API_KEY. --ollama uses a local
Ollama server (OLLAMA_HOST, default http://localhost:11434) and needs no key. person_labels in the file is a list of prime ids per entry,
filled in by the person; entries without labels are read but not scored.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .agreement import agreement
import os

from .reader import read_entry, read_entry_ollama

HERE = Path(__file__).parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("entries", nargs="?", default=str(HERE / "fixtures" / "sample_entries.json"))
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", help="Anthropic model id")
    group.add_argument("--ollama", help="local Ollama model name")
    args = ap.parse_args()

    if args.ollama:
        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        if not host.startswith("http"):
            host = "http://" + host
        read = lambda text: read_entry_ollama(text, args.ollama, host)
    else:
        import anthropic  # imported here so the rest of the package works without it

        client = anthropic.Anthropic()
        read = lambda text: read_entry(text, client, args.model)
    entries = json.loads(Path(args.entries).read_text())["entries"]
    model_sets, person_sets, out = {}, {}, []
    for e in entries:
        proposals, refused = read(e["text"])
        model_sets[e["id"]] = {p.scope.steps[0] for p in proposals}
        if e.get("person_labels") is not None:
            person_sets[e["id"]] = set(e["person_labels"])
        out.append({"id": e["id"], "proposals": [
            {"scope": str(p.scope), "amplitude": p.amplitude, "quote": p.quote, "note": p.note} for p in proposals
        ], "refused": refused})
    print(json.dumps({"readings": out, "agreement": agreement(model_sets, person_sets) if person_sets else None}, indent=2))


if __name__ == "__main__":
    main()
