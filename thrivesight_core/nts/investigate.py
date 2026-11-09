"""Guided investigation: cues from language, then a tree of open questions.

The structure (frames, facet order, coercion sources, which facets are normative) comes
from the Lean model, exported to structure.json by `lake exe nts-export` in lean/. This
module only follows it. The model may produce cues and nothing else; questions come from
the question files, and every question is checked for leading or prescriptive wording.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from .reader import categories_section, _norm
from .records import ScopePath

HERE = Path(__file__).parent
STRUCTURE = json.loads((HERE / "structure.json").read_text())
WILL = json.loads((HERE / "questions" / "will.json").read_text())

LEADING = re.compile(
    r"\b(weren'?t|didn'?t|don'?t|wasn'?t|isn'?t) you\b|\byou (must|should|will|would|probably)\b|\bshould you\b",
    re.I,
)


@dataclass(frozen=True)
class Cue:
    element: str
    quote: str
    open: str


@dataclass(frozen=True)
class Ask:
    scope: str
    facet: str
    stance: str  # descriptive or normative, from the Lean model
    text: str


def check_question(text: str) -> None:
    """Refuse a question that is not open, or that assumes, prescribes or predicts."""
    if not text.strip().endswith("?"):
        raise ValueError(f"not a question: {text!r}")
    if LEADING.search(text):
        raise ValueError(f"leading or prescriptive wording: {text!r}")


def cue_prompt() -> str:
    return (HERE / "prompts" / "cues.md").read_text().replace("{categories}", categories_section())


def parse_cues(entry_text: str, answer: str) -> tuple[list[Cue], list[str]]:
    match = re.search(r"\{.*\}", answer, re.S)
    if not match:
        return [], ["answer has no JSON object"]
    primes = {p["id"] for p in STRUCTURE["primes"]}
    cues, refused = [], []
    for raw in json.loads(match.group(0)).get("cues", []):
        el, quote = str(raw.get("element", "")), str(raw.get("quote", ""))
        if el not in primes:
            refused.append(f"{el}: not a prime")
        elif not quote.strip() or _norm(quote) not in _norm(entry_text):
            refused.append(f"{el}: quote not in entry: {quote!r}")
        else:
            cues.append(Cue(el, quote, str(raw.get("open", ""))))
    return cues, refused


def _facet_key(f: dict) -> str:
    return f["facet"].get("prime") or f["facet"]["axis"]


def will_tree(coerced: bool = False) -> list[Ask]:
    """The decision frame's questions in the Lean order; with coerced, the next level too."""
    asks = []
    for f in STRUCTURE["will_order"]:
        key = _facet_key(f)
        scope = "will" if "axis" in f["facet"] else str(ScopePath(("will", key)) if key != "will" else ScopePath(("will", "will")))
        asks.append(Ask(scope, key, f["stance"], WILL["root"][key]))
    if coerced:
        for f in STRUCTURE["coercion_sources"]:
            key = _facet_key(f)
            scope = f"will[{key}]" if "prime" in f["facet"] else "will"
            asks.append(Ask(scope, key, f["stance"], WILL["coercion_sources"][key]))
    for a in asks:
        check_question(a.text)
    return asks


def read_cues_ollama(entry_text: str, model: str, host: str) -> tuple[list[Cue], list[str]]:
    import os
    import urllib.request

    body = json.dumps({
        "model": model, "stream": False, "think": False, "format": "json",
        "options": {"temperature": 0},
        "messages": [{"role": "system", "content": cue_prompt()}, {"role": "user", "content": entry_text}],
    }).encode()
    headers = {"Content-Type": "application/json"}
    if os.environ.get("OLLAMA_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["OLLAMA_API_KEY"]
    req = urllib.request.Request(f"{host}/api/chat", data=body, headers=headers)
    with urllib.request.urlopen(req, timeout=600) as r:
        return parse_cues(entry_text, json.loads(r.read())["message"]["content"])
