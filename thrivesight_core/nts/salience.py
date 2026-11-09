"""How prominent each element is in a text, as cues that choose the next question.

A model proposes weights and self-term bases; code checks them; the person confirms
anything kept. Nothing here is a finding about the person (lean/NTS/State.lean).
"""
from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).parent
STRUCTURE = json.loads((HERE / "structure.json").read_text())
PRIMES = [p["id"] for p in STRUCTURE["primes"]]
INTRODUCTION = STRUCTURE["introduction_order"]
DOMAINS = {d["id"]: d for d in STRUCTURE["state"]["domains"]}


@dataclass(frozen=True)
class SelfTerm:
    quote: str
    term: str
    whose: str
    basis: str


@dataclass(frozen=True)
class Salience:
    weights: dict[str, float]
    self_terms: list[SelfTerm]


def prompt() -> str:
    return (HERE / "prompts" / "salience.md").read_text()


def parse(text: str, answer: str) -> tuple[Salience | None, list[str]]:
    """Check a model's answer. Returns the salience, or None, and every problem found."""
    problems: list[str] = []
    try:
        raw = json.loads(answer)
    except json.JSONDecodeError:
        return None, ["not JSON"]
    w = raw.get("weights") if isinstance(raw, dict) else None
    if not isinstance(w, dict):
        return None, ["no weights"]
    weights: dict[str, float] = {}
    for p in PRIMES:
        v = w.get(p)
        if not isinstance(v, (int, float)) or isinstance(v, bool) or not 0 <= v <= 1:
            problems.append(f"weight for {p} missing or outside 0..1")
            continue
        weights[p] = float(v)
    extra = set(w) - set(PRIMES)
    if extra:
        problems.append("unknown elements: " + ", ".join(sorted(extra)))
    if len(weights) != len(PRIMES):
        return None, problems
    terms: list[SelfTerm] = []
    for t in raw.get("self_terms") or []:
        if not isinstance(t, dict):
            problems.append("self-term is not an object")
            continue
        quote, term, whose, basis = (str(t.get(k, "")) for k in ("quote", "term", "whose", "basis"))
        if not quote or quote not in text:
            problems.append(f"self-term quote not in text: {quote!r}")
        elif whose not in ("self", "other"):
            problems.append(f"self-term whose must be self or other: {whose!r}")
        elif basis not in PRIMES or basis == "self-differentiation":
            problems.append(f"self-term basis is not another element: {basis!r}")
        else:
            terms.append(SelfTerm(quote, term, whose, basis))
    return Salience(weights, terms), problems


def next_focus(s: Salience, asked: set[str]) -> str | None:
    """The element to ask about next: the most prominent one not yet asked, ties in the
    introduction order (body first). Deterministic given the weights."""
    open_ = [p for p in INTRODUCTION if p not in asked]
    if not open_:
        return None
    return max(open_, key=lambda p: (s.weights[p], -INTRODUCTION.index(p)))


def proper(domain: str) -> list[str]:
    return DOMAINS[domain]["proper"]


def mismatch(asked: bool, domain: str, basis: str) -> str | None:
    """Silent unless the person asked; then only an open question."""
    if asked and basis not in proper(domain):
        return "What tells you that?"
    return None


def read_ollama(text: str, model: str, host: str) -> tuple[Salience | None, list[str]]:
    body = json.dumps({
        "model": model, "stream": False, "think": False, "format": "json", "options": {"temperature": 0},
        "messages": [{"role": "system", "content": prompt()}, {"role": "user", "content": text}],
    }).encode()
    headers = {"Content-Type": "application/json"}
    if os.environ.get("OLLAMA_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["OLLAMA_API_KEY"]
    req = urllib.request.Request(f"{host}/api/chat", data=body, headers=headers)
    with urllib.request.urlopen(req, timeout=600) as r:
        return parse(text, json.loads(r.read())["message"]["content"])


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Read element salience for texts with an Ollama model.")
    ap.add_argument("--ollama", default="gpt-oss:120b")
    ap.add_argument("file", nargs="?", default=str(HERE / "fixtures" / "salience_texts.json"))
    args = ap.parse_args()
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    out = []
    for item in json.loads(Path(args.file).read_text())["texts"]:
        s, problems = read_ollama(item["text"], args.ollama, host)
        top = sorted(s.weights.items(), key=lambda kv: -kv[1])[:4] if s else []
        out.append({"id": item["id"], "top": top, "self_terms": [t.__dict__ for t in s.self_terms] if s else [],
                    "next_focus": next_focus(s, set()) if s else None, "problems": problems})
    print(json.dumps({"model": args.ollama, "readings": out}, indent=1))
