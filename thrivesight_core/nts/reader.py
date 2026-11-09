"""Build the reader prompt from the table, and turn a model's answer into checked proposals.

The model proposes; this module refuses any proposal that does not hold: an unknown
element, an identity offered as a finding, a bad scope path, or a quote that is not in
the entry word for word (citation fails closed).
"""
from __future__ import annotations

import json
import os
import re
import urllib.request
from pathlib import Path

from .records import Proposal, RealityTags, ScopePath
from .table import load_table

HERE = Path(__file__).parent


def categories_section() -> str:
    table = load_table()
    lines = []
    for c in table.data["categories"]:
        lines.append(f"{c['name']}: {c['function']}. {c['pairing']}.")
        for n in c["elements"]:
            e = table.element(n)
            lines.append(f"  - {e['name']} ({e['number']}, id \"{e['id']}\"): {e['description']}")
    return "\n".join(lines)


def system_prompt() -> str:
    return (HERE / "prompts" / "reader.md").read_text().replace("{categories}", categories_section())


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def parse_answer(entry_text: str, answer: str) -> tuple[list[Proposal], list[str]]:
    """Return (accepted proposals, reasons for each refused one)."""
    match = re.search(r"\{.*\}", answer, re.S)
    if not match:
        return [], ["answer has no JSON object"]
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        return [], [f"answer is not JSON: {exc}"]
    accepted, refused = [], []
    for raw in data.get("proposals", []):
        try:
            scope = ScopePath.parse(str(raw["scope"]))
            if _norm(raw["quote"]) not in _norm(entry_text):
                raise ValueError(f"quote not found in entry: {raw['quote']!r}")
            reality = RealityTags(**{k: v for k, v in (raw.get("reality") or {}).items() if v is not None})
            accepted.append(Proposal(scope, float(raw["amplitude"]), raw["quote"], reality, raw.get("note", "")))
        except (KeyError, ValueError, TypeError) as exc:
            refused.append(f"{raw.get('scope', '?')}: {exc}")
    return accepted, refused


def read_entry(entry_text: str, client, model: str) -> tuple[list[Proposal], list[str]]:
    """Ask a model to read one entry. `client` is an anthropic.Anthropic instance."""
    message = client.messages.create(
        model=model,
        max_tokens=2000,
        system=system_prompt(),
        messages=[{"role": "user", "content": entry_text}],
    )
    text = "".join(block.text for block in message.content if getattr(block, "type", "") == "text")
    return parse_answer(entry_text, text)


def read_entry_ollama(entry_text: str, model: str, host: str = "http://localhost:11434") -> tuple[list[Proposal], list[str]]:
    """Ask an Ollama model to read one entry: a local server, or Ollama Cloud
    (host="https://ollama.com") with OLLAMA_API_KEY set."""
    body = json.dumps({
        "model": model,
        "stream": False,
        "think": False,
        "format": "json",
        "options": {"temperature": 0},
        "messages": [
            {"role": "system", "content": system_prompt()},
            {"role": "user", "content": entry_text},
        ],
    }).encode()
    headers = {"Content-Type": "application/json"}
    if os.environ.get("OLLAMA_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["OLLAMA_API_KEY"]
    request = urllib.request.Request(f"{host}/api/chat", data=body, headers=headers)
    with urllib.request.urlopen(request, timeout=600) as response:
        text = json.loads(response.read())["message"]["content"]
    return parse_answer(entry_text, text)
