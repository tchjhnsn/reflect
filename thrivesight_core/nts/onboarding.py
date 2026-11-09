"""Onboarding: five questions, each answered or skipped by the person. No model runs.

Mirrors lean/NTS/Onboarding.lean. The order and where each answer goes come from
structure.json, which Lean exports. Nothing is written until the person finishes.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from .person import Person, Stated

STRUCTURE = json.loads((Path(__file__).parent / "structure.json").read_text())
ORDER: list[dict] = STRUCTURE["onboarding"]


class AnswerError(ValueError):
    def __init__(self, question: str, message: str):
        super().__init__(f"{question}: {message}")
        self.question = question


def _clean(question: str, text: str, today: date) -> str | None:
    text = " ".join(text.split())
    if not text:
        return None
    if question == "date_of_birth":
        try:
            born = date.fromisoformat(text)
        except ValueError:
            raise AnswerError(question, "write the date as YYYY-MM-DD") from None
        if born > today:
            raise AnswerError(question, "a date of birth cannot be in the future")
    return text


@dataclass
class Onboarding:
    today: date
    step: int = 0
    answers: dict[str, str] = field(default_factory=dict)

    @property
    def question(self) -> str | None:
        return ORDER[self.step]["id"] if self.step < len(ORDER) else None

    def answer(self, text: str) -> None:
        q = self.question
        if q is None:
            raise IndexError("onboarding is finished")
        value = _clean(q, text, self.today)
        if value is None:
            self.answers.pop(q, None)
        else:
            self.answers[q] = value
        self.step += 1

    def skip(self) -> None:
        self.answer("")

    def back(self) -> None:
        self.step = max(0, self.step - 1)

    def finish(self, person: Person, at: str) -> list:
        """Write every answer to its store, as stated by the person."""
        written = []
        for q in ORDER:
            value = self.answers.get(q["id"])
            if value is None:
                continue
            dest = q["destination"]
            if dest["store"] == "profile":
                written.append(person.state_fact(dest["key"], value, at, at, Stated()))
            elif dest["store"] == "relationships":
                written.append(person.state_relationship(value, dest["kind"], at, at))
            elif dest["store"] == "puzzles":
                written.append(person.state_puzzle(value, at))
            else:
                raise ValueError(f"unknown store {dest['store']}")
        return written
