"""Onboarding: a few questions, each answered or skipped by the person. No model runs.

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
HOLDS: dict[str, dict] = {h["id"]: h for h in STRUCTURE["holds"]}

# Fixed choices. The person picks one; the ids are what is stored.
CHOICES: dict[str, tuple[str, ...]] = {
    "marital_status": ("single", "dating", "partnered", "married", "separated", "divorced", "widowed", "rather_not_say"),
    "seeing_someone": ("yes", "no", "complicated"),
}


class AnswerError(ValueError):
    def __init__(self, question: str, message: str):
        super().__init__(f"{question}: {message}")
        self.question = question


def _clean(question: str, text: str, today: date) -> str | None:
    text = " ".join(text.split())
    if not text:
        return None
    if question in CHOICES:
        if text not in CHOICES[question]:
            raise AnswerError(question, "pick one of the choices")
        return text
    if question == "children":
        if not text.isdigit() or int(text) > 30:
            raise AnswerError(question, "write a number")
        return str(int(text))
    if question.startswith("hold_"):
        return clean_hold(question, text)
    if question == "date_of_birth":
        try:
            born = date.fromisoformat(text)
        except ValueError:
            raise AnswerError(question, "write the date as YYYY-MM-DD") from None
        if born > today:
            raise AnswerError(question, "a date of birth cannot be in the future")
    return text


def clean_hold(question: str, text: str) -> str:
    """A hold is 'id,id' or 'id,id; own words'. 'Not for me' and 'unsure' stand alone."""
    picks, _, words = text.partition(";")
    ids = [p.strip() for p in picks.split(",") if p.strip()]
    if not ids or any(i not in HOLDS for i in ids) or len(set(ids)) != len(ids):
        raise AnswerError(question, "pick at least one of the choices")
    alone = {"not_for_me", "unsure"}
    if len(ids) > 1 and alone & set(ids):
        raise AnswerError(question, "'not for me' and 'unsure' stand alone")
    words = " ".join(words.split())
    return ",".join(ids) + (f"; {words}" if words else "")


def hold_stances(answer: str) -> dict[str, str]:
    """The stance of each hold in an answer, from Lean: duty is normative."""
    return {i: HOLDS[i]["stance"] for i in answer.split(";")[0].split(",")}


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
            elif dest["store"] == "holds":
                written.append(("hold", dest["subject"], person.state_belief(value, at, Stated())))
            else:
                raise ValueError(f"unknown store {dest['store']}")
        return written
