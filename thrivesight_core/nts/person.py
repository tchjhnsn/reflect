"""The profile (dated facts), the beliefs store, and the grants that allow items from the journal.

Mirrors lean/NTS/Person.lean. Facts and beliefs are explicit: each is either stated by the
person, or taken from a journal entry under a grant the person gave, quoting the entry. An
item taken from a journal is a proposal until the person confirms it. Nothing is ever
overwritten: a new value for a fact closes the old one.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal

Store = Literal["profile", "beliefs"]


@dataclass(frozen=True)
class Grant:
    id: str
    store: Store
    entries: tuple[str, ...]
    revoked: bool = False


@dataclass(frozen=True)
class Stated:
    pass


@dataclass(frozen=True)
class FromJournal:
    entry: str
    quote: str
    grant: str


Source = Stated | FromJournal


@dataclass(frozen=True)
class Fact:
    key: str
    value: str
    valid_from: str
    recorded_at: str
    source: Source
    status: Literal["proposed", "confirmed"]
    valid_to: str | None = None


@dataclass(frozen=True)
class Belief:
    text: str
    recorded_at: str
    source: Source
    status: Literal["proposed", "confirmed"]


@dataclass
class Person:
    id: str
    grants: dict[str, Grant] = field(default_factory=dict)
    facts: list[Fact] = field(default_factory=list)
    beliefs: list[Belief] = field(default_factory=list)
    journal: dict[str, str] = field(default_factory=dict)  # entry id -> text, for quote checks

    # grants
    def grant(self, gid: str, store: Store, entries: list[str]) -> Grant:
        g = Grant(gid, store, tuple(entries))
        self.grants[gid] = g
        return g

    def revoke(self, gid: str) -> list[Fact | Belief]:
        """Revoke a grant and return every item taken under it, for the person to keep or remove."""
        self.grants[gid] = replace(self.grants[gid], revoked=True)
        taken = [x for x in [*self.facts, *self.beliefs] if isinstance(x.source, FromJournal) and x.source.grant == gid]
        return taken

    def allowed(self, source: Source, store: Store) -> bool:
        if isinstance(source, Stated):
            return True
        g = self.grants.get(source.grant)
        return bool(g and g.store == store and not g.revoked and source.entry in g.entries)

    def _check(self, source: Source, store: Store) -> str:
        if not self.allowed(source, store):
            raise PermissionError(f"no live {store} grant covers this journal entry")
        if isinstance(source, FromJournal):
            text = self.journal.get(source.entry, "")
            if " ".join(source.quote.split()).lower() not in " ".join(text.split()).lower():
                raise ValueError("the quote is not in the entry")
            return "proposed"
        return "confirmed"

    # facts
    def state_fact(self, key: str, value: str, valid_from: str, at: str, source: Source = Stated()) -> Fact:
        status = self._check(source, "profile")
        fact = Fact(key, value, valid_from, at, source, status)
        if status == "confirmed":
            self._close(key, valid_from)
        self.facts.append(fact)
        return fact

    def _close(self, key: str, valid_from: str) -> None:
        for i, f in enumerate(self.facts):
            if f.key == key and f.status == "confirmed" and f.valid_to is None:
                self.facts[i] = replace(f, valid_to=valid_from)

    def confirm_fact(self, fact: Fact) -> Fact:
        if not self.allowed(fact.source, "profile"):
            raise PermissionError("its grant was revoked")
        i = self.facts.index(fact)
        self._close(fact.key, fact.valid_from)
        self.facts[i] = replace(fact, status="confirmed")
        return self.facts[i]

    def current(self, key: str) -> str | None:
        live = [f for f in self.facts if f.key == key and f.status == "confirmed" and f.valid_to is None]
        return live[-1].value if live else None

    # beliefs
    def state_belief(self, text: str, at: str, source: Source = Stated()) -> Belief:
        b = Belief(text, at, source, self._check(source, "beliefs"))
        self.beliefs.append(b)
        return b

    def confirm_belief(self, belief: Belief) -> Belief:
        if not self.allowed(belief.source, "beliefs"):
            raise PermissionError("its grant was revoked")
        i = self.beliefs.index(belief)
        self.beliefs[i] = replace(belief, status="confirmed")
        return self.beliefs[i]
