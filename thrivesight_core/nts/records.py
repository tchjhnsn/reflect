"""Record types for the element model: scope paths, reality tags, Memory log entries, proposals.

Memory (0) is the append-only log of LogEntry records. Image (1) is the person id on
every entry. A proposal is what a model suggests; it becomes a LogEntry only when the
person confirms it.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Literal

from .table import load_table

Pole = Literal["shared", "inner", "embodied", "imagined", "present", "represented", "voluntary", "involuntary"]


@dataclass(frozen=True)
class ScopePath:
    """An ordered path into the fractal: ScopePath(("emotion", "will")) is Emotion[Will].

    Order matters (Part XXIII): Emotion[Will], willing to feel, is not Will[Emotion],
    feeling a will. The first step must be a prime, because neither identity has an
    interior. Memory may appear only as the last step, as the register of the scope
    before it (Emotion[Memory] is "remembering a feeling"). Image never appears: it is
    the person, present at every depth.
    """

    steps: tuple[str, ...]

    def __post_init__(self) -> None:
        table = load_table()
        if not self.steps:
            raise ValueError("a scope path has at least one step")
        for i, step in enumerate(self.steps):
            kind = table.element(step)["kind"]
            if step == "image":
                raise ValueError("Image is the person at every depth, not a scope step")
            if kind != "prime" and (i == 0 or i != len(self.steps) - 1):
                raise ValueError("Memory has no interior: it can only end a path, after a prime")

    @classmethod
    def parse(cls, text: str) -> "ScopePath":
        """Parse 'emotion[will][memory]' into ('emotion', 'will', 'memory')."""
        head, *rest = text.replace("]", "").split("[")
        return cls(tuple(s.strip() for s in [head, *rest]))

    def __str__(self) -> str:
        return self.steps[0] + "".join(f"[{s}]" for s in self.steps[1:])


@dataclass(frozen=True)
class RealityTags:
    """The four Reality Field coordinates of an experience (Part VI). None means not stated."""

    shared_inner: Literal["shared", "inner"] | None = None
    embodied_imagined: Literal["embodied", "imagined"] | None = None
    present_represented: Literal["present", "represented"] | None = None
    voluntary_involuntary: Literal["voluntary", "involuntary"] | None = None

    def __post_init__(self) -> None:
        axes = {a["id"]: a["poles"] for a in load_table().data["reality_axes"]}
        for axis, poles in axes.items():
            value = getattr(self, axis)
            if value is not None and value not in poles:
                raise ValueError(f"{axis} must be one of {poles}")


@dataclass(frozen=True)
class Proposal:
    """A model's reading of one element in one entry. Never written to Memory by itself."""

    scope: ScopePath
    amplitude: float  # 0 to 1: how strongly the element operates in this entry
    quote: str  # the words in the entry that show it
    reality: RealityTags = field(default_factory=RealityTags)
    note: str = ""

    def __post_init__(self) -> None:
        if not 0 <= self.amplitude <= 1:
            raise ValueError("amplitude is between 0 and 1")
        if not self.quote.strip():
            raise ValueError("a proposal quotes the entry (citation fails closed)")


@dataclass(frozen=True)
class LogEntry:
    """One record in Memory (0): append-only, weighted, owned by a person (Image, 1)."""

    person: str  # Image: the one key that never changes
    at: str  # ISO 8601 time of the experience
    entry_id: str  # the source entry this reading came from
    scope: ScopePath
    weight: float  # Axiom 6: the magnitude stored with the content
    reality: RealityTags
    confirmed_by: str  # a person; a model never confirms
    quote: str

    def to_json(self) -> dict:
        d = asdict(self)
        d["scope"] = str(self.scope)
        return d
