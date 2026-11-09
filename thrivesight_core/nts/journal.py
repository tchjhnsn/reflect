"""The journal: the person's own words, with no model.

The only prompt is a fixed "tell me more", shown after each thing the person writes until
they say they are done. Then the entry closes, or the person chooses to reflect on it.
Nothing is read, tagged or stored except what they wrote.
"""
from __future__ import annotations

from dataclasses import dataclass, field

OPENING = "What would you like to remember?"
TELL_ME_MORE = "Tell me more."
CLOSED = "Saved. You can leave it here, or reflect on it."


@dataclass
class Entry:
    id: str
    person: str
    started_at: str
    parts: list[str] = field(default_factory=list)
    closed: bool = False
    show_cues: bool = False  # cues stay internal unless the person asks, entry by entry
    name_emotions: bool | None = None  # None: use the person's usual choice

    @property
    def text(self) -> str:
        return "\n\n".join(self.parts)

    def prompt(self) -> str:
        """What the journal shows next. Deterministic: no model."""
        if self.closed:
            return CLOSED
        return OPENING if not self.parts else TELL_ME_MORE

    def write(self, words: str) -> str:
        if self.closed:
            raise ValueError("this entry is closed")
        if words.strip():
            self.parts.append(words.strip())
        return self.prompt()

    def done(self) -> str:
        self.closed = True
        return self.prompt()
