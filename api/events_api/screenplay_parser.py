"""
ThriveSight Screenplay Parser — Converts raw screenplay text into structured scenes.

This module handles the first step of the character dataset pipeline:
taking a raw screenplay (from IMSDB, Script Slug, etc.) and decomposing it
into a structured list of scenes with metadata.

A screenplay follows a predictable format:
- Scene headings (sluglines): INT./EXT. LOCATION - TIME
- Action/description lines (narrative)
- Character names (centered, uppercase before dialogue)
- Dialogue lines
- Parentheticals (acting directions within dialogue)

The parser produces a list of Scene objects, each containing:
- scene_number: Sequential ordering
- slugline: The INT./EXT. heading
- location: Extracted location
- time_of_day: DAY/NIGHT/CONTINUOUS/etc.
- characters_present: Who appears in the scene (from dialogue + action mentions)
- dialogue_turns: Structured turns for feeding into the analysis pipeline
- action_text: Non-dialogue narrative (important for contextual metadata)
- witnesses: Characters present but not speaking (your Flash/MJ insight)
"""

import re
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ============================================================================
# SCREENPLAY FORMAT PATTERNS
# ============================================================================

# Scene heading (slugline): INT./EXT. LOCATION - TIME
SLUGLINE_PATTERN = re.compile(
    r"^\s*((?:INT|EXT|INT\./EXT|EXT\./INT|I/E)\.?\s+.+?)$",
    re.MULTILINE | re.IGNORECASE
)

# Character name before dialogue (uppercase, possibly with parenthetical extension)
# e.g., "PETER", "PETER (V.O.)", "MARY JANE (CONT'D)"
CHARACTER_CUE_PATTERN = re.compile(
    r"^\s{10,}([A-Z][A-Z\s\.\-\']+?)(?:\s*\((?:V\.O\.|O\.S\.|CONT\'?D?|O\.C\.)[^)]*\))?\s*$",
    re.MULTILINE
)

# Parenthetical acting direction: (beat), (quietly), etc.
PARENTHETICAL_PATTERN = re.compile(
    r"^\s{10,}\(([^)]+)\)\s*$",
    re.MULTILINE
)

# Dialogue line (indented, follows character cue)
DIALOGUE_LINE_PATTERN = re.compile(
    r"^\s{5,}(.+?)$",
    re.MULTILINE
)

# Transition lines: CUT TO:, DISSOLVE TO:, FADE IN:, etc.
TRANSITION_PATTERN = re.compile(
    r"^\s*((?:CUT|DISSOLVE|FADE|SMASH CUT|MATCH CUT|WIPE)\s+TO:?|FADE (?:IN|OUT)\.?)\s*$",
    re.MULTILINE | re.IGNORECASE
)


# ============================================================================
# SCENE STRUCTURE
# ============================================================================

def create_empty_scene(scene_number: int, slugline: str = "") -> dict[str, Any]:
    """Create a new empty scene dict."""
    location, time_of_day = _parse_slugline(slugline)
    return {
        "scene_number": scene_number,
        "slugline": slugline.strip(),
        "location": location,
        "time_of_day": time_of_day,
        "characters_present": [],
        "dialogue_turns": [],
        "action_lines": [],
        "witnesses": [],  # Characters present but not speaking
        "raw_text": "",
    }


def _parse_slugline(slugline: str) -> tuple[str, str]:
    """Extract location and time of day from a slugline."""
    # Remove INT./EXT. prefix
    cleaned = re.sub(r"^(?:INT|EXT|INT\./EXT|EXT\./INT|I/E)\.?\s+", "", slugline.strip(), flags=re.IGNORECASE)

    # Split on last " - " to get location and time
    parts = cleaned.rsplit(" - ", 1)
    location = parts[0].strip() if parts else cleaned.strip()
    time_of_day = parts[1].strip() if len(parts) > 1 else "UNKNOWN"

    return location, time_of_day


# ============================================================================
# MAIN PARSER
# ============================================================================

class ScreenplayParser:
    """
    Parses raw screenplay text into structured scenes.

    The parser handles multiple screenplay format variations:
    - Standard Hollywood spec format
    - IMSDB HTML-extracted text
    - Script Slug plain text
    - PDF-extracted text (may have odd spacing)

    Usage:
        parser = ScreenplayParser()
        scenes = parser.parse(raw_screenplay_text, film_title="Spider-Man")
    """

    def __init__(self, known_characters: Optional[list[str]] = None):
        """
        Initialize parser with optional known character list.

        Args:
            known_characters: Pre-known character names to help with detection.
                             Useful for screenplays with non-standard formatting.
        """
        self.known_characters = set(c.upper() for c in (known_characters or []))
        self._detected_characters = set()

    def parse(self, raw_text: str, film_title: str = "Untitled") -> dict[str, Any]:
        """
        Parse a raw screenplay into structured scenes.

        Args:
            raw_text: Full screenplay text
            film_title: Title of the film (for metadata)

        Returns:
            Dict with keys:
            - film_title: str
            - total_scenes: int
            - total_dialogue_turns: int
            - characters: list[str] — all characters with dialogue
            - scenes: list[dict] — structured scene objects
            - parse_metadata: parsing stats
        """
        # Pre-process: normalize line endings, strip HTML artifacts
        text = self._preprocess(raw_text)

        # Split into scenes by sluglines
        scenes = self._split_into_scenes(text)

        # Parse each scene's content
        parsed_scenes = []
        total_turns = 0

        for scene in scenes:
            parsed = self._parse_scene_content(scene)
            total_turns += len(parsed["dialogue_turns"])
            parsed_scenes.append(parsed)

        # Compute witnesses for each scene
        self._compute_witnesses(parsed_scenes)

        # Build character list (ordered by dialogue frequency)
        char_counts = {}
        for scene in parsed_scenes:
            for turn in scene["dialogue_turns"]:
                speaker = turn["speaker"]
                char_counts[speaker] = char_counts.get(speaker, 0) + 1

        characters = sorted(char_counts.keys(), key=lambda c: -char_counts[c])

        return {
            "film_title": film_title,
            "total_scenes": len(parsed_scenes),
            "total_dialogue_turns": total_turns,
            "characters": characters,
            "scenes": parsed_scenes,
            "parse_metadata": {
                "raw_length": len(raw_text),
                "preprocessed_length": len(text),
                "detected_characters": len(characters),
                "scenes_with_dialogue": sum(
                    1 for s in parsed_scenes if s["dialogue_turns"]
                ),
            },
        }

    def _preprocess(self, raw_text: str) -> str:
        """Normalize screenplay text."""
        # Normalize line endings
        text = raw_text.replace("\r\n", "\n").replace("\r", "\n")

        # CRITICAL: Expand tabs to spaces (screenplay standard: 1 tab = 8 spaces)
        # Many screenplay sources use tabs for indentation (Daily Script, etc.)
        text = text.expandtabs(8)

        # Remove common HTML artifacts from IMSDB
        text = re.sub(r"</?(?:b|i|u|pre|html|head|body|title|meta|script|style)[^>]*>", "", text, flags=re.IGNORECASE)
        text = re.sub(r"&nbsp;", " ", text)
        text = re.sub(r"&amp;", "&", text)
        text = re.sub(r"&lt;", "<", text)
        text = re.sub(r"&gt;", ">", text)

        # Remove page numbers (standalone numbers on a line)
        text = re.sub(r"^\s*\d{1,3}\s*$", "", text, flags=re.MULTILINE)

        # Collapse excessive blank lines (3+ → 2)
        text = re.sub(r"\n{4,}", "\n\n\n", text)

        return text

    def _split_into_scenes(self, text: str) -> list[dict[str, Any]]:
        """Split text into scenes by sluglines."""
        # Find all slugline positions
        slugline_matches = list(SLUGLINE_PATTERN.finditer(text))

        if not slugline_matches:
            # No sluglines found — treat entire text as one scene
            logger.warning("No sluglines found; treating entire text as one scene")
            scene = create_empty_scene(1, "UNKNOWN LOCATION")
            scene["raw_text"] = text
            return [scene]

        scenes = []
        for i, match in enumerate(slugline_matches):
            scene_number = i + 1
            slugline = match.group(1)
            start = match.start()

            # End is the start of the next slugline, or end of text
            if i + 1 < len(slugline_matches):
                end = slugline_matches[i + 1].start()
            else:
                end = len(text)

            scene = create_empty_scene(scene_number, slugline)
            scene["raw_text"] = text[start:end]
            scenes.append(scene)

        return scenes

    def _parse_scene_content(self, scene: dict[str, Any]) -> dict[str, Any]:
        """Parse the content of a single scene to extract dialogue and action."""
        raw = scene["raw_text"]
        lines = raw.split("\n")

        dialogue_turns = []
        action_lines = []
        characters_present = set()

        current_character = None
        current_dialogue = []
        current_parenthetical = None
        turn_number = 0

        for line in lines:
            stripped = line.strip()

            if not stripped:
                # Blank line — if we were collecting dialogue, finalize it
                if current_character and current_dialogue:
                    turn_number += 1
                    dialogue_turns.append({
                        "turn_number": turn_number,
                        "speaker": current_character,
                        "text": " ".join(current_dialogue).strip(),
                        "parenthetical": current_parenthetical,
                    })
                    current_dialogue = []
                    current_parenthetical = None
                current_character = None
                continue

            # Check if this is a slugline (skip — it's the scene heading)
            if SLUGLINE_PATTERN.match(line):
                continue

            # Check if this is a transition
            if TRANSITION_PATTERN.match(line):
                continue

            # Check if this is a character cue
            char_match = CHARACTER_CUE_PATTERN.match(line)
            if char_match:
                # Finalize any previous dialogue
                if current_character and current_dialogue:
                    turn_number += 1
                    dialogue_turns.append({
                        "turn_number": turn_number,
                        "speaker": current_character,
                        "text": " ".join(current_dialogue).strip(),
                        "parenthetical": current_parenthetical,
                    })
                    current_dialogue = []
                    current_parenthetical = None

                current_character = char_match.group(1).strip()
                characters_present.add(current_character)
                self._detected_characters.add(current_character)
                continue

            # Check if this is a parenthetical
            paren_match = PARENTHETICAL_PATTERN.match(line)
            if paren_match and current_character:
                current_parenthetical = paren_match.group(1).strip()
                continue

            # If we have a current character, this is dialogue
            if current_character:
                # Dialogue lines are typically indented ~25-35 chars
                if len(line) > len(stripped) + 3:  # Some indentation
                    current_dialogue.append(stripped)
                else:
                    # Might be action text that interrupts dialogue
                    if current_dialogue:
                        turn_number += 1
                        dialogue_turns.append({
                            "turn_number": turn_number,
                            "speaker": current_character,
                            "text": " ".join(current_dialogue).strip(),
                            "parenthetical": current_parenthetical,
                        })
                        current_dialogue = []
                        current_parenthetical = None
                    current_character = None
                    action_lines.append(stripped)
            else:
                # Action/description line
                if stripped and not stripped.startswith("("):
                    action_lines.append(stripped)

                    # Check if any known characters are mentioned in action
                    for char in self._detected_characters | self.known_characters:
                        if char in stripped.upper():
                            characters_present.add(char)

        # Finalize any remaining dialogue
        if current_character and current_dialogue:
            turn_number += 1
            dialogue_turns.append({
                "turn_number": turn_number,
                "speaker": current_character,
                "text": " ".join(current_dialogue).strip(),
                "parenthetical": current_parenthetical,
            })

        scene["dialogue_turns"] = dialogue_turns
        scene["action_lines"] = action_lines
        scene["characters_present"] = sorted(characters_present)

        return scene

    def _compute_witnesses(self, scenes: list[dict[str, Any]]):
        """
        Compute witnesses for each scene.

        A witness is a character who is present in the scene (mentioned in action
        lines or characters_present) but does NOT speak in dialogue. This captures
        the exact dynamic you described — Mary Jane witnessing Flash bullying Peter.
        """
        for scene in scenes:
            speakers = set(t["speaker"] for t in scene["dialogue_turns"])
            present = set(scene["characters_present"])
            scene["witnesses"] = sorted(present - speakers)
