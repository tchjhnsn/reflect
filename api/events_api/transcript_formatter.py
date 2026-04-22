"""
ThriveSight Transcript Formatter — Converts ElevenLabs diarized output
into the labeled dialogue format that ConversationParser already handles.

ElevenLabs Scribe v2 returns segments like:
    [{"speaker": "speaker_0", "text": "Hello", "start": 0.0, "end": 1.2}, ...]

ConversationParser._try_labeled_parse() expects:
    "Speaker 0: Hello\nSpeaker 1: What?\n..."

This module bridges the two formats.
"""

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def merge_consecutive_segments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Merge adjacent segments from the same speaker into single turns.

    ElevenLabs may split one person's utterance into multiple segments
    (e.g., due to pauses). This merges them for cleaner turn boundaries.

    Args:
        segments: Raw segments from ElevenLabs API

    Returns:
        Merged segments where consecutive same-speaker segments are combined
    """
    if not segments:
        return []

    merged = []
    current = None

    for seg in segments:
        speaker = seg.get("speaker", "unknown")
        text = seg.get("text", "").strip()

        if not text:
            continue

        if current is None:
            current = {
                "speaker": speaker,
                "text": text,
                "start": seg.get("start", 0.0),
                "end": seg.get("end", 0.0),
            }
        elif speaker == current["speaker"]:
            # Same speaker — merge
            current["text"] += " " + text
            current["end"] = seg.get("end", current["end"])
        else:
            # Different speaker — flush current, start new
            merged.append(current)
            current = {
                "speaker": speaker,
                "text": text,
                "start": seg.get("start", 0.0),
                "end": seg.get("end", 0.0),
            }

    if current:
        merged.append(current)

    logger.info(
        f"Merged {len(segments)} raw segments into {len(merged)} turns"
    )
    return merged


def format_diarized_to_labeled(
    segments: list[dict[str, Any]],
    speaker_names: Optional[dict[str, str]] = None,
) -> str:
    """
    Convert ElevenLabs diarized segments to labeled dialogue format.

    Args:
        segments: Segments from ElevenLabs Scribe v2 response
        speaker_names: Optional mapping of speaker IDs to display names
                      e.g., {"speaker_0": "Alex", "speaker_1": "Jordan"}

    Returns:
        Labeled dialogue string:
            "Speaker 0: Hello, how are you?
             Speaker 1: I'm fine, thanks."
    """
    merged = merge_consecutive_segments(segments)

    if not speaker_names:
        speaker_names = {}

    lines = []
    for seg in merged:
        raw_speaker = seg.get("speaker", "unknown")
        # Map speaker IDs to human-readable names
        display_name = speaker_names.get(
            raw_speaker,
            _default_speaker_name(raw_speaker),
        )
        text = seg.get("text", "").strip()
        if text:
            lines.append(f"{display_name}: {text}")

    result = "\n".join(lines)
    logger.info(
        f"Formatted transcript: {len(lines)} turns, {len(result)} chars"
    )
    return result


def build_entity_summary(
    entities: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Structure entity detection results for frontend display.

    ElevenLabs returns entities like:
        [{"text": "John", "type": "person_name", "start": 10, "end": 14, ...}]

    We restructure into a format the frontend can render as highlights.

    Args:
        entities: Raw entities from ElevenLabs API

    Returns:
        Structured entity list with display-friendly types and categories
    """
    if not entities:
        return []

    # Map ElevenLabs entity types to display categories
    ENTITY_CATEGORIES = {
        "person_name": {"category": "person", "color": "#e3f2fd", "label": "Person"},
        "organization": {"category": "org", "color": "#f3e5f5", "label": "Organization"},
        "location": {"category": "place", "color": "#e8f5e9", "label": "Location"},
        "address": {"category": "place", "color": "#e8f5e9", "label": "Address"},
        "phone_number": {"category": "pii", "color": "#fff3e0", "label": "Phone"},
        "email_address": {"category": "pii", "color": "#fff3e0", "label": "Email"},
        "date_of_birth": {"category": "pii", "color": "#fff3e0", "label": "Date of Birth"},
        "ssn": {"category": "pii", "color": "#ffebee", "label": "SSN"},
        "credit_card": {"category": "pci", "color": "#ffebee", "label": "Credit Card"},
        "date": {"category": "other", "color": "#f5f5f5", "label": "Date"},
        "time": {"category": "other", "color": "#f5f5f5", "label": "Time"},
        "money": {"category": "other", "color": "#f5f5f5", "label": "Money"},
    }

    structured = []
    for entity in entities:
        entity_type = entity.get("type", "unknown")
        meta = ENTITY_CATEGORIES.get(entity_type, {
            "category": "other",
            "color": "#f5f5f5",
            "label": entity_type.replace("_", " ").title(),
        })

        structured.append({
            "text": entity.get("text", ""),
            "type": entity_type,
            "category": meta["category"],
            "label": meta["label"],
            "color": meta["color"],
            "start": entity.get("start", 0),
            "end": entity.get("end", 0),
            "is_pii": meta["category"] in ("pii", "pci"),
        })

    # Deduplicate by text + type
    seen = set()
    unique = []
    for e in structured:
        key = (e["text"].lower(), e["type"])
        if key not in seen:
            seen.add(key)
            unique.append(e)

    logger.info(
        f"Processed {len(entities)} entities into {len(unique)} unique entries"
    )
    return unique


def extract_speakers_from_segments(
    segments: list[dict[str, Any]],
) -> list[str]:
    """
    Extract unique speaker IDs from segments in order of appearance.

    Args:
        segments: Segments from ElevenLabs

    Returns:
        List of speaker IDs, e.g., ["speaker_0", "speaker_1"]
    """
    speakers = []
    seen = set()
    for seg in segments:
        speaker = seg.get("speaker")
        if speaker and speaker not in seen:
            speakers.append(speaker)
            seen.add(speaker)
    return speakers


def _default_speaker_name(speaker_id: str) -> str:
    """
    Convert ElevenLabs speaker ID to a human-readable name.

    "speaker_0" → "Speaker 0"
    "speaker_1" → "Speaker 1"
    """
    if speaker_id.startswith("speaker_"):
        try:
            idx = int(speaker_id.split("_")[1])
            return f"Speaker {idx}"
        except (IndexError, ValueError):
            pass
    return speaker_id.replace("_", " ").title()
