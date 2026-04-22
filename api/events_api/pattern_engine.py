from collections import defaultdict
from typing import Any
import logging

from .models import Event
from . import llm_client
from . import validators

logger = logging.getLogger(__name__)

SNIPPET_LENGTH = 160
MIN_EVIDENCE_PER_PATTERN = 1
MAX_EVIDENCE_PER_PATTERN = 10


def clean_label(value: str) -> str:
    return value.strip().lower() if value and value.strip() else "unknown"


def display_label(value: str) -> str:
    return value.title() if value != "unknown" else "Unknown"


def primary_tag(event: Event) -> str:
    if isinstance(event.context_tags, list):
        for tag in event.context_tags:
            if isinstance(tag, str) and tag.strip():
                return clean_label(tag)
    return "unknown"


def event_emotion(event: Event) -> str:
    return clean_label(event.emotion or "")


def event_key(event: Event) -> str:
    return f"{primary_tag(event)}|{event_emotion(event)}"


def build_snippet(text: str) -> str:
    collapsed = " ".join((text or "").split())
    return collapsed[:SNIPPET_LENGTH]


def recompute_patterns_v0(
    events: list[Event], *, max_patterns: int, evidence_per_pattern: int
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Event]] = defaultdict(list)
    for event in events:
        grouped[event_key(event)].append(event)

    ranked_groups: list[tuple[str, list[Event], float, float]] = []
    for key, grouped_events in grouped.items():
        intensities = [event.intensity for event in grouped_events if event.intensity is not None]
        avg_intensity = sum(intensities) / len(intensities) if intensities else None
        score = len(grouped_events) * (avg_intensity if avg_intensity is not None else 1.0)
        most_recent_timestamp = grouped_events[0].occurred_at.timestamp()
        ranked_groups.append((key, grouped_events, score, most_recent_timestamp))

    ranked_groups.sort(key=lambda item: (-item[2], -item[3], item[0]))
    selected_groups = ranked_groups[:max_patterns]

    bounded_evidence = min(
        max(evidence_per_pattern, MIN_EVIDENCE_PER_PATTERN),
        MAX_EVIDENCE_PER_PATTERN,
    )

    patterns: list[dict[str, Any]] = []
    for key, grouped_events, score, _ in selected_groups:
        tag, emotion = key.split("|", maxsplit=1)
        tag_label = display_label(tag)
        emotion_label = display_label(emotion)
        evidence_events = grouped_events[:bounded_evidence]
        relevance_denominator = max(len(evidence_events), 1)

        evidence = []
        for index, event in enumerate(evidence_events):
            relevance = round((relevance_denominator - index) / relevance_denominator, 3)
            evidence.append(
                {
                    "event_id": str(event.id),
                    "occurred_at": event.occurred_at.isoformat(),
                    "snippet": build_snippet(event.text),
                    "relevance": relevance,
                }
            )

        patterns.append(
            {
                "key": key,
                "name": f"{tag_label} + {emotion_label} pattern",
                "hypothesis": (
                    f"In {tag_label} contexts, {emotion_label} events recur and often lead "
                    "to similar reactions/outcomes."
                ),
                "score": round(score, 4),
                "evidence": evidence,
            }
        )

    return patterns


# ============================================================================
# PATTERN DETECTION FROM CONVERSATION SIGNALS
# ============================================================================

def detect_conversation_patterns(signals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Detect conversation patterns from behavioral signals.

    This function takes signals from the conversation analysis pipeline and:
    1. Groups signals by (trigger_category + response_emotion)
    2. Scores each group: occurrence_count × average_intensity
    3. Filters for minimum evidence threshold (2 occurrences)
    4. Calls LLM to generate pattern names and hypotheses
    5. Validates output against Pattern Schema
    6. Falls back to deterministic naming if LLM fails

    Args:
        signals: List of Signal Schema objects with keys:
                - turn_number: int
                - speaker: str
                - emotion: str
                - intensity: float (1.0-5.0)
                - reaction: str
                - trigger_action: dict with 'category', 'action_text'
                - signal_address: str

    Returns:
        List of Pattern Schema objects with keys:
        - pattern_name: str
        - hypothesis: str
        - score: float (occurrence_count × average_intensity)
        - evidence: list[dict]
        - trigger_category: str
        - response_emotion: str
        - occurrence_count: int
    """
    if not signals:
        logger.warning("No signals provided for pattern detection")
        return []

    # Minimum evidence threshold (from constrained-generation.md)
    MIN_EVIDENCE_THRESHOLD = 2

    # Step 1: Group signals by (trigger_category, response_emotion)
    pattern_groups = defaultdict(list)

    for signal in signals:
        trigger_action = signal.get("trigger_action", {})
        trigger_category = trigger_action.get("category", "unknown")
        response_emotion = signal.get("emotion", "unknown")

        # Create pattern key
        pattern_key = (trigger_category, response_emotion)
        pattern_groups[pattern_key].append(signal)

    # Step 2: Score each pattern and filter by minimum threshold
    ranked_patterns = []

    for (trigger_category, response_emotion), group_signals in pattern_groups.items():
        occurrence_count = len(group_signals)

        # Filter: minimum 2 occurrences
        if occurrence_count < MIN_EVIDENCE_THRESHOLD:
            logger.debug(
                f"Pattern {trigger_category}→{response_emotion} has only "
                f"{occurrence_count} occurrence(s); skipping"
            )
            continue

        # Calculate average intensity
        intensities = [
            s.get("intensity", 1.0) for s in group_signals
            if s.get("intensity") is not None
        ]
        average_intensity = sum(intensities) / len(intensities) if intensities else 1.0

        # Compute score
        score = occurrence_count * average_intensity

        # Build evidence array
        evidence = []
        for signal in group_signals:
            evidence_item = {
                "turn_number": signal.get("turn_number"),
                "signal_address": signal.get("signal_address", "SA(*,*,*,*)"),
                "text_excerpt": _build_signal_excerpt(signal),
            }
            evidence.append(evidence_item)

        ranked_patterns.append(
            {
                "pattern_key": f"{trigger_category}|{response_emotion}",
                "trigger_category": trigger_category,
                "response_emotion": response_emotion,
                "occurrence_count": occurrence_count,
                "average_intensity": average_intensity,
                "score": score,
                "evidence": evidence,
            }
        )

    # Step 3: Sort by score (descending)
    ranked_patterns.sort(key=lambda p: p["score"], reverse=True)

    # Step 4: Call LLM to generate pattern names and hypotheses
    try:
        named_patterns = _generate_pattern_names_and_hypotheses(ranked_patterns)
    except Exception as e:
        logger.warning(f"LLM pattern naming failed ({type(e).__name__}: {e}); using deterministic fallback")
        named_patterns = _generate_deterministic_pattern_names(ranked_patterns)

    # Step 5: Validate all patterns
    validation_errors = validators.validate_patterns(named_patterns)
    if validation_errors:
        logger.warning(f"Pattern validation warnings: {validation_errors}")

    return named_patterns


def _build_signal_excerpt(signal: dict[str, Any]) -> str:
    """
    Build a text excerpt from a signal's trigger action for evidence.

    Args:
        signal: Signal object

    Returns:
        Text excerpt string
    """
    trigger_action = signal.get("trigger_action", {})
    action_text = trigger_action.get("action_text", "")

    if action_text:
        # Truncate to reasonable length for evidence display
        return action_text[:200]
    else:
        return f"Signal from {signal.get('speaker', 'unknown')} at turn {signal.get('turn_number')}"


def _generate_pattern_names_and_hypotheses(
    patterns: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Call LLM to generate human-readable names and testable hypotheses for patterns.

    Args:
        patterns: List of raw pattern dicts with keys:
                 - pattern_key: str (trigger_category|response_emotion)
                 - trigger_category: str
                 - response_emotion: str
                 - occurrence_count: int
                 - score: float
                 - evidence: list[dict]

    Returns:
        List of Pattern Schema objects with pattern_name and hypothesis added

    Raises:
        ValueError: If LLM fails
    """
    if not patterns:
        return []

    # Build prompt for LLM
    pattern_descriptions = []
    for p in patterns:
        key = p.get("pattern_key", "unknown")
        occurrences = p.get("occurrence_count", 0)
        pattern_descriptions.append(
            f"- Trigger: {p['trigger_category']}, Response: {p['response_emotion']} "
            f"(observed {occurrences} times, score {p['score']:.1f})"
        )

    patterns_text = "\n".join(pattern_descriptions)

    user_prompt = (
        f"Generate human-readable pattern names and hypotheses for these detected patterns:\n\n"
        f"{patterns_text}\n\n"
        f"For each pattern, provide a name that describes the DYNAMIC (not blaming anyone) "
        f"and a testable hypothesis about what maintains the pattern."
    )

    system_prompt = llm_client.PATTERN_SYSTEM

    try:
        result = llm_client.analyze_with_retry(system_prompt, user_prompt)
    except Exception as e:
        raise ValueError(f"LLM pattern naming failed: {type(e).__name__}: {str(e)}")

    # Ensure result is a list
    if isinstance(result, dict) and "patterns" in result:
        naming_results = result["patterns"]
    elif isinstance(result, list):
        naming_results = result
    else:
        raise ValueError(f"Unexpected LLM response format: {type(result)}")

    # Merge LLM results with original pattern data
    result_by_key = {}
    for naming in naming_results:
        key = naming.get("pattern_key", "unknown")
        result_by_key[key] = naming

    named_patterns = []
    for pattern in patterns:
        key = pattern.get("pattern_key", "unknown")
        naming = result_by_key.get(key, {})

        named_pattern = {
            "pattern_name": naming.get("pattern_name", _default_pattern_name(pattern)),
            "hypothesis": naming.get("hypothesis", _default_hypothesis(pattern)),
            "score": pattern.get("score", 0),
            "evidence": pattern.get("evidence", []),
            "trigger_category": pattern.get("trigger_category", "unknown"),
            "response_emotion": pattern.get("response_emotion", "unknown"),
            "occurrence_count": pattern.get("occurrence_count", 0),
        }
        named_patterns.append(named_pattern)

    return named_patterns


def _generate_deterministic_pattern_names(
    patterns: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Generate pattern names and hypotheses deterministically when LLM fails.

    Uses simple template: "{trigger_category} → {response_emotion} pattern"

    Args:
        patterns: List of raw pattern dicts

    Returns:
        List of Pattern Schema objects with deterministic names/hypotheses
    """
    named_patterns = []

    for pattern in patterns:
        trigger = pattern.get("trigger_category", "unknown").title()
        emotion = pattern.get("response_emotion", "unknown").title()
        occurrence_count = pattern.get("occurrence_count", 0)

        named_pattern = {
            "pattern_name": f"{trigger}–{emotion} Cycle",
            "hypothesis": (
                f"When {trigger.lower()} occurs, {emotion.lower()} typically follows. "
                f"This pattern has been observed {occurrence_count} times in the conversation."
            ),
            "score": pattern.get("score", 0),
            "evidence": pattern.get("evidence", []),
            "trigger_category": pattern.get("trigger_category", "unknown"),
            "response_emotion": pattern.get("response_emotion", "unknown"),
            "occurrence_count": occurrence_count,
        }
        named_patterns.append(named_pattern)

    return named_patterns


def _default_pattern_name(pattern: dict[str, Any]) -> str:
    """Generate a default pattern name."""
    trigger = pattern.get("trigger_category", "unknown").title()
    emotion = pattern.get("response_emotion", "unknown").title()
    return f"{trigger}–{emotion} Cycle"


def _default_hypothesis(pattern: dict[str, Any]) -> str:
    """Generate a default hypothesis."""
    trigger = pattern.get("trigger_category", "unknown").lower()
    emotion = pattern.get("response_emotion", "unknown").lower()
    occurrences = pattern.get("occurrence_count", 0)
    return (
        f"When {trigger} occurs, {emotion} typically follows. "
        f"This pattern has been observed {occurrences} times in the conversation."
    )
