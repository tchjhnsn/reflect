import re
from collections import Counter
from typing import Any

from .models import Event, Workspace
from .pattern_engine import (
    build_snippet,
    display_label,
    recompute_patterns_v0,
)

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "me",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "what",
    "when",
    "why",
    "with",
}

ASK_TOP_K = 5
ASK_MIN_SCORE_THRESHOLD = 1
ASK_MIN_CONFIDENT_CITATIONS = 2
FOCUS_EVENT_SCORE_BOOST = 3


def recompute_pattern_candidates(
    workspace: Workspace,
    max_patterns: int,
    evidence_per_pattern: int,
) -> tuple[list[dict[str, Any]], int]:
    events = list(
        Event.objects.filter(workspace=workspace)
        .only("id", "created_at", "occurred_at", "text", "context_tags", "emotion", "intensity")
        .order_by("-occurred_at", "-created_at", "id")
    )
    patterns = recompute_patterns_v0(
        events,
        max_patterns=max_patterns,
        evidence_per_pattern=evidence_per_pattern,
    )
    return patterns, len(events)


def _question_tokens(question: str) -> list[str]:
    tokens = [token.lower() for token in re.findall(r"[a-zA-Z0-9]+", question)]
    return [token for token in tokens if token not in STOPWORDS and len(token) > 1]


def _event_search_tokens(event: Event) -> set[str]:
    parts = [event.text or "", event.reaction or "", event.outcome or "", event.emotion or ""]
    if isinstance(event.context_tags, list):
        parts.extend([tag for tag in event.context_tags if isinstance(tag, str)])
    tokens = [token.lower() for token in re.findall(r"[a-zA-Z0-9]+", " ".join(parts))]
    return {token for token in tokens if len(token) > 1}


def _normalized_value(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = " ".join(value.split()).strip()
    return normalized if normalized else None


def _top_counter_label(counter: Counter) -> str | None:
    if not counter:
        return None
    return counter.most_common(1)[0][0]


def _compose_evidence_answer(selected_events: list[Event], citation_count: int) -> str:
    tag_counter: Counter[str] = Counter()
    emotion_counter: Counter[str] = Counter()
    reaction_counter: Counter[str] = Counter()
    outcome_counter: Counter[str] = Counter()

    for event in selected_events:
        if isinstance(event.context_tags, list):
            for raw_tag in event.context_tags:
                if isinstance(raw_tag, str):
                    normalized_tag = _normalized_value(raw_tag)
                    if normalized_tag:
                        tag_counter.update([display_label(normalized_tag.lower())])

        normalized_emotion = _normalized_value(event.emotion)
        if normalized_emotion:
            emotion_counter.update([display_label(normalized_emotion.lower())])

        normalized_reaction = _normalized_value(event.reaction)
        if normalized_reaction:
            reaction_counter.update([build_snippet(normalized_reaction)])

        normalized_outcome = _normalized_value(event.outcome)
        if normalized_outcome:
            outcome_counter.update([build_snippet(normalized_outcome)])

    statements = []

    top_tag = _top_counter_label(tag_counter)
    if top_tag is not None:
        statements.append(f"recurring context tag: {top_tag}")

    top_emotion = _top_counter_label(emotion_counter)
    if top_emotion is not None:
        statements.append(f"recurring emotion: {top_emotion}")

    top_reaction = _top_counter_label(reaction_counter)
    top_outcome = _top_counter_label(outcome_counter)
    if top_reaction and top_outcome:
        statements.append(f"recurring reaction/outcome: {top_reaction} -> {top_outcome}")
    elif top_reaction:
        statements.append(f"recurring reaction: {top_reaction}")
    elif top_outcome:
        statements.append(f"recurring outcome: {top_outcome}")

    if not statements:
        return (
            f"Based on {citation_count} cited events, there is not yet a stable recurring "
            "context, emotion, or reaction/outcome pattern."
        )

    return f"Based on {citation_count} cited events, " + "; ".join(statements) + "."


def answer_question(question: str, workspace: Workspace, focus_event: Event | None = None) -> dict[str, Any]:
    tokens = _question_tokens(question)
    token_set = set(tokens)
    events = list(
        Event.objects.filter(workspace=workspace).only(
            "id",
            "occurred_at",
            "created_at",
            "text",
            "context_tags",
            "emotion",
            "reaction",
            "outcome",
        ).order_by("-occurred_at", "-created_at", "id")
    )

    scored = []
    for event in events:
        event_tokens = _event_search_tokens(event)
        overlap = token_set.intersection(event_tokens)
        score = len(overlap)
        if focus_event and event.id == focus_event.id:
            score += FOCUS_EVENT_SCORE_BOOST
        if score >= ASK_MIN_SCORE_THRESHOLD:
            scored.append((event, score, overlap))

    scored.sort(key=lambda item: (-item[1], -item[0].occurred_at.timestamp(), str(item[0].id)))

    selected_tuples = scored[:ASK_TOP_K]
    selected_events = [event for event, _, _ in selected_tuples]

    citations = [str(event.id) for event in selected_events]
    used_events = [
        {
            "event_id": str(event.id),
            "snippet": build_snippet(event.text),
            "overlap_keywords": sorted(overlap),
            "score": score,
        }
        for event, score, overlap in selected_tuples
    ]

    if len(citations) < ASK_MIN_CONFIDENT_CITATIONS:
        return {
            "answer": (
                "Insufficient evidence: fewer than 2 relevant events matched this question. "
                "Log more related events for a confident answer."
            ),
            "citations": citations,
            "used_events": used_events,
        }

    answer = _compose_evidence_answer(selected_events, len(citations))

    return {
        "answer": answer,
        "citations": citations,
        "used_events": used_events,
    }
