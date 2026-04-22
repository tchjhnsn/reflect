from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import timedelta

from django.utils.dateparse import parse_datetime
from django.utils import timezone

from .models import Event
from .neo4j_client import cypher_query

NEGATIVE_EMOTIONS = {
    "anger",
    "anxiety",
    "ashamed",
    "defensive",
    "defensiveness",
    "discouragement",
    "distance",
    "embarrassment",
    "fear",
    "frustration",
    "guilt",
    "hurt",
    "inadequacy",
    "irritation",
    "loneliness",
    "overwhelm",
    "overwhelmed",
    "panic",
    "pressure",
    "regret",
    "resentment",
    "sadness",
    "shame",
    "stress",
    "suspicion",
    "tension",
    "threat",
    "unease",
    "urgency",
    "vulnerability",
}

POSITIVE_EMOTIONS = {
    "acceptance",
    "calm",
    "clarity",
    "closeness",
    "confidence",
    "connection",
    "courage",
    "gratitude",
    "hope",
    "humility",
    "love",
    "pride",
    "relief",
    "resolve",
    "self-awareness",
    "warmth",
}

POSITIVE_CONTEXT_HINTS = {
    "repair",
    "repair attempt",
    "repair progress",
    "positive connection",
    "pattern interruption",
    "relationship",
}

POSITIVE_BEHAVIORS = {
    "accountability",
    "apology",
    "curiosity",
    "early repair",
    "exercise",
    "honesty",
    "journaling",
    "naming flooding",
    "naming insecurity",
    "reframing trust",
    "relaxing together",
    "self-regulation",
    "shared activity",
    "staying present",
    "vulnerability",
    "walking",
}

NEGATIVE_BEHAVIORS = {
    "avoidance",
    "blame",
    "defensiveness",
    "delayed response",
    "dismissal",
    "distance",
    "doomscrolling",
    "freezing",
    "hostility",
    "intellectualizing",
    "joking",
    "lying by omission",
    "overgeneralizing",
    "poor delivery",
    "premature closure",
    "projection",
    "procrastination",
    "sarcasm",
    "scrolling",
    "self-protection",
    "shutdown",
    "silence",
    "snapping",
    "withdrawal",
    "withholding",
}

RESULT_DIMENSION_KEYWORDS = {
    "immediate_relief": {"relief", "calm", "functional", "easier", "better", "space", "stabilized"},
    "later_regret": {"regret", "badly", "distance", "painful", "worse", "guilt", "shame", "unresolved"},
    "clarity": {"clarity", "identified", "named", "obvious", "insight", "understood", "message got through"},
    "reconnection": {"repair", "connected", "closeness", "trust", "bonding", "reassured", "improved"},
    "energy_recovery": {"sleep", "rest", "de-escalated", "recovered", "lighter", "hope", "recovery"},
}

CONCEPT_RULES = {
    "poor sleep": {"sleep", "tired", "fatigue", "night"},
    "anxiety": {"anxiety", "panic", "pressure", "flooded"},
    "procrastination": {"procrastination", "delay", "delayed response", "avoidance"},
    "self-criticism": {"shame", "guilt", "embarrassment", "self-criticism", "failure", "inadequacy"},
    "exercise": {"exercise", "walking", "movement"},
    "relief": {"relief", "hope", "calm"},
    "repair": {"repair", "apology", "accountability", "consistency", "vulnerability"},
    "withdrawal": {"withdrawal", "distance", "shutdown", "silence"},
    "money stress": {"money", "budgeting", "expenses", "saving", "financial discussion"},
    "feeling ignored": {"ignored", "half-listening", "attention", "dismissal"},
    "work overload": {"work", "loaded", "stress spillover", "overload"},
    "conflict": {"argument", "conflict", "fight", "criticism"},
}

BELIEF_RULES = {
    "I'm behind": {"behind", "catch up", "late", "deadline", "not enough time"},
    "I'm disappointing people": {"disappointed", "let her down", "failed", "failure", "not enough"},
    "I should be doing more": {"should", "more", "not enough", "consistency", "try harder"},
    "I'm not safe to relax": {"pressure", "loaded", "functional", "can't relax", "already tense"},
    "I'm hard to love": {"lonely", "rejected", "hard to love", "not enough", "with me"},
    "I'm being judged": {"judged", "criticized", "interrogated", "compared", "cornered"},
}

BEHAVIOR_KEYWORDS = {
    "doomscrolling": {"doomscroll", "scrolling", "scroll"},
    "exercise": {"exercise", "walk", "walking", "movement"},
    "text friend": {"text", "reach out", "friend"},
    "journaling": {"journal", "writing", "reflection"},
    "procrastination": {"procrastination", "delay", "avoid"},
    "isolate": {"withdraw", "distance", "quiet", "shut down", "silence"},
    "overeat": {"eat", "overeating", "snack"},
}


@dataclass
class DerivedEvent:
    id: str
    occurred_at: timezone.datetime
    text: str
    contexts: list[str]
    people: list[str]
    emotions: list[str]
    behaviors: list[str]
    concepts: list[str]
    beliefs: list[str]
    interpretations: list[str]
    outcome: str
    result_dims: list[str]
    valence: str
    intensity: int | None


def _normalize_term(value: str) -> str:
    return " ".join(str(value or "").replace("_", " ").replace("-", " ").strip().lower().split())


def _title_case(value: str) -> str:
    return " ".join(part.capitalize() for part in _normalize_term(value).split())


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _safe_list(value) -> list[str]:
    if not isinstance(value, list):
        return []
    items: list[str] = []
    for item in value:
        if isinstance(item, str):
            normalized = _normalize_term(item)
            if normalized:
                items.append(normalized)
    return items


def _safe_json_list(value) -> list:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (TypeError, json.JSONDecodeError):
            return []
        return parsed if isinstance(parsed, list) else []
    return []


def _parse_graph_datetime(value) -> timezone.datetime:
    if value is None:
        return timezone.now()
    if hasattr(value, "to_native"):
        native = value.to_native()
        if native is not None:
            value = native
    if isinstance(value, timezone.datetime):
        dt = value
    else:
        dt = parse_datetime(str(value))
    if dt is None:
        return timezone.now()
    if timezone.is_naive(dt):
        return timezone.make_aware(dt)
    return dt


def _extract_prefixed(tags: list[str], prefix: str) -> list[str]:
    extracted: list[str] = []
    marker = f"{prefix}:"
    for tag in tags:
        if tag.startswith(marker):
            extracted.append(_normalize_term(tag[len(marker):]))
    return extracted


def _extract_contexts(tags: list[str]) -> list[str]:
    ignored_prefixes = ("import_id:", "action:", "emotion_tag:")
    return [tag for tag in tags if not tag.startswith(ignored_prefixes)]


def _extract_emotions(event: Event, tags: list[str]) -> list[str]:
    emotions: list[str] = []
    if event.emotion:
        emotions.append(_normalize_term(event.emotion))
    emotions.extend(_extract_prefixed(tags, "emotion_tag"))
    text = _normalize_term(event.text)
    for emotion in sorted(POSITIVE_EMOTIONS | NEGATIVE_EMOTIONS):
        if emotion in text and emotion not in emotions:
            emotions.append(emotion)
    return _dedupe(emotions)


def _extract_behaviors(event: Event, tags: list[str]) -> list[str]:
    behaviors = _extract_prefixed(tags, "action")
    if event.reaction:
        behaviors.extend(_normalize_term(part) for part in re.split(r"[,\n;/]+", event.reaction) if part.strip())
    text = _normalize_term(event.text)
    for behavior, keywords in BEHAVIOR_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            behaviors.append(behavior)
    return _dedupe([value for value in behaviors if value])


def _extract_beliefs(text: str) -> list[str]:
    lowered = _normalize_term(text)
    beliefs = [
        belief
        for belief, keywords in BELIEF_RULES.items()
        if any(keyword in lowered for keyword in keywords)
    ]
    if not beliefs and "i " in lowered and ("should" in lowered or "failed" in lowered or "behind" in lowered):
        beliefs.append("I should be doing more")
    return _dedupe(beliefs)


def _extract_interpretations(text: str, emotions: list[str], contexts: list[str]) -> list[str]:
    lowered = _normalize_term(text)
    interpretations: list[str] = []
    if "critic" in lowered or "judg" in lowered:
        interpretations.append("being judged")
    if "cornered" in lowered or "trapped" in lowered:
        interpretations.append("cornered")
    if "ignored" in lowered or "half listening" in lowered or "attention" in lowered:
        interpretations.append("being ignored")
    if "compare" in lowered or "enough" in lowered:
        interpretations.append("not enough")
    if "support" in lowered and ("invasive" in lowered or "pressure" in lowered):
        interpretations.append("support feels invasive")
    if "relationship" in contexts and any(emotion in {"guilt", "shame", "fear"} for emotion in emotions):
        interpretations.append("fear of rejection")
    return _dedupe(interpretations)


def _extract_concepts(text: str, contexts: list[str], emotions: list[str], behaviors: list[str], beliefs: list[str]) -> list[str]:
    lowered = _normalize_term(text)
    concepts = list(contexts) + list(emotions) + list(behaviors)
    concepts.extend(_normalize_term(belief) for belief in beliefs)
    for concept, keywords in CONCEPT_RULES.items():
        if any(keyword in lowered for keyword in keywords):
            concepts.append(concept)
    return _dedupe([value for value in concepts if value])


def _classify_result_dims(text: str, outcome: str, behaviors: list[str], valence: str) -> list[str]:
    haystack = f"{_normalize_term(text)} {_normalize_term(outcome)} {' '.join(behaviors)}"
    dims = [
        name for name, keywords in RESULT_DIMENSION_KEYWORDS.items()
        if any(keyword in haystack for keyword in keywords)
    ]
    if valence == "helping_me_recover":
        dims.extend(["clarity", "reconnection"])
    if valence == "dragging_me_down":
        dims.append("later_regret")
    return _dedupe(dims)


def _classify_valence(emotions: list[str], behaviors: list[str], contexts: list[str], outcome: str, text: str) -> str:
    positive = sum(1 for emotion in emotions if emotion in POSITIVE_EMOTIONS)
    negative = sum(1 for emotion in emotions if emotion in NEGATIVE_EMOTIONS)
    positive += sum(1 for behavior in behaviors if behavior in POSITIVE_BEHAVIORS)
    negative += sum(1 for behavior in behaviors if behavior in NEGATIVE_BEHAVIORS)
    positive += sum(1 for context in contexts if context in POSITIVE_CONTEXT_HINTS)
    lowered = f"{_normalize_term(text)} {_normalize_term(outcome)}"
    if any(token in lowered for token in {"improved", "repair", "recovered", "hope", "bonding", "closeness"}):
        positive += 1
    if any(token in lowered for token in {"badly", "distance", "unresolved", "worse", "escalated", "painful"}):
        negative += 1
    return "helping_me_recover" if positive > negative else "dragging_me_down"


def _event_weight(event: DerivedEvent, metric_mode: str) -> float:
    if metric_mode == "intensity":
        return float(event.intensity or max(1, len(event.emotions)))
    if metric_mode == "recency":
        age_days = max(0.0, (timezone.now() - event.occurred_at).total_seconds() / 86400.0)
        return 1.0 / (1.0 + age_days / 30.0)
    if metric_mode == "impact":
        return float(1 + min(4, len(event.emotions) + len(event.behaviors) + len(event.result_dims)))
    return 1.0


def _serialize_event_sample(event: DerivedEvent) -> dict:
    return {
        "id": event.id,
        "date": event.occurred_at.isoformat(),
        "text": event.text[:240],
        "contexts": event.contexts[:4],
        "people": event.people[:4],
        "emotions": event.emotions[:4],
        "behaviors": event.behaviors[:4],
        "outcome": event.outcome[:140],
    }


def _short_date_label(dt) -> str:
    try:
        return dt.strftime("%b %-d")
    except ValueError:
        return dt.strftime("%b %d").replace(" 0", " ")


def _timeline_label(event: DerivedEvent, mode: str = "weekday") -> str:
    if mode == "short_date":
        return _short_date_label(event.occurred_at)
    return event.occurred_at.strftime("%a")


def _timeline_item(event: DerivedEvent, label: str | None = None, *, kind: str = "event") -> dict:
    return {
        "id": event.id,
        "date": event.occurred_at.isoformat(),
        "label": label or _timeline_label(event),
        "title": (event.concepts or event.contexts or ["moment"])[0],
        "kind": kind,
    }


def _top_counts(values: list[str], limit: int = 4) -> list[str]:
    counter = Counter(values)
    return [name for name, _ in counter.most_common(limit)]


def _build_summary(events: list[DerivedEvent], label: str) -> dict:
    contexts: list[str] = []
    behaviors: list[str] = []
    outcomes: list[str] = []
    for event in events:
        contexts.extend(event.contexts)
        behaviors.extend(event.behaviors)
        if event.outcome:
            outcomes.append(event.outcome)

    return {
        "selected_key": label,
        "linked_count": len(events),
        "common_contexts": _top_counts(contexts, 3),
        "common_behaviors": _top_counts(behaviors, 3),
        "common_outcomes": outcomes[:3],
        "sample_events": [_serialize_event_sample(event) for event in events[:5]],
    }


def _choose_terms(counter_or_values, fallback: list[str], limit: int = 6) -> list[str]:
    counter = counter_or_values if isinstance(counter_or_values, Counter) else Counter(counter_or_values)
    terms = [term for term, count in counter.most_common(limit) if count > 0]
    if len(terms) < min(limit, len(fallback)):
        for term in fallback:
            if term not in terms:
                terms.append(term)
            if len(terms) >= limit:
                break
    return terms[:limit]


def _make_matrix_card(
    *,
    card_id: str,
    title: str,
    question: str,
    rows: list[str],
    cols: list[str],
    cell_events: dict[tuple[str, str], list[DerivedEvent]],
    highlights: list[str],
    symmetric: bool = False,
    normalize: bool = False,
    metric_mode: str,
    min_support: int = 1,
) -> dict:
    values: dict[tuple[str, str], float] = {}
    max_value = 0.0
    for key, events in cell_events.items():
        total = sum(_event_weight(event, metric_mode) for event in events)
        values[key] = total
        max_value = max(max_value, total)

    row_totals: dict[str, float] = defaultdict(float)
    for (row, _), value in values.items():
        row_totals[row] += value

    cells = []
    default_selection_key = None
    for row in rows:
        for col in cols:
            if symmetric and row == col:
                continue
            key = (row, col)
            events = cell_events.get(key, [])
            value = values.get(key, 0.0)
            if len(events) < min_support:
                events = []
                value = 0.0
            if normalize:
                denom = row_totals.get(row, 0.0) or 1.0
                score = value / denom
            else:
                score = value / max_value if max_value else 0.0
            cell_key = f"{row}::{col}"
            if default_selection_key is None and events:
                default_selection_key = cell_key
            cells.append(
                {
                    "key": cell_key,
                    "row": row,
                    "col": col,
                    "label": f"{row} ↔ {col}" if symmetric else f"{row} -> {col}",
                    "value": round(value, 2),
                    "score": round(score, 4),
                    "summary": _build_summary(events, cell_key) if events else None,
                }
            )

    return {
        "id": card_id,
        "title": title,
        "kind": "matrix",
        "question": question,
        "rows": rows,
        "cols": cols,
        "cells": cells,
        "highlights": highlights,
        "default_selection_key": default_selection_key,
    }


def _time_of_day_label(dt) -> str:
    hour = dt.hour
    if hour < 12:
        return "Morning"
    if hour < 17:
        return "Afternoon"
    if hour < 21:
        return "Evening"
    return "Night"


def _compare_group_label(event: DerivedEvent, compare_by: str) -> str:
    if compare_by == "day_of_week":
        return event.occurred_at.strftime("%A")
    if compare_by == "month":
        return event.occurred_at.strftime("%b %Y")
    if compare_by == "person":
        return event.people[0] if event.people else "Nobody named"
    if compare_by == "context":
        return event.contexts[0] if event.contexts else "Uncategorized"
    return _time_of_day_label(event.occurred_at)


def _graph_text_preview(signal_row: dict) -> str:
    preview = str(signal_row.get("text") or "").strip()
    if preview:
        return preview
    for candidate in signal_row.get("turn_previews") or []:
        candidate_text = str(candidate or "").strip()
        if candidate_text:
            return candidate_text
    return str(signal_row.get("signal_address") or "Signal").strip()


def _graph_emotions(signal_row: dict) -> tuple[list[str], int | None]:
    payload = _safe_json_list(signal_row.get("emotions_payload"))
    emotions: list[str] = []
    intensities: list[float] = []

    for item in payload:
        if isinstance(item, dict):
            name = _normalize_term(item.get("emotion"))
            if name:
                emotions.append(name)
            try:
                if item.get("intensity") is not None:
                    intensities.append(float(item["intensity"]))
            except (TypeError, ValueError):
                pass
        elif isinstance(item, str):
            name = _normalize_term(item)
            if name:
                emotions.append(name)

    emotions.extend(_safe_list(signal_row.get("emotion_names")))
    deduped = _dedupe(emotions)

    legacy_intensity = signal_row.get("legacy_intensity")
    if legacy_intensity is not None:
        try:
            intensities.append(float(legacy_intensity))
        except (TypeError, ValueError):
            pass

    intensity = int(round(max(intensities))) if intensities else None
    return deduped, intensity


def _derive_events_from_graph(*, workspace) -> tuple[list[DerivedEvent], bool]:
    query = """
    MATCH (s:Signal {workspace_id: $workspace_id})
    OPTIONAL MATCH (s)-[:IN_CONTEXT]->(ctx:ContextNode {workspace_id: $workspace_id})
    WITH s, collect(DISTINCT ctx.name) AS contexts
    OPTIONAL MATCH (p:Person {workspace_id: $workspace_id})-[:PARTICIPANT_IN]->(s)
    WITH s, contexts, collect(DISTINCT p.name) AS people
    OPTIONAL MATCH (s)-[:EXPRESSES_EMOTION]->(e:Emotion {workspace_id: $workspace_id})
    WITH s, contexts, people, collect(DISTINCT e.name) AS emotion_names
    OPTIONAL MATCH (s)-[:SHOWS_BEHAVIOR]->(b:Behavior {workspace_id: $workspace_id})
    WITH s, contexts, people, emotion_names, collect(DISTINCT b.name) AS behavior_names
    OPTIONAL MATCH (s)-[:INVOLVES_ACTION]->(a:ActionNode {workspace_id: $workspace_id})
    WITH s, contexts, people, emotion_names, behavior_names, collect(DISTINCT a.name) AS action_names
    OPTIONAL MATCH (s)-[:LED_TO]->(o:Outcome {workspace_id: $workspace_id})
    WITH s, contexts, people, emotion_names, behavior_names, action_names, collect(DISTINCT o.name) AS outcome_names
    OPTIONAL MATCH (s)<-[:PRODUCES]-(t:Turn {workspace_id: $workspace_id})
    WITH s, contexts, people, emotion_names, behavior_names, action_names, outcome_names,
         collect(DISTINCT coalesce(t.content_preview, t.content, '')) AS turn_previews
    RETURN s.uid AS id,
           coalesce(toString(s.created_at), '') AS occurred_at,
           coalesce(s.content_preview, '') AS text,
           coalesce(s.signal_address, '') AS signal_address,
           s.emotions AS emotions_payload,
           s.intensity AS legacy_intensity,
           contexts,
           people,
           emotion_names,
           behavior_names,
           action_names,
           outcome_names,
           turn_previews
    ORDER BY occurred_at DESC
    """

    try:
        rows, columns = cypher_query(query, {"workspace_id": str(workspace.id)})
    except Exception:
        return [], False

    derived: list[DerivedEvent] = []
    for row in rows:
        signal_row = dict(zip(columns, row))
        occurred_at = _parse_graph_datetime(signal_row.get("occurred_at"))
        text = _graph_text_preview(signal_row)
        contexts = _safe_list(signal_row.get("contexts"))
        people = _safe_list(signal_row.get("people"))
        emotions, intensity = _graph_emotions(signal_row)
        behaviors = _dedupe(_safe_list(signal_row.get("behavior_names")) + _safe_list(signal_row.get("action_names")))
        outcome_names = _safe_list(signal_row.get("outcome_names"))
        outcome = " | ".join(_title_case(name) for name in outcome_names[:3])
        beliefs = _extract_beliefs(text)
        interpretations = _extract_interpretations(text, emotions, contexts)
        concepts = _extract_concepts(text, contexts, emotions, behaviors, beliefs)
        valence = _classify_valence(emotions, behaviors, contexts, outcome, text)
        result_dims = _classify_result_dims(text, outcome, behaviors, valence)

        derived.append(
            DerivedEvent(
                id=str(signal_row.get("id") or signal_row.get("signal_address") or len(derived)),
                occurred_at=occurred_at,
                text=text,
                contexts=contexts,
                people=people,
                emotions=emotions,
                behaviors=behaviors,
                concepts=concepts,
                beliefs=beliefs,
                interpretations=interpretations,
                outcome=outcome,
                result_dims=result_dims,
                valence=valence,
                intensity=intensity,
            )
        )

    derived.sort(key=lambda item: item.occurred_at, reverse=True)
    return derived, True


def _derive_events(events: list[Event]) -> list[DerivedEvent]:
    derived: list[DerivedEvent] = []
    for event in events:
        tags = _safe_list(event.context_tags)
        contexts = _extract_contexts(tags)
        people = _safe_list(event.people)
        emotions = _extract_emotions(event, tags)
        behaviors = _extract_behaviors(event, tags)
        beliefs = _extract_beliefs(event.text)
        interpretations = _extract_interpretations(event.text, emotions, contexts)
        concepts = _extract_concepts(event.text, contexts, emotions, behaviors, beliefs)
        valence = _classify_valence(emotions, behaviors, contexts, event.outcome or "", event.text)
        result_dims = _classify_result_dims(event.text, event.outcome or "", behaviors, valence)
        derived.append(
            DerivedEvent(
                id=str(event.id),
                occurred_at=event.occurred_at,
                text=event.text,
                contexts=contexts,
                people=people,
                emotions=emotions,
                behaviors=behaviors,
                concepts=concepts,
                beliefs=beliefs,
                interpretations=interpretations,
                outcome=event.outcome or "",
                result_dims=result_dims,
                valence=valence,
                intensity=event.intensity,
            )
        )
    derived.sort(key=lambda item: item.occurred_at, reverse=True)
    return derived


def _filter_events(events: list[DerivedEvent], filters: dict) -> list[DerivedEvent]:
    filtered = list(events)
    time_range = filters.get("time_range", "30d")
    if time_range not in {"7d", "30d", "90d", "all"}:
        time_range = "30d"
    if time_range != "all":
        days = int(time_range[:-1])
        cutoff = timezone.now() - timedelta(days=days)
        filtered = [event for event in filtered if event.occurred_at >= cutoff]

    contexts = {_normalize_term(value) for value in filters.get("contexts", []) if value}
    if contexts:
        filtered = [event for event in filtered if contexts.intersection(event.contexts)]

    people = {_normalize_term(value) for value in filters.get("people", []) if value}
    if people:
        filtered = [event for event in filtered if people.intersection(event.people)]

    pattern_trend = filters.get("pattern_trend") or filters.get("pattern_valence", "helping_me_recover")
    if pattern_trend in {"dragging_me_down", "helping_me_recover"}:
        filtered = [event for event in filtered if event.valence == pattern_trend]

    return filtered


def _available_filters(events: list[DerivedEvent]) -> dict:
    contexts: Counter[str] = Counter()
    people: Counter[str] = Counter()
    for event in events:
        contexts.update(event.contexts)
        people.update(event.people)
    return {
        "contexts": [name for name, _ in contexts.most_common(12)],
        "people": [name for name, _ in people.most_common(12)],
        "time_ranges": ["7d", "30d", "90d", "all"],
        "compare_by": ["none", "time_of_day", "day_of_week", "month", "person", "context"],
        "pattern_trend": ["helping_me_recover", "dragging_me_down", "all"],
    }


def _emotion_matrix(events: list[DerivedEvent], metric_mode: str, normalize: bool, min_support: int) -> dict:
    counter = Counter()
    for event in events:
        counter.update(event.emotions)
    labels = _choose_terms(counter, ["anxiety", "shame", "frustration", "relief", "hope"])
    cell_events: dict[tuple[str, str], list[DerivedEvent]] = defaultdict(list)
    for event in events:
        for row in labels:
            for col in labels:
                if row == col:
                    continue
                if row in event.emotions and col in event.emotions:
                    cell_events[(row, col)].append(event)
    hottest = [f"{_title_case(a)} + {_title_case(b)}" for (a, b), _ in Counter({k: len(v) for k, v in cell_events.items()}).most_common(3)]
    return _make_matrix_card(
        card_id="emotional-pattern-matrix",
        title="Emotional Pattern Matrix",
        question="Which emotions tend to travel together?",
        rows=labels,
        cols=labels,
        cell_events=cell_events,
        highlights=hottest,
        symmetric=True,
        normalize=normalize,
        metric_mode=metric_mode,
        min_support=min_support,
    )


def _pattern_story_matrix(events: list[DerivedEvent], metric_mode: str, normalize: bool, min_support: int) -> dict:
    counter = Counter()
    for event in events:
        counter.update(event.concepts)
    labels = _choose_terms(counter, ["poor sleep", "anxiety", "procrastination", "self criticism", "exercise", "relief"])
    cell_events: dict[tuple[str, str], list[DerivedEvent]] = defaultdict(list)
    for event in events:
        for row in labels:
            for col in labels:
                if row == col:
                    continue
                if row in event.concepts and col in event.concepts:
                    cell_events[(row, col)].append(event)
    ranked = Counter({key: len(value) for key, value in cell_events.items()}).most_common(4)
    highlights = [f"{_title_case(row)} -> {_title_case(col)}" for (row, col), _ in ranked]
    return _make_matrix_card(
        card_id="pattern-story-matrix",
        title="Pattern Story Matrix",
        question="What sequences keep showing up together?",
        rows=labels,
        cols=labels,
        cell_events=cell_events,
        highlights=highlights,
        symmetric=False,
        normalize=normalize,
        metric_mode=metric_mode,
        min_support=min_support,
    )


def _trigger_emotion_matrix(events: list[DerivedEvent], metric_mode: str, normalize: bool, min_support: int) -> dict:
    trigger_counter = Counter()
    emotion_counter = Counter()
    for event in events:
        trigger_counter.update(event.contexts or event.interpretations or event.concepts[:2])
        emotion_counter.update(event.emotions)
    rows = _choose_terms(trigger_counter, ["conflict", "lack of sleep", "money stress", "feeling ignored", "work overload"])
    cols = _choose_terms(emotion_counter, ["anger", "sadness", "anxiety", "shame", "numbness"])
    cell_events: dict[tuple[str, str], list[DerivedEvent]] = defaultdict(list)
    for event in events:
        triggers = set(event.contexts + event.interpretations + event.concepts)
        for row in rows:
            if row not in triggers:
                continue
            for col in cols:
                if col in event.emotions:
                    cell_events[(row, col)].append(event)
    highlights = [f"{_title_case(row)} -> {_title_case(col)}" for (row, col), _ in Counter({k: len(v) for k, v in cell_events.items()}).most_common(3)]
    return _make_matrix_card(
        card_id="trigger-emotion-matrix",
        title="Trigger -> Emotion Matrix",
        question="What tends to activate what?",
        rows=rows,
        cols=cols,
        cell_events=cell_events,
        highlights=highlights,
        normalize=normalize,
        metric_mode=metric_mode,
        min_support=min_support,
    )


def _emotion_behavior_matrix(events: list[DerivedEvent], metric_mode: str, normalize: bool, min_support: int) -> dict:
    emotion_counter = Counter()
    behavior_counter = Counter()
    for event in events:
        emotion_counter.update(event.emotions)
        behavior_counter.update(event.behaviors)
    rows = _choose_terms(emotion_counter, ["anxiety", "shame", "anger", "loneliness", "relief"])
    cols = _choose_terms(behavior_counter, ["isolate", "doomscrolling", "procrastination", "overeat", "exercise", "text friend"])
    cell_events: dict[tuple[str, str], list[DerivedEvent]] = defaultdict(list)
    for event in events:
        for row in rows:
            if row not in event.emotions:
                continue
            for col in cols:
                if col in event.behaviors:
                    cell_events[(row, col)].append(event)
    highlights = [f"{_title_case(row)} -> {_title_case(col)}" for (row, col), _ in Counter({k: len(v) for k, v in cell_events.items()}).most_common(3)]
    return _make_matrix_card(
        card_id="emotion-behavior-matrix",
        title="Emotion -> Behavior Matrix",
        question="When I feel this, what do I usually do?",
        rows=rows,
        cols=cols,
        cell_events=cell_events,
        highlights=highlights,
        normalize=normalize,
        metric_mode=metric_mode,
        min_support=min_support,
    )


def _context_entanglement_matrix(events: list[DerivedEvent], metric_mode: str, normalize: bool, min_support: int) -> dict:
    context_counter = Counter()
    for event in events:
        context_counter.update(event.contexts)
    labels = _choose_terms(context_counter, ["work", "relationships", "health", "self worth", "money", "purpose"])
    cell_events: dict[tuple[str, str], list[DerivedEvent]] = defaultdict(list)
    for event in events:
        for row in labels:
            for col in labels:
                if row == col:
                    continue
                if row in event.contexts and col in event.contexts:
                    cell_events[(row, col)].append(event)
    highlights = [f"{_title_case(a)} + {_title_case(b)}" for (a, b), _ in Counter({k: len(v) for k, v in cell_events.items()}).most_common(3)]
    return _make_matrix_card(
        card_id="context-entanglement-matrix",
        title="Context Entanglement Matrix",
        question="Which life areas are tangled together?",
        rows=labels,
        cols=labels,
        cell_events=cell_events,
        highlights=highlights,
        symmetric=True,
        normalize=normalize,
        metric_mode=metric_mode,
        min_support=min_support,
    )


def _people_state_matrix(events: list[DerivedEvent], metric_mode: str, normalize: bool, min_support: int) -> dict:
    people_counter = Counter()
    emotion_counter = Counter()
    for event in events:
        people_counter.update(event.people)
        emotion_counter.update(event.emotions)
    rows = _choose_terms(people_counter, ["self", "maya", "mom", "boss", "friend"])
    cols = _choose_terms(emotion_counter, ["calm", "defensive", "anxious", "ashamed", "loved", "resentful"])
    cell_events: dict[tuple[str, str], list[DerivedEvent]] = defaultdict(list)
    for event in events:
        event_people = set(event.people or ["self"])
        event_emotions = set(event.emotions)
        for row in rows:
            if row not in event_people:
                continue
            for col in cols:
                if col in event_emotions:
                    cell_events[(row, col)].append(event)
    highlights = [f"{_title_case(a)} -> {_title_case(b)}" for (a, b), _ in Counter({k: len(v) for k, v in cell_events.items()}).most_common(4)]
    return _make_matrix_card(
        card_id="people-state-matrix",
        title="People -> Internal State Matrix",
        question="Which people tend to evoke which internal states?",
        rows=rows,
        cols=cols,
        cell_events=cell_events,
        highlights=highlights,
        normalize=normalize,
        metric_mode=metric_mode,
        min_support=min_support,
    )


def _beliefs_matrix(events: list[DerivedEvent], metric_mode: str, normalize: bool, min_support: int) -> dict:
    belief_counter = Counter()
    emotion_counter = Counter()
    for event in events:
        belief_counter.update(event.beliefs)
        emotion_counter.update([emotion for emotion in event.emotions if emotion in NEGATIVE_EMOTIONS])
    rows = _choose_terms(list(belief_counter.elements()), list(BELIEF_RULES.keys()))
    cols = _choose_terms(emotion_counter, ["shame", "anxiety", "sadness", "urgency", "anger"])
    cell_events: dict[tuple[str, str], list[DerivedEvent]] = defaultdict(list)
    for event in events:
        for row in rows:
            if row not in event.beliefs:
                continue
            for col in cols:
                if col in event.emotions:
                    cell_events[(row, col)].append(event)
    highlights = [f"{row} -> {_title_case(col)}" for (row, col), _ in Counter({k: len(v) for k, v in cell_events.items()}).most_common(3)]
    return _make_matrix_card(
        card_id="expensive-beliefs-matrix",
        title="Expensive Beliefs Matrix",
        question="Which beliefs are emotionally expensive?",
        rows=rows,
        cols=cols,
        cell_events=cell_events,
        highlights=highlights,
        normalize=normalize,
        metric_mode=metric_mode,
        min_support=min_support,
    )


def _works_matrix(events: list[DerivedEvent], metric_mode: str, normalize: bool, min_support: int) -> dict:
    behavior_counter = Counter()
    result_counter = Counter()
    for event in events:
        behavior_counter.update(event.behaviors)
        result_counter.update(event.result_dims)
    rows = _choose_terms(behavior_counter, ["scrolling", "venting", "walking", "journaling", "avoiding", "sleep"])
    cols = _choose_terms(result_counter, ["immediate_relief", "later_regret", "clarity", "reconnection", "energy_recovery"])
    cell_events: dict[tuple[str, str], list[DerivedEvent]] = defaultdict(list)
    for event in events:
        for row in rows:
            if row not in event.behaviors:
                continue
            for col in cols:
                if col in event.result_dims:
                    cell_events[(row, col)].append(event)
    highlights = [f"{_title_case(a)} -> {_title_case(b)}" for (a, b), _ in Counter({k: len(v) for k, v in cell_events.items()}).most_common(3)]
    return _make_matrix_card(
        card_id="what-actually-works-matrix",
        title="What Actually Works Matrix",
        question="What actually works vs what just feels like it works?",
        rows=rows,
        cols=cols,
        cell_events=cell_events,
        highlights=highlights,
        normalize=normalize,
        metric_mode=metric_mode,
        min_support=min_support,
    )


def _landscape_card(events: list[DerivedEvent], metric_mode: str) -> dict:
    negative_counter = Counter()
    positive_counter = Counter()
    negative_events: dict[str, list[DerivedEvent]] = defaultdict(list)
    positive_events: dict[str, list[DerivedEvent]] = defaultdict(list)
    for event in events:
        concepts = event.concepts or event.contexts or event.emotions
        for concept in concepts:
            if event.valence == "helping_me_recover":
                positive_counter[concept] += _event_weight(event, metric_mode)
                positive_events[concept].append(event)
            else:
                negative_counter[concept] += _event_weight(event, metric_mode)
                negative_events[concept].append(event)

    dragging = [name for name, _ in negative_counter.most_common(5)]
    helping = [name for name, _ in positive_counter.most_common(5)]
    bridge_counter = Counter()
    for concept in set(dragging).intersection(helping):
        bridge_counter[concept] = min(negative_counter[concept], positive_counter[concept])
    if not bridge_counter:
        for concept, _ in (negative_counter & positive_counter).most_common(3):
            bridge_counter[concept] += 1

    columns = [
        {
            "id": "dragging",
            "label": "Dragging Me Down",
            "items": [
                {
                    "key": f"dragging::{item}",
                    "label": item,
                    "value": round(negative_counter[item], 2),
                    "summary": _build_summary(negative_events[item], item),
                }
                for item in dragging
            ],
        },
        {
            "id": "helping",
            "label": "Helping Me Recover",
            "items": [
                {
                    "key": f"helping::{item}",
                    "label": item,
                    "value": round(positive_counter[item], 2),
                    "summary": _build_summary(positive_events[item], item),
                }
                for item in helping
            ],
        },
    ]
    bridge_nodes = [
        {
            "key": f"bridge::{item}",
            "label": item,
            "value": round(bridge_counter[item], 2),
        }
        for item, _ in bridge_counter.most_common(4)
    ]
    default_selection_key = columns[0]["items"][0]["key"] if columns[0]["items"] else None
    return {
        "id": "pattern-landscape",
        "title": "Pattern Landscape",
        "kind": "landscape",
        "question": "What patterns drag me down vs help me recover?",
        "columns": columns,
        "bridge_nodes": bridge_nodes,
        "highlights": [f"Bridge: {_title_case(item['label'])}" for item in bridge_nodes[:3]],
        "default_selection_key": default_selection_key,
    }


def _compare_card(events: list[DerivedEvent], metric_mode: str, normalize: bool, compare_by: str, min_support: int) -> dict:
    compare_by = compare_by if compare_by and compare_by != "none" else "time_of_day"
    grouped: dict[str, list[DerivedEvent]] = defaultdict(list)
    for event in events:
        grouped[_compare_group_label(event, compare_by)].append(event)

    if compare_by == "time_of_day":
        labels = ["Morning", "Afternoon", "Evening", "Night"]
    else:
        labels = [label for label, _ in Counter({key: len(value) for key, value in grouped.items()}).most_common(4)]
    labels = labels[:4]

    slices = []
    for label in labels:
        subset = grouped.get(label, [])
        emotion_counter = Counter()
        for event in subset:
            emotion_counter.update(event.emotions)
        axes = _choose_terms(emotion_counter, ["anxiety", "shame", "frustration", "relief"], limit=4)
        cell_events: dict[tuple[str, str], list[DerivedEvent]] = defaultdict(list)
        for event in subset:
            for row in axes:
                for col in axes:
                    if row == col:
                        continue
                    if row in event.emotions and col in event.emotions:
                        cell_events[(row, col)].append(event)
        matrix = _make_matrix_card(
            card_id=f"compare-{_normalize_term(label).replace(' ', '-')}",
            title=label,
            question="",
            rows=axes,
            cols=axes,
            cell_events=cell_events,
            highlights=[],
            symmetric=True,
            normalize=normalize,
            metric_mode=metric_mode,
            min_support=min_support,
        )
        slices.append(
            {
                "label": label,
                "matrix": {
                    "rows": matrix["rows"],
                    "cols": matrix["cols"],
                    "cells": matrix["cells"],
                },
            }
        )

    return {
        "id": "compare-by-card",
        "title": "Compare By",
        "kind": "compare",
        "question": "How does the pattern shift across slices?",
        "compare_by": compare_by,
        "slices": slices,
        "highlights": [f"{slice_item['label']}: {len(grouped.get(slice_item['label'], []))} moments" for slice_item in slices[:4]],
    }


def _interconnection_chord_card(events: list[DerivedEvent], metric_mode: str, min_support: int) -> dict:
    concept_counter = Counter()
    for event in events:
        concept_counter.update(event.concepts[:8])

    node_labels = _choose_terms(
        concept_counter,
        ["poor sleep", "anxiety", "procrastination", "self-criticism", "exercise", "relief"],
        limit=12,
    )
    node_set = set(node_labels)
    edge_events: dict[tuple[str, str], list[DerivedEvent]] = defaultdict(list)
    edge_weights: Counter[tuple[str, str]] = Counter()

    for event in events:
        event_concepts = [concept for concept in event.concepts if concept in node_set]
        event_concepts = _dedupe(event_concepts)[:8]
        for index, source in enumerate(event_concepts):
            for target in event_concepts[index + 1:]:
                key = tuple(sorted((source, target)))
                edge_events[key].append(event)
                edge_weights[key] += _event_weight(event, metric_mode)

    links = []
    strongest_edges = []
    for (source, target), weight in edge_weights.most_common(24):
        events_for_edge = edge_events[(source, target)]
        if len(events_for_edge) < min_support:
            continue
        link_key = f"{source}::{target}"
        links.append(
            {
                "key": link_key,
                "source": source,
                "target": target,
                "value": round(weight, 2),
                "summary": _build_summary(events_for_edge, f"{source} ↔ {target}"),
            }
        )
        strongest_edges.append(f"{_title_case(source)} ↔ {_title_case(target)}")

    nodes = [
        {
            "id": label,
            "label": label,
            "value": round(concept_counter[label], 2),
        }
        for label in node_labels
    ]

    default_selection_key = links[0]["key"] if links else None
    return {
        "id": "interconnection-chord",
        "title": "Interconnection Chord",
        "kind": "chord",
        "question": "Which patterns, states, and behaviors are most tightly interwoven?",
        "nodes": nodes,
        "links": links,
        "highlights": strongest_edges[:4],
        "default_selection_key": default_selection_key,
    }


def _precursor_window_card(events: list[DerivedEvent]) -> dict:
    if not events:
        return {
            "id": "precursor-window",
            "title": "48-Hour Build-Up",
            "kind": "timeline-graph",
            "question": "What was building in the 48 hours before a selected event?",
            "graph": {"nodes": [], "edges": []},
            "timeline": [],
            "notes": ["Not enough data yet."],
            "sidebar": None,
        }

    ordered = sorted(events, key=lambda event: event.occurred_at)
    anchor = next((event for event in reversed(ordered) if event.valence == "dragging_me_down"), ordered[-1])
    window_start = anchor.occurred_at - timedelta(hours=48)
    window_events = [event for event in ordered if window_start <= event.occurred_at <= anchor.occurred_at]
    prior_events = [event for event in window_events if event.id != anchor.id][-2:]
    precursor_a = prior_events[0] if len(prior_events) > 0 else None
    precursor_b = prior_events[1] if len(prior_events) > 1 else None
    leaves = _dedupe(anchor.emotions[:2] + anchor.behaviors[:2] + anchor.interpretations[:1])[:3]

    nodes = []
    edges = []
    if precursor_a:
        nodes.append({
            "id": f"pre-a-{precursor_a.id}",
            "label": (precursor_a.concepts or precursor_a.contexts or ["build up"])[0],
            "x": 12,
            "y": 18,
            "tone": "support",
        })
    if precursor_b:
        nodes.append({
            "id": f"pre-b-{precursor_b.id}",
            "label": (precursor_b.concepts or precursor_b.contexts or ["build up"])[0],
            "x": 12,
            "y": 46,
            "tone": "support",
        })
    nodes.append({
        "id": f"anchor-{anchor.id}",
        "label": (anchor.concepts or anchor.contexts or ["spiral event"])[0],
        "x": 38,
        "y": 46,
        "tone": "focus",
    })
    for index, leaf in enumerate(leaves):
        nodes.append({
            "id": f"anchor-leaf-{index}",
            "label": leaf,
            "x": 58 + index * 14,
            "y": 82,
            "tone": "outcome",
        })

    if precursor_a and precursor_b:
        edges.append({"source": f"pre-a-{precursor_a.id}", "target": f"pre-b-{precursor_b.id}", "style": "solid"})
        edges.append({"source": f"pre-b-{precursor_b.id}", "target": f"anchor-{anchor.id}", "style": "solid"})
    elif precursor_a:
        edges.append({"source": f"pre-a-{precursor_a.id}", "target": f"anchor-{anchor.id}", "style": "solid"})
    for index in range(len(leaves)):
        edges.append({"source": f"anchor-{anchor.id}", "target": f"anchor-leaf-{index}", "style": "dashed"})

    notes = [
        f"Selected event: {_title_case((anchor.concepts or anchor.contexts or ['spiral'])[0])}",
        f"{len(window_events)} linked moments found in the previous 48 hours",
    ]
    if precursor_a:
        notes.append(f"Common lead-in: {_title_case((precursor_a.concepts or precursor_a.contexts or ['build up'])[0])}")
    if precursor_b:
        notes.append(f"Immediate precursor: {_title_case((precursor_b.concepts or precursor_b.contexts or ['build up'])[0])}")

    return {
        "id": "precursor-window",
        "title": "48-Hour Build-Up",
        "kind": "timeline-graph",
        "question": "What was building in the 48 hours before a selected event?",
        "graph": {"nodes": nodes, "edges": edges},
        "timeline": [_timeline_item(event, _timeline_label(event, "short_date")) for event in window_events],
        "notes": notes,
        "sidebar": {
            "title": "Window summary",
            "items": [
                {"label": "anchor", "value": _title_case((anchor.concepts or anchor.contexts or ['spiral'])[0])},
                {"label": "emotion", "value": _title_case(anchor.emotions[0] if anchor.emotions else "unlabeled")},
                {"label": "behavior", "value": _title_case(anchor.behaviors[0] if anchor.behaviors else "unlabeled")},
            ],
        },
    }


def _weekly_pattern_view_card(events: list[DerivedEvent], metric_mode: str) -> dict:
    ordered = sorted(events, key=lambda event: event.occurred_at)
    if not ordered:
        return {
            "id": "weekly-pattern-view",
            "title": "Weekly Pattern View",
            "kind": "timeline-graph",
            "question": "How are themes recurring across the week?",
            "graph": {"nodes": [], "edges": []},
            "timeline": [],
            "notes": ["Not enough weekly data yet."],
            "sidebar": {"title": "Theme counts this week", "bars": []},
        }

    recent_window = ordered[-7:] if len(ordered) >= 7 else ordered
    theme_counter = Counter()
    for event in recent_window:
        theme_counter.update(event.concepts[:3] or event.contexts[:2])
    top_themes = [name for name, _ in theme_counter.most_common(5)]

    positions = [(40, 18), (58, 18), (66, 42), (32, 42), (40, 76)]
    graph_nodes = []
    graph_edges = []
    for index, theme in enumerate(top_themes):
        x, y = positions[index % len(positions)]
        graph_nodes.append({"id": f"theme-{index}", "label": theme, "x": x, "y": y, "tone": "focus" if index == 0 else "support"})
    if len(graph_nodes) >= 2:
        graph_edges.append({"source": graph_nodes[0]["id"], "target": graph_nodes[1]["id"], "style": "solid"})
    if len(graph_nodes) >= 3:
        graph_edges.append({"source": graph_nodes[0]["id"], "target": graph_nodes[2]["id"], "style": "dashed"})
    if len(graph_nodes) >= 4:
        graph_edges.append({"source": graph_nodes[0]["id"], "target": graph_nodes[3]["id"], "style": "dashed"})
    if len(graph_nodes) >= 5:
        graph_edges.append({"source": graph_nodes[2]["id"], "target": graph_nodes[4]["id"], "style": "solid"})
        graph_edges.append({"source": graph_nodes[3]["id"], "target": graph_nodes[4]["id"], "style": "solid"})

    bars = [{"label": theme, "value": round(count * (1 if metric_mode == "frequency" else 1.0), 2)} for theme, count in theme_counter.most_common(5)]
    notes = [
        f"Leading theme this week: {_title_case(top_themes[0])}" if top_themes else "No dominant theme yet.",
        "This makes the graph useful for recurrence, not just relationship structure.",
    ]
    return {
        "id": "weekly-pattern-view",
        "title": "Weekly Pattern View",
        "kind": "timeline-graph",
        "question": "How are themes recurring across the week?",
        "graph": {"nodes": graph_nodes, "edges": graph_edges},
        "timeline": [_timeline_item(event, _timeline_label(event), kind="weekly") for event in recent_window],
        "notes": notes,
        "sidebar": {
            "title": "Theme counts this week",
            "bars": bars,
        },
    }


def _relationship_dynamics_card(events: list[DerivedEvent]) -> dict:
    relationship_events = [
        event for event in sorted(events, key=lambda event: event.occurred_at)
        if "relationship" in event.contexts or "maya" in event.people
    ]
    if not relationship_events:
        return {
            "id": "relationship-dynamics",
            "title": "Partner Conflict Dynamics",
            "kind": "timeline-graph",
            "question": "What fight sequence keeps repeating, and how is it changing?",
            "graph": {"nodes": [], "edges": []},
            "timeline": [],
            "notes": ["No relationship-specific events found in the current filter slice."],
            "sidebar": {"title": "Repeated fight sequence", "items": []},
        }

    sequence_counter = Counter()
    for event in relationship_events:
        sequence = []
        concepts = set(event.concepts + event.contexts)
        behaviors = set(event.behaviors)
        if "conflict" in concepts or "criticism" in concepts:
            sequence.append("criticism")
        if {"defensiveness", "blame", "sarcasm"}.intersection(behaviors):
            sequence.append("defensive")
        if {"withdrawal", "shutdown", "silence", "distance"}.intersection(behaviors):
            sequence.append("shutdown")
        if {"apology", "repair", "vulnerability", "accountability"}.intersection(behaviors):
            sequence.append("repair")
        if len(sequence) >= 2:
            sequence_counter[tuple(sequence[:4])] += 1

    dominant = sequence_counter.most_common(1)[0][0] if sequence_counter else ("criticism", "defensive", "shutdown")
    dominant_count = sequence_counter[dominant] if sequence_counter else 0
    labels = list(dominant)
    positions = [(18, 24), (38, 24), (58, 44), (42, 78), (62, 78)]
    graph_nodes = []
    graph_edges = []
    for index, label in enumerate(labels[:5]):
        x, y = positions[index]
        graph_nodes.append({"id": f"rel-{index}", "label": label, "x": x, "y": y, "tone": "focus" if index == 0 else "support"})
        if index > 0:
            graph_edges.append({"source": f"rel-{index - 1}", "target": f"rel-{index}", "style": "solid" if index < 2 else "dashed"})

    recent_slice = relationship_events[-5:]
    repair_showing = any("repair" in event.behaviors or "apology" in event.behaviors for event in recent_slice[-2:])
    notes = [
        f"{' -> '.join(labels)} appeared {dominant_count} times in this slice",
        "Repair attempt appears later in newer conflicts." if repair_showing else "Repair still shows up inconsistently in newer conflicts.",
    ]
    return {
        "id": "relationship-dynamics",
        "title": "Partner Conflict Dynamics",
        "kind": "timeline-graph",
        "question": "What fight sequence keeps repeating, and how is it changing?",
        "graph": {"nodes": graph_nodes, "edges": graph_edges},
        "timeline": [_timeline_item(event, _timeline_label(event, "short_date"), kind="conflict") for event in recent_slice],
        "notes": notes,
        "sidebar": {
            "title": "Repeated fight sequence",
            "items": [
                {"label": "sequence", "value": " -> ".join(labels)},
                {"label": "count", "value": str(dominant_count)},
                {"label": "trend", "value": "repair later" if repair_showing else "repair uneven"},
            ],
        },
    }


def build_graph_tests_payload(*, workspace, filters: dict) -> dict:
    derived_events, graph_available = _derive_events_from_graph(workspace=workspace)
    filtered_events = _filter_events(derived_events, filters)
    metric_mode = "frequency"
    normalize = bool(filters.get("normalize", False))
    compare_by = filters.get("compare_by", "time_of_day")
    min_support = int(filters.get("min_support", 1) or 1)

    cards = [
        _pattern_story_matrix(filtered_events, metric_mode, normalize, min_support),
        _trigger_emotion_matrix(filtered_events, metric_mode, normalize, min_support),
        _emotion_behavior_matrix(filtered_events, metric_mode, normalize, min_support),
        _context_entanglement_matrix(filtered_events, metric_mode, normalize, min_support),
        _people_state_matrix(filtered_events, metric_mode, normalize, min_support),
        _beliefs_matrix(filtered_events, metric_mode, normalize, min_support),
        _works_matrix(filtered_events, metric_mode, normalize, min_support),
        _landscape_card(filtered_events, metric_mode),
        _compare_card(filtered_events, metric_mode, normalize, compare_by, min_support),
        _interconnection_chord_card(filtered_events, metric_mode, min_support),
        _precursor_window_card(filtered_events),
        _weekly_pattern_view_card(filtered_events, metric_mode),
        _relationship_dynamics_card(filtered_events),
    ]

    if filtered_events:
        date_start = min(event.occurred_at for event in filtered_events).isoformat()
        date_end = max(event.occurred_at for event in filtered_events).isoformat()
    else:
        date_start = None
        date_end = None

    return {
        "filters": {
            "time_range": filters.get("time_range", "30d"),
            "contexts": filters.get("contexts", []),
            "people": filters.get("people", []),
            "compare_by": compare_by,
            "pattern_trend": filters.get("pattern_trend") or filters.get("pattern_valence", "helping_me_recover"),
            "min_support": filters.get("min_support", 1),
            "normalize": normalize,
        },
        "meta": {
            "event_count": len(filtered_events),
            "graph_available": graph_available,
            "date_start": date_start,
            "date_end": date_end,
            "available_filters": _available_filters(derived_events),
        },
        "cards": cards,
    }
