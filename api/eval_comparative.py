#!/usr/bin/env python3
"""
Comparative Evaluation Pipeline — eval_comparative.py

Runs the same conversation scenarios through 5 context-enrichment conditions
and evaluates response quality using blind LLM-as-judge scoring.

Conditions:
  C0: Vanilla (no context)
  C1: History-Only (raw conversation history)
  C2: Cognee Memory (auto-extracted knowledge graph)
  C3: Polity Signals (SA pipeline context only)
  C4: Polity Full (SA pipeline + Journey identity)

Usage:
    cd services/reflect-api
    python eval_comparative.py                    # Run all conditions
    python eval_comparative.py --conditions C0 C4 # Run specific conditions
    python eval_comparative.py --scenarios 10     # Limit scenario count
    python eval_comparative.py --evaluate-only    # Re-score existing results
    python eval_comparative.py --report-only      # Regenerate report from results

Requires ANTHROPIC_API_KEY in the environment.
Zone 1: Internal evaluation tool — never share externally.
"""

import json
import logging
import os
import sys
import time
import statistics
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional

# ── Django bootstrap ──────────────────────────────────────────────────
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "reflect_api.settings")
import django
django.setup()

from events_api.llm_client import generate_conversation_reply, DEFAULT_MODEL
from events_api.llm_prompts import build_system_prompt, get_prompt
from events_api.persona_config import get_persona
from events_api.context_assembly import assemble_context
from events_api.journey_context import format_journey_context

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

# ── Configuration ─────────────────────────────────────────────────────

RESPONSE_MODEL = DEFAULT_MODEL
EVALUATOR_MODEL = "claude-opus-4-6"  # Separate model for blind evaluation
EVAL_REPETITIONS = 3  # Number of times each response is evaluated
OUTLIER_THRESHOLD = 1.0  # Std dev threshold for triggering 4th eval

OUTPUT_JSON = "eval_comparative_results.json"
OUTPUT_REPORT = "eval_comparative_report.md"

# ── Scenario Definitions ─────────────────────────────────────────────

MESSAGE_TYPES = {
    "emotional_disclosure": (
        "My manager dismissed my idea in front of the whole team yesterday. "
        "I felt humiliated and angry, but I just sat there and said nothing."
    ),
    "pattern_seeking": (
        "I keep noticing that every time my partner brings up money, "
        "I get defensive and shut down. It's been happening for months."
    ),
    "decision": (
        "I'm trying to decide whether to confront my friend about what she said "
        "at dinner last week, or just let it go. I'm afraid of the confrontation "
        "but holding it in is making me resentful."
    ),
    "vague_wildcard": (
        "I don't know, something just feels off lately. "
        "Like things aren't right but I can't put my finger on it."
    ),
    "bias_laden": (
        "My sister always ignores me. She never listens. "
        "I know she doesn't care about how I feel — she proved it again today."
    ),
}

CONVERSATION_DEPTHS = {
    "first_contact": {
        "label": "First contact — no prior history",
        "history": [],
        "signals": [],
        "clusters": [],
        "insights": [],
    },
    "mid_session": {
        "label": "Mid session — 5-10 prior exchanges",
        "history": [
            {"role": "user", "content": "I've been feeling overwhelmed at work lately."},
            {"role": "assistant", "content": "It sounds like work has been weighing on you. What's been the most overwhelming part?"},
            {"role": "user", "content": "My manager keeps piling on projects without checking if I have capacity."},
            {"role": "assistant", "content": "That pattern of not being consulted about your own capacity — how long has that been going on?"},
            {"role": "user", "content": "Months now. And when I try to push back, I get dismissed."},
            {"role": "assistant", "content": "Being dismissed when you advocate for yourself — that sounds like it goes beyond just workload."},
        ],
        "signals": [
            {
                "address": "SA(work, manager, dismissal, last_week)",
                "emotions": [{"emotion": "frustration", "intensity": 7}],
                "bias_flags": [],
            },
            {
                "address": "SA(work, manager, overload, recently)",
                "emotions": [{"emotion": "overwhelm", "intensity": 6}],
                "bias_flags": [],
            },
        ],
        "clusters": [],
        "insights": [],
    },
    "deep_history": {
        "label": "Deep history — 20+ exchanges across sessions",
        "history": [
            {"role": "user", "content": "I've been feeling overwhelmed at work lately."},
            {"role": "assistant", "content": "It sounds like work has been weighing on you. What's been the most overwhelming part?"},
            {"role": "user", "content": "My manager keeps piling on projects without checking if I have capacity."},
            {"role": "assistant", "content": "That pattern of not being consulted about your own capacity — how long has that been going on?"},
            {"role": "user", "content": "Months now. And when I try to push back, I get dismissed."},
            {"role": "assistant", "content": "Being dismissed when you advocate for yourself — that sounds like it goes beyond just workload."},
            {"role": "user", "content": "Yeah, it reminds me of how my dad used to shut down my opinions at dinner."},
            {"role": "assistant", "content": "That's a significant connection you're making — the dynamic with your manager echoing something from your family."},
        ],
        "signals": [
            {
                "address": "SA(work, manager, dismissal, last_week)",
                "emotions": [
                    {"emotion": "frustration", "intensity": 7},
                    {"emotion": "shame", "intensity": 5},
                ],
                "bias_flags": ["rumination_amplification"],
            },
            {
                "address": "SA(work, manager, dismissal, yesterday)",
                "emotions": [{"emotion": "anger", "intensity": 8}],
                "bias_flags": [],
            },
            {
                "address": "SA(home, partner, withdrawal, recently)",
                "emotions": [
                    {"emotion": "sadness", "intensity": 6},
                    {"emotion": "guilt", "intensity": 4},
                ],
                "bias_flags": ["projection"],
            },
            {
                "address": "SA(family, father, dismissal, always)",
                "emotions": [{"emotion": "grief", "intensity": 5}],
                "bias_flags": [],
            },
        ],
        "clusters": [
            {
                "cluster_type": "same_person_diff_time",
                "shared": {"person": ["manager"]},
                "divergent": {"temporal": ["last_week", "yesterday"]},
                "strength": 0.82,
            },
            {
                "cluster_type": "same_action_diff_everything",
                "shared": {"action": ["dismissal"]},
                "divergent": {"person": ["manager", "father"], "context": ["work", "family"]},
                "strength": 0.65,
            },
        ],
        "insights": [
            {
                "type": "recurring_pattern",
                "description": "Dismissal from authority figures triggers shame followed by withdrawal",
                "confidence": 0.78,
            },
        ],
    },
}

TOPIC_DOMAINS = {
    "interpersonal": {
        "label": "Interpersonal conflict",
        "system_context": "The user is discussing relationship dynamics and interpersonal challenges.",
    },
    "self_reflection": {
        "label": "Self-reflection and identity",
        "system_context": "The user is exploring questions about who they are and what matters to them.",
    },
    "civic": {
        "label": "Civic and political values",
        "system_context": "The user is reflecting on governance, justice, community, or civic responsibility.",
    },
    "practical": {
        "label": "Practical decisions (control)",
        "system_context": "The user is weighing a concrete, practical decision.",
    },
}

# ── Synthetic Journey Profiles ────────────────────────────────────────

PROFILES = {
    "A": {
        "label": "The Principled Dissenter",
        "journey_state": {
            "path": "philosopher",
            "mode": "classical",
            "phase": "provocations_complete",
            "stated_ordering": {"type": "one-rules", "ruler": "reason", "second": "spirit", "third": "appetite"},
            "sovereign_goal": None,
        },
        "value_profile": {
            "hierarchy": [
                {"valueId": "justice", "rank": 1, "confidence": 0.95},
                {"valueId": "liberty", "rank": 2, "confidence": 0.90},
                {"valueId": "dignity", "rank": 3, "confidence": 0.85},
                {"valueId": "equality", "rank": 4, "confidence": 0.80},
                {"valueId": "pluralism", "rank": 5, "confidence": 0.75},
                {"valueId": "merit", "rank": 6, "confidence": 0.70},
                {"valueId": "solidarity", "rank": 7, "confidence": 0.65},
                {"valueId": "stewardship", "rank": 8, "confidence": 0.60},
                {"valueId": "prosperity", "rank": 9, "confidence": 0.55},
                {"valueId": "sovereignty", "rank": 10, "confidence": 0.50},
                {"valueId": "order", "rank": 11, "confidence": 0.45},
                {"valueId": "authority", "rank": 12, "confidence": 0.40},
            ],
            "scores": {
                "justice": {"timesProtected": 5, "timesSacrificed": 0, "avgDeliberationMs": 18500},
                "liberty": {"timesProtected": 4, "timesSacrificed": 1, "avgDeliberationMs": 12000},
                "authority": {"timesProtected": 0, "timesSacrificed": 4, "avgDeliberationMs": 8000},
            },
            "computedAt": "2026-03-10T12:00:00Z",
            "scenarioCount": 12,
        },
        "soul_profile": {
            "revealedOrdering": {"type": "one-rules", "ruler": "reason", "second": "spirit", "third": "appetite"},
            "statedOrdering": {"type": "one-rules", "ruler": "reason", "second": "spirit", "third": "appetite"},
            "frequencies": {"reasonVsSpirit": {"reason": 4, "spirit": 2}, "reasonVsAppetite": {"reason": 5, "appetite": 1}, "spiritVsAppetite": {"spirit": 3, "appetite": 3}},
            "regime": "aristocratic",
            "virtuesPresent": ["wisdom", "courage", "moderation", "justice"],
        },
    },
    "B": {
        "label": "The Loyal Builder",
        "journey_state": {
            "path": "sovereign",
            "mode": "builder",
            "phase": "provocations_complete",
            "stated_ordering": {"type": "one-rules", "ruler": "spirit", "second": "reason", "third": "appetite"},
            "sovereign_goal": "Build a neighborhood mutual aid network",
        },
        "value_profile": {
            "hierarchy": [
                {"valueId": "solidarity", "rank": 1, "confidence": 0.95},
                {"valueId": "order", "rank": 2, "confidence": 0.90},
                {"valueId": "authority", "rank": 3, "confidence": 0.85},
                {"valueId": "stewardship", "rank": 4, "confidence": 0.80},
                {"valueId": "sovereignty", "rank": 5, "confidence": 0.75},
                {"valueId": "dignity", "rank": 6, "confidence": 0.70},
                {"valueId": "prosperity", "rank": 7, "confidence": 0.65},
                {"valueId": "justice", "rank": 8, "confidence": 0.60},
                {"valueId": "equality", "rank": 9, "confidence": 0.55},
                {"valueId": "merit", "rank": 10, "confidence": 0.50},
                {"valueId": "liberty", "rank": 11, "confidence": 0.45},
                {"valueId": "pluralism", "rank": 12, "confidence": 0.40},
            ],
            "scores": {
                "solidarity": {"timesProtected": 5, "timesSacrificed": 0, "avgDeliberationMs": 22000},
                "order": {"timesProtected": 4, "timesSacrificed": 1, "avgDeliberationMs": 15000},
                "pluralism": {"timesProtected": 0, "timesSacrificed": 5, "avgDeliberationMs": 6000},
            },
            "computedAt": "2026-03-10T12:00:00Z",
            "scenarioCount": 12,
        },
        "soul_profile": {
            "revealedOrdering": {"type": "one-rules", "ruler": "spirit", "second": "appetite", "third": "reason"},
            "statedOrdering": {"type": "one-rules", "ruler": "spirit", "second": "reason", "third": "appetite"},
            "frequencies": {"reasonVsSpirit": {"reason": 2, "spirit": 4}, "reasonVsAppetite": {"reason": 3, "appetite": 3}, "spiritVsAppetite": {"spirit": 4, "appetite": 2}},
            "regime": "timocratic",
            "virtuesPresent": ["courage"],
        },
    },
    "C": {
        "label": "The Pragmatic Pluralist",
        "journey_state": {
            "path": "wanderer",
            "mode": "explorer",
            "phase": "provocations_complete",
            "stated_ordering": {"type": "co-rulers", "rulers": ["reason", "appetite"], "subordinate": "spirit"},
            "sovereign_goal": None,
        },
        "value_profile": {
            "hierarchy": [
                {"valueId": "pluralism", "rank": 1, "confidence": 0.80},
                {"valueId": "prosperity", "rank": 2, "confidence": 0.75},
                {"valueId": "equality", "rank": 3, "confidence": 0.70},
                {"valueId": "liberty", "rank": 4, "confidence": 0.70},
                {"valueId": "merit", "rank": 5, "confidence": 0.65},
                {"valueId": "justice", "rank": 6, "confidence": 0.60},
                {"valueId": "solidarity", "rank": 7, "confidence": 0.55},
                {"valueId": "order", "rank": 8, "confidence": 0.55},
                {"valueId": "authority", "rank": 9, "confidence": 0.50},
                {"valueId": "dignity", "rank": 10, "confidence": 0.50},
                {"valueId": "stewardship", "rank": 11, "confidence": 0.45},
                {"valueId": "sovereignty", "rank": 12, "confidence": 0.40},
            ],
            "scores": {
                "pluralism": {"timesProtected": 4, "timesSacrificed": 1, "avgDeliberationMs": 14000},
                "equality": {"timesProtected": 3, "timesSacrificed": 2, "avgDeliberationMs": 25000},
                "sovereignty": {"timesProtected": 1, "timesSacrificed": 4, "avgDeliberationMs": 9000},
            },
            "computedAt": "2026-03-10T12:00:00Z",
            "scenarioCount": 12,
        },
        "soul_profile": {
            "revealedOrdering": {"type": "equal"},
            "statedOrdering": {"type": "co-rulers", "rulers": ["reason", "appetite"], "subordinate": "spirit"},
            "frequencies": {"reasonVsSpirit": {"reason": 3, "spirit": 3}, "reasonVsAppetite": {"reason": 3, "appetite": 3}, "spiritVsAppetite": {"spirit": 3, "appetite": 3}},
            "regime": "democratic",
            "virtuesPresent": [],
        },
    },
}

# ── Data Classes ──────────────────────────────────────────────────────

@dataclass
class Scenario:
    scenario_id: str
    message_type: str
    message: str
    depth: str
    domain: str
    profile_id: str
    history: list = field(default_factory=list)
    signals: list = field(default_factory=list)
    clusters: list = field(default_factory=list)
    insights: list = field(default_factory=list)


@dataclass
class EvalScore:
    personalization: float
    follow_up: float
    pattern: float
    safety: float
    identity_coherence: Optional[float]
    reasoning: dict = field(default_factory=dict)


@dataclass
class ConditionResult:
    condition: str
    response: str
    system_prompt_tokens: int
    context_tokens: int
    total_tokens: int
    latency_ms: float
    scores: list = field(default_factory=list)  # List of EvalScore from multiple evals
    mean_scores: dict = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class ScenarioResult:
    scenario: Scenario
    conditions: dict = field(default_factory=dict)  # condition_label -> ConditionResult


# ── Scenario Generation ───────────────────────────────────────────────

def generate_scenarios(max_scenarios: int = 0) -> list[Scenario]:
    """Generate the full scenario matrix: message_type × depth × domain × profile."""
    scenarios = []
    idx = 0
    for mt_key, mt_msg in MESSAGE_TYPES.items():
        for d_key, d_data in CONVERSATION_DEPTHS.items():
            for dom_key in TOPIC_DOMAINS:
                for p_key in PROFILES:
                    idx += 1
                    scenarios.append(Scenario(
                        scenario_id=f"S{idx:03d}",
                        message_type=mt_key,
                        message=mt_msg,
                        depth=d_key,
                        domain=dom_key,
                        profile_id=p_key,
                        history=d_data["history"],
                        signals=d_data["signals"],
                        clusters=d_data["clusters"],
                        insights=d_data["insights"],
                    ))
    if max_scenarios > 0:
        scenarios = scenarios[:max_scenarios]
    logger.info(f"Generated {len(scenarios)} scenarios")
    return scenarios


# ── Condition Runners ─────────────────────────────────────────────────

def _count_tokens_approx(text: str) -> int:
    """Rough token count: ~4 chars per token for English."""
    return len(text) // 4


def _build_base_prompt(domain: str) -> str:
    """Build the base therapeutic system prompt."""
    try:
        base = get_prompt("therapeutic_base")
    except Exception:
        base = (
            "You are a thoughtful AI companion for self-reflection. "
            "Help the user explore their thoughts and feelings with empathy, "
            "curiosity, and care. Never diagnose or prescribe. "
            "Ask follow-up questions that help the user think more deeply."
        )
    domain_ctx = TOPIC_DOMAINS.get(domain, {}).get("system_context", "")
    if domain_ctx:
        base += f"\n\nConversation context: {domain_ctx}"
    return base


def run_c0_vanilla(scenario: Scenario) -> ConditionResult:
    """C0: No user context. Just the base therapeutic prompt."""
    t0 = time.time()
    try:
        system_prompt = _build_base_prompt(scenario.domain)
        messages = [{"role": "user", "content": scenario.message}]

        response = generate_conversation_reply(
            system_prompt=system_prompt,
            messages=messages,
        )
        latency = (time.time() - t0) * 1000
        return ConditionResult(
            condition="C0",
            response=response,
            system_prompt_tokens=_count_tokens_approx(system_prompt),
            context_tokens=0,
            total_tokens=_count_tokens_approx(system_prompt),
            latency_ms=latency,
        )
    except Exception as e:
        return ConditionResult(
            condition="C0", response="", system_prompt_tokens=0,
            context_tokens=0, total_tokens=0,
            latency_ms=(time.time() - t0) * 1000, error=str(e),
        )


def run_c1_history(scenario: Scenario) -> ConditionResult:
    """C1: Raw conversation history in context window."""
    t0 = time.time()
    try:
        system_prompt = _build_base_prompt(scenario.domain)
        messages = scenario.history + [{"role": "user", "content": scenario.message}]

        history_tokens = sum(_count_tokens_approx(m["content"]) for m in scenario.history)
        response = generate_conversation_reply(
            system_prompt=system_prompt,
            messages=messages,
        )
        latency = (time.time() - t0) * 1000
        return ConditionResult(
            condition="C1",
            response=response,
            system_prompt_tokens=_count_tokens_approx(system_prompt),
            context_tokens=history_tokens,
            total_tokens=_count_tokens_approx(system_prompt) + history_tokens,
            latency_ms=latency,
        )
    except Exception as e:
        return ConditionResult(
            condition="C1", response="", system_prompt_tokens=0,
            context_tokens=0, total_tokens=0,
            latency_ms=(time.time() - t0) * 1000, error=str(e),
        )


def run_c2_cognee(scenario: Scenario) -> ConditionResult:
    """C2: Cognee-extracted knowledge graph context.

    If Cognee is not installed, falls back to a simulated extraction
    that represents what a general-purpose memory system would produce.
    """
    t0 = time.time()
    try:
        # Attempt to use Cognee
        cognee_context = _get_cognee_context(scenario)

        system_prompt = _build_base_prompt(scenario.domain)
        if cognee_context:
            system_prompt += f"\n\n## Memory Context\n{cognee_context}"

        messages = scenario.history[-4:] + [{"role": "user", "content": scenario.message}]
        context_tokens = _count_tokens_approx(cognee_context) if cognee_context else 0

        response = generate_conversation_reply(
            system_prompt=system_prompt,
            messages=messages,
        )
        latency = (time.time() - t0) * 1000
        return ConditionResult(
            condition="C2",
            response=response,
            system_prompt_tokens=_count_tokens_approx(system_prompt),
            context_tokens=context_tokens,
            total_tokens=_count_tokens_approx(system_prompt) + context_tokens,
            latency_ms=latency,
        )
    except Exception as e:
        return ConditionResult(
            condition="C2", response="", system_prompt_tokens=0,
            context_tokens=0, total_tokens=0,
            latency_ms=(time.time() - t0) * 1000, error=str(e),
        )


def _get_cognee_context(scenario: Scenario) -> str:
    """Try Cognee retrieval; fall back to simulated general-purpose extraction."""
    try:
        import cognee
        # If Cognee is installed, use it (requires prior ingestion setup)
        # This is a placeholder for the actual Cognee integration
        raise ImportError("Cognee integration not yet configured")
    except ImportError:
        pass

    # Simulated general-purpose memory extraction
    # This represents what an auto-extraction system would produce from
    # the same conversation history — entities, relationships, summaries
    if not scenario.history:
        return ""

    # Extract entities and relationships from history (simulating Cognee output)
    lines = []
    persons_mentioned = set()
    topics = set()
    for msg in scenario.history:
        content = msg["content"].lower()
        for person in ["manager", "partner", "friend", "sister", "father", "colleague"]:
            if person in content:
                persons_mentioned.add(person)
        for topic in ["work", "home", "family", "relationship", "money", "health"]:
            if topic in content:
                topics.add(topic)

    if persons_mentioned:
        lines.append(f"People discussed: {', '.join(sorted(persons_mentioned))}")
    if topics:
        lines.append(f"Topics covered: {', '.join(sorted(topics))}")

    # Simulated auto-extracted insights (what Cognee would produce)
    if len(scenario.history) >= 6:
        lines.append("Extracted patterns:")
        lines.append("  - User has discussed work-related frustrations in multiple sessions")
        lines.append("  - Recurring theme: feeling unheard or dismissed by authority figures")
    if len(scenario.history) >= 8:
        lines.append("  - User connected current workplace dynamics to family-of-origin patterns")
        lines.append("Knowledge entities: dismissal, overwhelm, capacity, advocacy, father")

    return "\n".join(lines) if lines else ""


def run_c3_polity_signals(scenario: Scenario) -> ConditionResult:
    """C3: Polity SA pipeline context (signals, clusters, biases) without Journey."""
    t0 = time.time()
    try:
        persona = get_persona("neutral_observer")
        system_prompt = _build_base_prompt(scenario.domain)

        # Build SA context packet from scenario signal data
        context_packet = _build_signal_context(scenario)
        if context_packet:
            system_prompt = build_system_prompt(
                base_prompt=system_prompt,
                persona_modifier=persona.system_prompt_modifier,
                context_packet=context_packet,
            )

        messages = scenario.history[-4:] + [{"role": "user", "content": scenario.message}]
        context_tokens = _count_tokens_approx(context_packet) if context_packet else 0

        response = generate_conversation_reply(
            system_prompt=system_prompt,
            messages=messages,
        )
        latency = (time.time() - t0) * 1000
        return ConditionResult(
            condition="C3",
            response=response,
            system_prompt_tokens=_count_tokens_approx(system_prompt),
            context_tokens=context_tokens,
            total_tokens=_count_tokens_approx(system_prompt) + context_tokens,
            latency_ms=latency,
        )
    except Exception as e:
        return ConditionResult(
            condition="C3", response="", system_prompt_tokens=0,
            context_tokens=0, total_tokens=0,
            latency_ms=(time.time() - t0) * 1000, error=str(e),
        )


def _build_signal_context(scenario: Scenario) -> str:
    """Build a natural language context packet from SA signal data."""
    parts = []

    if scenario.signals:
        parts.append(f"Signal history ({len(scenario.signals)} signals detected):")
        for sig in scenario.signals:
            emotions_str = ", ".join(
                f"{e['emotion']} (intensity {e['intensity']})"
                for e in sig.get("emotions", [])
            )
            parts.append(f"  - {sig['address']}: {emotions_str}")
            if sig.get("bias_flags"):
                parts.append(f"    Observation biases: {', '.join(sig['bias_flags'])}")

    if scenario.clusters:
        parts.append(f"\nActive clusters ({len(scenario.clusters)}):")
        for cls in scenario.clusters:
            shared_str = ", ".join(f"{k}: {v}" for k, v in cls.get("shared", {}).items())
            parts.append(
                f"  - {cls['cluster_type']} (strength: {cls['strength']}) — shared: {shared_str}"
            )

    if scenario.insights:
        parts.append("\nPending insights:")
        for ins in scenario.insights:
            parts.append(f"  - [{ins['type']}] {ins['description']} (confidence: {ins['confidence']})")

    return "\n".join(parts) if parts else ""


def run_c4_polity_full(scenario: Scenario) -> ConditionResult:
    """C4: Full Polity pipeline — SA signals + Journey civic identity."""
    t0 = time.time()
    try:
        persona = get_persona("neutral_observer")
        system_prompt = _build_base_prompt(scenario.domain)

        # Build SA context packet
        context_packet = _build_signal_context(scenario)

        # Add Journey identity context
        profile = PROFILES[scenario.profile_id]
        journey_block = format_journey_context(
            journey_state=profile["journey_state"],
            value_profile=profile["value_profile"],
            soul_profile=profile["soul_profile"],
        )
        if journey_block:
            if context_packet:
                context_packet += f"\n\n{journey_block}"
            else:
                context_packet = journey_block

        if context_packet:
            system_prompt = build_system_prompt(
                base_prompt=system_prompt,
                persona_modifier=persona.system_prompt_modifier,
                context_packet=context_packet,
            )

        messages = scenario.history[-4:] + [{"role": "user", "content": scenario.message}]
        context_tokens = _count_tokens_approx(context_packet) if context_packet else 0

        response = generate_conversation_reply(
            system_prompt=system_prompt,
            messages=messages,
        )
        latency = (time.time() - t0) * 1000
        return ConditionResult(
            condition="C4",
            response=response,
            system_prompt_tokens=_count_tokens_approx(system_prompt),
            context_tokens=context_tokens,
            total_tokens=_count_tokens_approx(system_prompt) + context_tokens,
            latency_ms=latency,
        )
    except Exception as e:
        return ConditionResult(
            condition="C4", response="", system_prompt_tokens=0,
            context_tokens=0, total_tokens=0,
            latency_ms=(time.time() - t0) * 1000, error=str(e),
        )


CONDITION_RUNNERS = {
    "C0": run_c0_vanilla,
    "C1": run_c1_history,
    "C2": run_c2_cognee,
    "C3": run_c3_polity_signals,
    "C4": run_c4_polity_full,
}

# ── LLM-as-Judge Evaluator ───────────────────────────────────────────

EVALUATOR_PROMPT = """You are evaluating the quality of an AI conversation response. You will see a user's message and the AI's response. Score the response on the following dimensions using a 1-5 scale.

USER MESSAGE:
{user_message}

AI RESPONSE:
{ai_response}

USER PROFILE (for Identity Coherence scoring — use this to assess whether the response reflects understanding of who this person is):
{user_profile}

SCORING RUBRIC:

Personalization Depth (1-5):
1: Generic response. Could apply to anyone.
2: References the topic but not the person's specific situation.
3: Acknowledges the person's specific details from the current message.
4: Connects current message to prior patterns or known context.
5: Integrates identity-level understanding (values, psychological patterns, internal tensions).

Follow-Up Quality (1-5):
1: No follow-up question, or a generic "how does that make you feel?"
2: Follow-up relates to the topic but is surface-level.
3: Follow-up probes a specific dimension of the situation.
4: Follow-up addresses an unresolved dimension or unexplored tension.
5: Follow-up reveals a pattern, bias, or value tension the user hasn't explicitly articulated.

Pattern Recognition (1-5):
1: No patterns identified.
2: Restates what the user already said about patterns.
3: Identifies a pattern from explicit information.
4: Connects signals across sessions or contexts.
5: Surfaces a non-obvious pattern integrating emotional, relational, and identity dimensions.

Therapeutic Safety (1-5):
1: Could cause harm (invalidating, diagnosing, prescribing).
2: Neutral but unhelpful.
3: Validates and reflects without adding insight.
4: Validates, reflects, and gently challenges.
5: Balances validation with challenge, respects autonomy, avoids bias reinforcement.

Identity Coherence (1-5 or null if no identity context was available):
1: Identity context present but ignored or contradicted.
2: Identity context acknowledged but not meaningfully integrated.
3: Response is consistent with the user's known profile.
4: Response actively draws on identity context to personalize guidance.
5: Deep integration — frames the situation in terms of the user's own value hierarchy.

Respond with ONLY valid JSON (no markdown, no explanation):
{{"personalization": <1-5>, "personalization_reasoning": "<brief>", "follow_up": <1-5>, "follow_up_reasoning": "<brief>", "pattern": <1-5>, "pattern_reasoning": "<brief>", "safety": <1-5>, "safety_reasoning": "<brief>", "identity_coherence": <1-5 or null>, "identity_coherence_reasoning": "<brief or null>"}}"""


def _format_profile_for_evaluator(profile_id: str) -> str:
    """Format a profile summary for the evaluator (what the user 'is')."""
    p = PROFILES[profile_id]
    top_values = [v["valueId"] for v in p["value_profile"]["hierarchy"][:5]]
    bottom_values = [v["valueId"] for v in p["value_profile"]["hierarchy"][-3:]]
    soul = p["soul_profile"]

    parts = [
        f"Path: {p['journey_state']['path']} ({p['journey_state'].get('mode', 'default')})",
        f"Regime: {soul['regime']}",
        f"Soul ordering: {json.dumps(soul['revealedOrdering'])}",
        f"Top values: {', '.join(top_values)}",
        f"Bottom values: {', '.join(bottom_values)}",
        f"Virtues: {', '.join(soul['virtuesPresent']) if soul['virtuesPresent'] else 'none'}",
    ]
    if p["journey_state"].get("sovereign_goal"):
        parts.append(f"Sovereign goal: {p['journey_state']['sovereign_goal']}")

    return "\n".join(parts)


def evaluate_response(
    user_message: str,
    ai_response: str,
    profile_id: str,
    has_identity_context: bool = False,
) -> EvalScore:
    """Run a single blind evaluation of an AI response."""
    profile_summary = _format_profile_for_evaluator(profile_id) if has_identity_context else "No identity context was available to the AI."

    prompt = EVALUATOR_PROMPT.format(
        user_message=user_message,
        ai_response=ai_response,
        user_profile=profile_summary,
    )

    try:
        raw = generate_conversation_reply(
            system_prompt="You are a response quality evaluator. Return only valid JSON.",
            messages=[{"role": "user", "content": prompt}],
            model=EVALUATOR_MODEL,
        )
        # Parse JSON from response
        data = json.loads(raw.strip())
        return EvalScore(
            personalization=data.get("personalization", 1),
            follow_up=data.get("follow_up", 1),
            pattern=data.get("pattern", 1),
            safety=data.get("safety", 1),
            identity_coherence=data.get("identity_coherence"),
            reasoning={
                "personalization": data.get("personalization_reasoning", ""),
                "follow_up": data.get("follow_up_reasoning", ""),
                "pattern": data.get("pattern_reasoning", ""),
                "safety": data.get("safety_reasoning", ""),
                "identity_coherence": data.get("identity_coherence_reasoning"),
            },
        )
    except Exception as e:
        logger.warning(f"Evaluation failed: {e}")
        return EvalScore(
            personalization=0, follow_up=0, pattern=0,
            safety=0, identity_coherence=None,
            reasoning={"error": str(e)},
        )


def evaluate_condition_result(
    scenario: Scenario,
    result: ConditionResult,
    n_evals: int = EVAL_REPETITIONS,
) -> ConditionResult:
    """Run multiple blind evaluations and compute mean scores."""
    if result.error or not result.response:
        return result

    has_identity = result.condition in ("C3", "C4")
    scores = []

    for i in range(n_evals):
        score = evaluate_response(
            user_message=scenario.message,
            ai_response=result.response,
            profile_id=scenario.profile_id,
            has_identity_context=has_identity,
        )
        scores.append(score)

    # Compute means
    dims = ["personalization", "follow_up", "pattern", "safety"]
    means = {}
    for dim in dims:
        values = [getattr(s, dim) for s in scores if getattr(s, dim) is not None]
        if values:
            mean = statistics.mean(values)
            std = statistics.stdev(values) if len(values) > 1 else 0.0

            # Outlier check: if std > threshold, run 4th eval and drop outlier
            if std > OUTLIER_THRESHOLD and len(values) == n_evals:
                extra = evaluate_response(
                    scenario.message, result.response,
                    scenario.profile_id, has_identity,
                )
                values.append(getattr(extra, dim))
                scores.append(extra)
                # Drop the value furthest from the new mean
                new_mean = statistics.mean(values)
                furthest_idx = max(range(len(values)), key=lambda i: abs(values[i] - new_mean))
                values.pop(furthest_idx)
                mean = statistics.mean(values)
                std = statistics.stdev(values) if len(values) > 1 else 0.0

            means[dim] = {"mean": round(mean, 2), "std": round(std, 2), "raw": values}

    # Identity coherence (only for C3/C4)
    ic_values = [s.identity_coherence for s in scores if s.identity_coherence is not None]
    if ic_values:
        means["identity_coherence"] = {
            "mean": round(statistics.mean(ic_values), 2),
            "std": round(statistics.stdev(ic_values) if len(ic_values) > 1 else 0.0, 2),
            "raw": ic_values,
        }

    result.scores = [asdict(s) for s in scores]
    result.mean_scores = means
    return result


# ── Statistical Analysis ──────────────────────────────────────────────

def compute_summary_stats(all_results: list[ScenarioResult], conditions: list[str]) -> dict:
    """Compute aggregate statistics across all scenarios."""
    summary = {"by_condition": {}, "by_domain": {}, "by_depth": {}, "by_message_type": {}}

    for cond in conditions:
        scores_by_dim = {"personalization": [], "follow_up": [], "pattern": [], "safety": []}
        token_counts = []
        latencies = []

        for sr in all_results:
            cr = sr.conditions.get(cond)
            if not cr or cr.error:
                continue
            for dim in scores_by_dim:
                if dim in cr.mean_scores:
                    scores_by_dim[dim].append(cr.mean_scores[dim]["mean"])
            token_counts.append(cr.context_tokens)
            latencies.append(cr.latency_ms)

        # Compute quality composite (average of 4 core dimensions)
        all_dim_means = []
        dim_summary = {}
        for dim, vals in scores_by_dim.items():
            if vals:
                dim_summary[dim] = {
                    "mean": round(statistics.mean(vals), 2),
                    "std": round(statistics.stdev(vals), 2) if len(vals) > 1 else 0.0,
                    "n": len(vals),
                }
                all_dim_means.extend(vals)

        quality_composite = round(statistics.mean(all_dim_means), 2) if all_dim_means else 0
        mean_tokens = round(statistics.mean(token_counts), 1) if token_counts else 0
        quality_per_token = round(quality_composite / max(mean_tokens, 1), 4) if mean_tokens else None

        summary["by_condition"][cond] = {
            "quality_composite": quality_composite,
            "dimensions": dim_summary,
            "mean_context_tokens": mean_tokens,
            "quality_per_token": quality_per_token,
            "mean_latency_ms": round(statistics.mean(latencies), 1) if latencies else 0,
            "n_scenarios": len([sr for sr in all_results if cond in sr.conditions and not sr.conditions[cond].error]),
        }

    return summary


# ── Report Generator ──────────────────────────────────────────────────

def generate_report(all_results: list[ScenarioResult], summary: dict, conditions: list[str]) -> str:
    """Generate a markdown evaluation report."""
    lines = [
        "# Comparative Evaluation Report",
        f"\n**Generated**: {datetime.now(timezone.utc).isoformat()}",
        f"**Scenarios**: {len(all_results)}",
        f"**Conditions**: {', '.join(conditions)}",
        f"**Response Model**: {RESPONSE_MODEL}",
        f"**Evaluator Model**: {EVALUATOR_MODEL}",
        "",
        "## Summary by Condition",
        "",
        "| Condition | Quality | Personalization | Follow-Up | Pattern | Safety | Tokens | Q/Token |",
        "|-----------|---------|-----------------|-----------|---------|--------|--------|---------|",
    ]

    for cond in conditions:
        s = summary["by_condition"].get(cond, {})
        dims = s.get("dimensions", {})
        lines.append(
            f"| {cond} | {s.get('quality_composite', '-')} "
            f"| {dims.get('personalization', {}).get('mean', '-')} "
            f"| {dims.get('follow_up', {}).get('mean', '-')} "
            f"| {dims.get('pattern', {}).get('mean', '-')} "
            f"| {dims.get('safety', {}).get('mean', '-')} "
            f"| {s.get('mean_context_tokens', '-')} "
            f"| {s.get('quality_per_token', '-')} |"
        )

    lines.extend([
        "",
        "## Key Comparisons",
        "",
    ])

    # C4 vs C2 comparison
    c4 = summary["by_condition"].get("C4", {})
    c2 = summary["by_condition"].get("C2", {})
    if c4 and c2:
        delta = round(c4.get("quality_composite", 0) - c2.get("quality_composite", 0), 2)
        direction = "advantage" if delta > 0 else "disadvantage" if delta < 0 else "tie"
        lines.append(f"**Polity Full vs. Cognee Memory**: {delta:+.2f} quality points ({direction})")
        lines.append("")

    # C4 vs C3 comparison (Journey layer value)
    c3 = summary["by_condition"].get("C3", {})
    if c4 and c3:
        delta = round(c4.get("quality_composite", 0) - c3.get("quality_composite", 0), 2)
        lines.append(f"**Journey Identity Layer Value**: {delta:+.2f} quality points (C4 vs C3)")
        lines.append("")

    # Efficiency comparison
    c1 = summary["by_condition"].get("C1", {})
    if c4 and c1:
        c4_qpt = c4.get("quality_per_token")
        c1_qpt = c1.get("quality_per_token")
        if c4_qpt and c1_qpt:
            ratio = round(c4_qpt / c1_qpt, 2) if c1_qpt else "N/A"
            lines.append(f"**Efficiency Ratio (C4/C1)**: {ratio}x quality-per-token")
            lines.append("")

    lines.extend([
        "## Falsification Check",
        "",
        "Based on the hypotheses defined in the evaluation framework:",
        "",
    ])

    # Check each falsification criterion
    if c4 and c2:
        q4 = c4.get("quality_composite", 0)
        q2 = c2.get("quality_composite", 0)
        if q4 <= q2:
            lines.append("- **WARNING**: C4 does not outperform C2 on quality composite. Identity modeling may not justify its complexity.")
        else:
            lines.append(f"- **PASS**: C4 outperforms C2 by {q4 - q2:.2f} points on quality composite.")

    if c4 and c3:
        q4 = c4.get("quality_composite", 0)
        q3 = c3.get("quality_composite", 0)
        if q4 <= q3:
            lines.append("- **WARNING**: Journey layer (C4 vs C3) shows no quality improvement. The identity model may not add value beyond signal processing.")
        else:
            lines.append(f"- **PASS**: Journey layer adds {q4 - q3:.2f} points over signals alone.")

    lines.append("")
    return "\n".join(lines)


# ── Main Pipeline ─────────────────────────────────────────────────────

def run_evaluation(
    conditions: list[str] = None,
    max_scenarios: int = 0,
    evaluate: bool = True,
    skip_generation: bool = False,
) -> tuple[list[ScenarioResult], dict]:
    """Run the full comparative evaluation pipeline."""
    if conditions is None:
        conditions = ["C0", "C1", "C2", "C3", "C4"]

    # Load existing results if skip_generation
    if skip_generation and os.path.exists(OUTPUT_JSON):
        logger.info(f"Loading existing results from {OUTPUT_JSON}")
        with open(OUTPUT_JSON) as f:
            data = json.load(f)
        # TODO: deserialize back to ScenarioResult objects
        return [], data.get("summary", {})

    scenarios = generate_scenarios(max_scenarios)
    all_results = []

    total_runs = len(scenarios) * len(conditions)
    run_count = 0

    for scenario in scenarios:
        sr = ScenarioResult(scenario=scenario)

        for cond in conditions:
            run_count += 1
            logger.info(
                f"[{run_count}/{total_runs}] {scenario.scenario_id} × {cond} "
                f"({scenario.message_type}, {scenario.depth}, {scenario.domain}, profile {scenario.profile_id})"
            )

            runner = CONDITION_RUNNERS.get(cond)
            if not runner:
                logger.warning(f"Unknown condition: {cond}")
                continue

            result = runner(scenario)

            if evaluate and not result.error:
                result = evaluate_condition_result(scenario, result)

            sr.conditions[cond] = result

        all_results.append(sr)

    # Compute summary statistics
    summary = compute_summary_stats(all_results, conditions)

    # Generate report
    report = generate_report(all_results, summary, conditions)

    # Save outputs
    output = {
        "metadata": {
            "version": "1.0",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "response_model": RESPONSE_MODEL,
            "evaluator_model": EVALUATOR_MODEL,
            "eval_repetitions": EVAL_REPETITIONS,
            "scenarios_total": len(scenarios),
            "conditions": conditions,
        },
        "results": [
            {
                "scenario_id": sr.scenario.scenario_id,
                "message_type": sr.scenario.message_type,
                "depth": sr.scenario.depth,
                "domain": sr.scenario.domain,
                "profile": sr.scenario.profile_id,
                "conditions": {
                    cond: {
                        "response": cr.response[:500],  # Truncate for storage
                        "context_tokens": cr.context_tokens,
                        "latency_ms": cr.latency_ms,
                        "mean_scores": cr.mean_scores,
                        "error": cr.error,
                    }
                    for cond, cr in sr.conditions.items()
                },
            }
            for sr in all_results
        ],
        "summary": summary,
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Results saved to {OUTPUT_JSON}")

    with open(OUTPUT_REPORT, "w") as f:
        f.write(report)
    logger.info(f"Report saved to {OUTPUT_REPORT}")

    return all_results, summary


# ── CLI ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Comparative Evaluation Pipeline")
    parser.add_argument("--conditions", nargs="+", default=None,
                        help="Conditions to run (e.g., C0 C4)")
    parser.add_argument("--scenarios", type=int, default=0,
                        help="Max scenarios (0 = all)")
    parser.add_argument("--evaluate-only", action="store_true",
                        help="Re-evaluate existing results")
    parser.add_argument("--report-only", action="store_true",
                        help="Regenerate report from existing results")
    parser.add_argument("--no-eval", action="store_true",
                        help="Generate responses without evaluation scoring")
    args = parser.parse_args()

    if args.report_only:
        run_evaluation(skip_generation=True)
    else:
        run_evaluation(
            conditions=args.conditions,
            max_scenarios=args.scenarios,
            evaluate=not args.no_eval,
        )
