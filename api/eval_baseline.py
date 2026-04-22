#!/usr/bin/env python3
"""
ThriveSight LLM Quality Baseline — eval_baseline.py

Exports 20 conversation examples + 10 signal generation examples across:
  - 3 personas × 4 templates × 3 graph context states × 5 message types
Runs them through the actual pipeline and captures full prompt→output tuples.

Outputs:
  - eval_results.json — structured data for programmatic analysis
  - eval_report.md  — human-readable markdown report

Usage:
    cd apps/api
    python eval_baseline.py

Requires ANTHROPIC_API_KEY in the environment.
Zone 1: Internal evaluation tool — never share externally.
"""

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone

# ── Django bootstrap ──────────────────────────────────────────────────
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "reflect_api.settings")

import django
django.setup()

from events_api.llm_client import (
    generate_conversation_reply,
    generate_signal_from_message,
    DEFAULT_MODEL,
)
from events_api.llm_prompts import build_system_prompt, get_prompt
from events_api.persona_config import PERSONAS, get_persona
from events_api.context_assembly import assemble_context

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

# ──────────────────────────────────────────────────────────────────────
# Scenario Definitions
# ──────────────────────────────────────────────────────────────────────

# Five message types representing different conversation entry points
MESSAGES = {
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

# Three graph context states: empty, light history, rich history
# These are mock context packets that simulate what assemble_context() would produce

GRAPH_CONTEXTS = {
    "empty": {
        "label": "Empty graph (new user)",
        "entities": {"persons": [], "contexts": [], "actions": [], "temporal": []},
        "signals": None,
        "clusters": None,
        "insights": None,
        "bias_flags": None,
    },
    "light": {
        "label": "Light history (2-3 signals)",
        "entities": {
            "persons": [{"mention": "manager", "normalized": "manager", "role": "manager", "confidence": 0.8}],
            "contexts": [{"mention": "work", "context": "work", "confidence": 0.8}],
            "actions": [],
            "temporal": [],
        },
        "signals": [
            {
                "uid": "sig-001",
                "address": "SA(work, manager, dismissal, last_week)",
                "emotions": [{"emotion": "frustration", "intensity": 6, "confidence": 0.8}],
                "confidence": 0.75,
                "bias_flags": [],
                "preview": "Felt frustrated when my manager shot down the proposal without reading it",
            },
            {
                "uid": "sig-002",
                "address": "SA(work, colleague, validation, yesterday)",
                "emotions": [{"emotion": "relief", "intensity": 4, "confidence": 0.7}],
                "confidence": 0.7,
                "bias_flags": [],
                "preview": "My teammate actually backed me up during the standup",
            },
        ],
        "clusters": None,
        "insights": None,
        "bias_flags": None,
    },
    "rich": {
        "label": "Rich history (signals + cluster + bias + insight)",
        "entities": {
            "persons": [
                {"mention": "manager", "normalized": "manager", "role": "manager", "confidence": 0.8},
                {"mention": "partner", "normalized": "partner", "role": "partner", "confidence": 0.8},
            ],
            "contexts": [
                {"mention": "work", "context": "work", "confidence": 0.8},
                {"mention": "home", "context": "home", "confidence": 0.8},
            ],
            "actions": [{"mention": "dismissed", "category": "dismissal", "confidence": 0.7}],
            "temporal": [{"mention": "yesterday", "temporal": "yesterday", "confidence": 0.8}],
        },
        "signals": [
            {
                "uid": "sig-001",
                "address": "SA(work, manager, dismissal, last_week)",
                "emotions": [
                    {"emotion": "frustration", "intensity": 7, "confidence": 0.85},
                    {"emotion": "shame", "intensity": 5, "confidence": 0.7},
                ],
                "confidence": 0.8,
                "bias_flags": [],
                "preview": "Manager dismissed my presentation in the all-hands",
            },
            {
                "uid": "sig-002",
                "address": "SA(work, manager, dismissal, yesterday)",
                "emotions": [{"emotion": "anger", "intensity": 8, "confidence": 0.9}],
                "confidence": 0.85,
                "bias_flags": ["rumination_amplification"],
                "preview": "Same thing happened again yesterday — idea shut down immediately",
            },
            {
                "uid": "sig-003",
                "address": "SA(home, partner, withdrawal, recently)",
                "emotions": [
                    {"emotion": "sadness", "intensity": 6, "confidence": 0.75},
                    {"emotion": "guilt", "intensity": 4, "confidence": 0.6},
                ],
                "confidence": 0.7,
                "bias_flags": ["projection"],
                "preview": "Partner went quiet after I vented about work. I know she's tired of hearing it",
            },
        ],
        "clusters": [
            {
                "cluster_id": "cls-001",
                "cluster_type": "same_person_diff_time",
                "shared_coordinates": {"person": ["manager"]},
                "divergent_dimensions": {"temporal": ["last_week", "yesterday"]},
                "strength": 0.82,
                "member_count": 4,
            },
        ],
        "insights": [
            {
                "uid": "ins-001",
                "detection_type": "recurring_pattern",
                "description": "Dismissal from authority figures consistently triggers shame followed by withdrawal",
                "confidence": 0.78,
            },
        ],
        "bias_flags": [
            {
                "type": "rumination_amplification",
                "count": 3,
                "example": "Same thing happened again yesterday — idea shut down immediately",
            },
            {
                "type": "projection",
                "count": 1,
                "example": "I know she's tired of hearing it",
                "source": "current_conversation",
            },
        ],
    },
}

# Template IDs the system supports
TEMPLATES = {
    None: "Default (no template)",
    "reflection": "Reflection template",
    "decision": "Decision template",
    "patterns": "Patterns template",
}

# ──────────────────────────────────────────────────────────────────────
# Scenario Matrix — 20 conversation examples
# ──────────────────────────────────────────────────────────────────────

def build_conversation_scenarios():
    """
    Build 20 conversation scenarios covering the key dimensions.

    Strategy: rather than full Cartesian product (3×4×3×5 = 180), we
    sample meaningfully across dimensions to cover the most important
    combinations within 20 scenarios.
    """
    scenarios = []

    # Group 1: All 3 personas × default template × emotional_disclosure × empty graph
    # Tests: how each persona responds to a new user's emotional disclosure
    for persona_id in PERSONAS:
        scenarios.append({
            "id": f"persona-{persona_id}-empty",
            "persona_id": persona_id,
            "template_id": None,
            "graph_state": "empty",
            "message_type": "emotional_disclosure",
            "description": f"{PERSONAS[persona_id].name} persona, no history, emotional disclosure",
        })

    # Group 2: Neutral observer × all 4 templates × emotional_disclosure × light history
    # Tests: how different templates shape the response
    for template_id, template_label in TEMPLATES.items():
        scenarios.append({
            "id": f"template-{template_id or 'default'}-light",
            "persona_id": "neutral_observer",
            "template_id": template_id,
            "graph_state": "light",
            "message_type": "emotional_disclosure",
            "description": f"Neutral observer, {template_label}, light history",
        })

    # Group 3: All 5 message types × neutral_observer × default template × rich history
    # Tests: how the system handles different message types with full context
    for msg_type in MESSAGES:
        scenarios.append({
            "id": f"msgtype-{msg_type}-rich",
            "persona_id": "neutral_observer",
            "template_id": None,
            "graph_state": "rich",
            "message_type": msg_type,
            "description": f"Neutral observer, rich history, {msg_type}",
        })

    # Group 4: Direct challenger × rich history × bias-laden message × each template
    # Tests: how the most aggressive persona handles bias with different templates
    for template_id, template_label in TEMPLATES.items():
        scenarios.append({
            "id": f"challenger-bias-{template_id or 'default'}",
            "persona_id": "direct_challenger",
            "template_id": template_id,
            "graph_state": "rich",
            "message_type": "bias_laden",
            "description": f"Direct challenger, {template_label}, rich history, bias-laden message",
        })

    # Group 5: Gentle explorer × light history × vague message
    # Tests: gentlest persona with the most ambiguous input
    scenarios.append({
        "id": "gentle-vague-light",
        "persona_id": "gentle_explorer",
        "template_id": None,
        "graph_state": "light",
        "message_type": "vague_wildcard",
        "description": "Gentle explorer, light history, vague message",
    })

    # Group 6: Gentle explorer × rich history × pattern-seeking
    # Tests: gentle persona with full context on a pattern question
    scenarios.append({
        "id": "gentle-pattern-rich",
        "persona_id": "gentle_explorer",
        "template_id": "patterns",
        "graph_state": "rich",
        "message_type": "pattern_seeking",
        "description": "Gentle explorer, patterns template, rich history, pattern-seeking",
    })

    # Group 7: Decision template × different personas × decision message
    # Tests: how the decision template interacts with different persona tones
    scenarios.append({
        "id": "challenger-decision-light",
        "persona_id": "direct_challenger",
        "template_id": "decision",
        "graph_state": "light",
        "message_type": "decision",
        "description": "Direct challenger, decision template, light history, decision message",
    })
    scenarios.append({
        "id": "gentle-decision-empty",
        "persona_id": "gentle_explorer",
        "template_id": "decision",
        "graph_state": "empty",
        "message_type": "decision",
        "description": "Gentle explorer, decision template, empty graph, decision message",
    })

    return scenarios[:20]  # Cap at 20


# ──────────────────────────────────────────────────────────────────────
# Signal Generation Scenarios — 10 examples
# ──────────────────────────────────────────────────────────────────────

def build_signal_scenarios():
    """Build 10 signal generation scenarios across message types."""
    scenarios = []

    for msg_type, message in MESSAGES.items():
        # Each message type: once without context, once with context
        scenarios.append({
            "id": f"signal-{msg_type}-no-context",
            "message": message,
            "conversation_context": None,
            "description": f"Signal gen: {msg_type}, no conversation context",
        })
        scenarios.append({
            "id": f"signal-{msg_type}-with-context",
            "message": message,
            "conversation_context": (
                "user: I've been feeling stressed about work lately.\n"
                "assistant: That sounds difficult. What's been weighing on you most?\n"
                f"user: {message[:80]}..."
            ),
            "description": f"Signal gen: {msg_type}, with conversation context",
        })

    return scenarios[:10]  # Cap at 10


# ──────────────────────────────────────────────────────────────────────
# Execution Engine
# ──────────────────────────────────────────────────────────────────────

# Import the live_conversation_service prompts directly
from events_api.live_conversation_service import (
    DEFAULT_BASE_PROMPT,
    THERAPEUTIC_TEMPLATES,
    FALLBACK_SYSTEM_PROMPT,
)


def build_mock_context_packet(graph_state_key: str) -> str:
    """Build a context packet from mock graph data, mimicking assemble_context()."""
    state = GRAPH_CONTEXTS[graph_state_key]

    return assemble_context(
        entities=state["entities"],
        signals=state.get("signals"),
        clusters=state.get("clusters"),
        insights=state.get("insights"),
        bias_flags=state.get("bias_flags"),
        token_budget=2000,
    )


def run_conversation_scenario(scenario: dict) -> dict:
    """Run a single conversation scenario and capture the full prompt→output tuple."""
    persona = get_persona(scenario["persona_id"])
    message = MESSAGES[scenario["message_type"]]
    template_id = scenario["template_id"]
    graph_state = scenario["graph_state"]

    # Build the exact system prompt the pipeline would construct
    base_prompt = THERAPEUTIC_TEMPLATES.get(template_id or "", DEFAULT_BASE_PROMPT)
    context_packet = build_mock_context_packet(graph_state)

    system_prompt = build_system_prompt(
        base_prompt=base_prompt,
        persona_modifier=persona.system_prompt_modifier,
        context_packet=context_packet,
    )

    # Build messages array (just the user message for baseline)
    messages = [{"role": "user", "content": message}]

    # Call the LLM
    start = time.time()
    try:
        response = generate_conversation_reply(
            system_prompt=system_prompt,
            messages=messages,
            model=DEFAULT_MODEL,
            max_tokens=800,
        )
        error = None
    except Exception as e:
        response = str(e)
        error = type(e).__name__

    latency_ms = round((time.time() - start) * 1000)

    return {
        "scenario_id": scenario["id"],
        "description": scenario["description"],
        "persona": scenario["persona_id"],
        "template": scenario["template_id"],
        "graph_state": graph_state,
        "message_type": scenario["message_type"],
        "user_message": message,
        "system_prompt": system_prompt,
        "system_prompt_length": len(system_prompt),
        "context_packet": context_packet,
        "context_packet_length": len(context_packet),
        "ai_response": response,
        "ai_response_length": len(response) if response else 0,
        "latency_ms": latency_ms,
        "error": error,
        "model": DEFAULT_MODEL,
    }


def run_signal_scenario(scenario: dict) -> dict:
    """Run a single signal generation scenario."""
    start = time.time()
    try:
        result = generate_signal_from_message(
            message=scenario["message"],
            conversation_context=scenario.get("conversation_context"),
            participants=[],
            active_categories=None,
            persona_modifier=None,
        )
        error = None
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        result = {"signals": [], "error": str(e), "traceback": tb}
        error = f"{type(e).__name__}: {e}"
        logger.warning("  Traceback: %s", tb.strip().split('\n')[-3:])

    latency_ms = round((time.time() - start) * 1000)

    signals = result.get("signals", [])
    return {
        "scenario_id": scenario["id"],
        "description": scenario["description"],
        "message": scenario["message"],
        "conversation_context": scenario.get("conversation_context"),
        "signals": signals,
        "signal_count": len(signals),
        "addresses": [s.get("signal_address", "") for s in signals],
        "emotions": [
            e.get("emotion", "")
            for s in signals
            for e in s.get("emotions", [])
        ],
        "bias_flags": [
            f for s in signals
            for f in s.get("observation_bias_flags", [])
        ],
        "latency_ms": latency_ms,
        "error": error,
        "model": DEFAULT_MODEL,
    }


# ──────────────────────────────────────────────────────────────────────
# Report Generator
# ──────────────────────────────────────────────────────────────────────

def generate_markdown_report(
    conversation_results: list,
    signal_results: list,
    output_path: str,
):
    """Generate a human-readable markdown report."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    total_conv = len(conversation_results)
    total_sig = len(signal_results)
    conv_errors = sum(1 for r in conversation_results if r["error"])
    sig_errors = sum(1 for r in signal_results if r["error"])

    conv_latencies = [r["latency_ms"] for r in conversation_results if not r["error"]]
    sig_latencies = [r["latency_ms"] for r in signal_results if not r["error"]]

    avg_conv_lat = round(sum(conv_latencies) / len(conv_latencies)) if conv_latencies else 0
    avg_sig_lat = round(sum(sig_latencies) / len(sig_latencies)) if sig_latencies else 0

    lines = [
        f"# ThriveSight LLM Quality Baseline Report",
        f"",
        f"Generated: {now}",
        f"Model: {DEFAULT_MODEL}",
        f"",
        f"## Summary",
        f"",
        f"| Metric | Conversation | Signal Generation |",
        f"|--------|-------------|-------------------|",
        f"| Total scenarios | {total_conv} | {total_sig} |",
        f"| Errors | {conv_errors} | {sig_errors} |",
        f"| Avg latency (ms) | {avg_conv_lat} | {avg_sig_lat} |",
        f"| Min latency (ms) | {min(conv_latencies) if conv_latencies else 'N/A'} | {min(sig_latencies) if sig_latencies else 'N/A'} |",
        f"| Max latency (ms) | {max(conv_latencies) if conv_latencies else 'N/A'} | {max(sig_latencies) if sig_latencies else 'N/A'} |",
        f"",
        f"---",
        f"",
        f"## Conversation Results",
        f"",
    ]

    for r in conversation_results:
        status = "ERROR" if r["error"] else "OK"
        lines.extend([
            f"### {r['scenario_id']} [{status}]",
            f"",
            f"**{r['description']}**",
            f"",
            f"- Persona: `{r['persona']}` | Template: `{r['template'] or 'default'}` | Graph: `{r['graph_state']}`",
            f"- System prompt: {r['system_prompt_length']} chars | Context packet: {r['context_packet_length']} chars",
            f"- Latency: {r['latency_ms']}ms",
            f"",
            f"**User message:**",
            f"> {r['user_message']}",
            f"",
            f"**AI response:**",
            f"> {r['ai_response']}",
            f"",
        ])
        if r["error"]:
            lines.append(f"**Error:** `{r['error']}`\n")
        lines.append("---\n")

    lines.extend([
        f"## Signal Generation Results",
        f"",
    ])

    for r in signal_results:
        status = "ERROR" if r["error"] else "OK"
        lines.extend([
            f"### {r['scenario_id']} [{status}]",
            f"",
            f"**{r['description']}**",
            f"",
            f"- Signals generated: {r['signal_count']}",
            f"- Addresses: {', '.join(r['addresses']) or 'none'}",
            f"- Emotions: {', '.join(r['emotions']) or 'none'}",
            f"- Bias flags: {', '.join(r['bias_flags']) or 'none'}",
            f"- Latency: {r['latency_ms']}ms",
            f"",
            f"**Message:**",
            f"> {r['message']}",
            f"",
        ])

        if r.get("conversation_context"):
            lines.extend([
                f"**Conversation context:**",
                f"```",
                r["conversation_context"],
                f"```",
                f"",
            ])

        if r["signals"]:
            lines.append("**Signals:**")
            lines.append("```json")
            lines.append(json.dumps(r["signals"], indent=2)[:2000])
            lines.append("```")
            lines.append("")

        if r["error"]:
            lines.append(f"**Error:** `{r['error']}`\n")
        lines.append("---\n")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    output_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(output_dir, "eval_results.json")
    md_path = os.path.join(output_dir, "eval_report.md")

    # Verify API key
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key or api_key.startswith("sk-placeholder"):
        logger.error("ANTHROPIC_API_KEY not set or is a placeholder. Set it before running.")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("ThriveSight LLM Quality Baseline")
    logger.info("Model: %s", DEFAULT_MODEL)
    logger.info("=" * 60)

    # Build scenarios
    conv_scenarios = build_conversation_scenarios()
    sig_scenarios = build_signal_scenarios()

    logger.info("Conversation scenarios: %d", len(conv_scenarios))
    logger.info("Signal scenarios: %d", len(sig_scenarios))

    # Run conversation scenarios
    logger.info("\n--- Running Conversation Scenarios ---")
    conv_results = []
    for i, scenario in enumerate(conv_scenarios, 1):
        logger.info("[%d/%d] %s", i, len(conv_scenarios), scenario["description"])
        result = run_conversation_scenario(scenario)
        conv_results.append(result)
        if result["error"]:
            logger.warning("  ERROR: %s", result["error"])
        else:
            logger.info("  OK — %dms, %d chars", result["latency_ms"], result["ai_response_length"])

    # Run signal scenarios
    logger.info("\n--- Running Signal Generation Scenarios ---")
    sig_results = []
    for i, scenario in enumerate(sig_scenarios, 1):
        logger.info("[%d/%d] %s", i, len(sig_scenarios), scenario["description"])
        result = run_signal_scenario(scenario)
        sig_results.append(result)
        if result["error"]:
            logger.warning("  ERROR: %s", result["error"])
        else:
            logger.info("  OK — %dms, %d signals", result["latency_ms"], result["signal_count"])

    # Write JSON results
    full_results = {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model": DEFAULT_MODEL,
            "conversation_count": len(conv_results),
            "signal_count": len(sig_results),
        },
        "conversation_results": conv_results,
        "signal_results": sig_results,
    }

    with open(json_path, "w") as f:
        json.dump(full_results, f, indent=2, default=str)
    logger.info("\nJSON results written to: %s", json_path)

    # Write markdown report
    generate_markdown_report(conv_results, sig_results, md_path)
    logger.info("Markdown report written to: %s", md_path)

    # Summary
    conv_errors = sum(1 for r in conv_results if r["error"])
    sig_errors = sum(1 for r in sig_results if r["error"])
    logger.info("\n=== SUMMARY ===")
    logger.info("Conversations: %d/%d succeeded", len(conv_results) - conv_errors, len(conv_results))
    logger.info("Signals: %d/%d succeeded", len(sig_results) - sig_errors, len(sig_results))


if __name__ == "__main__":
    main()
