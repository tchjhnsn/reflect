"""
ThriveSight Reflection Engine — AI-guided reflection conversation informed by signal data.

Layer 2 of the ThriveSight build plan. After the analysis pipeline produces signals,
patterns, trajectory, and reframe data, this module generates thoughtful reflection
questions that help the user explore their conversation dynamics before seeing the
full analysis.

Question types escalate with depth:
  - Anchoring (exchanges 1-2): Explore a specific moment or emotional shift
  - Pattern Recognition (exchanges 3-4): Name the recurring dynamic
  - Intervention (exchanges 5+): Where could the cycle break?
"""

import logging
from typing import Any

from . import llm_client

logger = logging.getLogger(__name__)


# ── Question Type Constants ──────────────────────────────────────────

QUESTION_TYPES = ("anchoring", "pattern_recognition", "intervention")


def _question_type_for_exchange(exchange_count: int) -> str:
    """Map exchange count to question type."""
    if exchange_count < 2:
        return "anchoring"
    elif exchange_count < 4:
        return "pattern_recognition"
    return "intervention"


# ── Analysis Summary Builder ─────────────────────────────────────────

def _build_analysis_summary(analysis_result: dict) -> str:
    """
    Build a rich text summary of analysis results for the LLM prompt.

    Includes turn excerpts, signal details, pattern descriptions,
    inflection points, and the reframe narrative — everything the LLM
    needs to ask data-informed questions.
    """
    parts = []

    # Conversation turns with text excerpts
    conversation = analysis_result.get("conversation", {})
    turns = conversation.get("turns", [])
    if turns:
        parts.append("CONVERSATION TURNS:")
        for t in turns[:30]:  # Cap to prevent token overflow
            text_excerpt = t.get("text", "")[:120]
            parts.append(
                f"  Turn {t.get('turn_number', '?')} ({t.get('speaker', '?')}): "
                f"\"{text_excerpt}\""
            )

    # Signals with emotions and triggers
    signals = analysis_result.get("signals", [])
    if signals:
        parts.append("\nSIGNALS:")
        for s in signals[:25]:
            trigger = s.get("trigger_action", {})
            parts.append(
                f"  Turn {s.get('turn_number', '?')} ({s.get('speaker', '?')}): "
                f"emotion={s.get('emotion', '?')}, intensity={s.get('intensity', '?')}, "
                f"reaction={s.get('reaction', '?')}, "
                f"trigger_category={trigger.get('category', '?')}"
            )

    # Patterns
    patterns = analysis_result.get("patterns", [])
    if patterns:
        parts.append("\nPATTERNS:")
        for p in patterns:
            parts.append(
                f"  {p.get('pattern_name', '?')} (score={p.get('score', '?')}, "
                f"occurrences={p.get('occurrence_count', '?')}): "
                f"{p.get('hypothesis', '')}"
            )

    # Trajectory inflection points
    trajectory = analysis_result.get("trajectory", {})
    speakers_data = trajectory.get("speakers", trajectory)  # Handle both shapes
    if isinstance(speakers_data, dict):
        inflection_parts = []
        for speaker, data in speakers_data.items():
            if not isinstance(data, dict):
                continue
            for ip in data.get("inflection_points", []):
                inflection_parts.append(
                    f"  {speaker} at Turn {ip.get('turn_number', '?')}: "
                    f"intensity shifted by {ip.get('intensity_delta', '?')} "
                    f"({ip.get('direction', '?')}), "
                    f"cause: \"{ip.get('cause', '?')}\""
                )
        if inflection_parts:
            parts.append("\nINFLECTION POINTS (major emotional shifts):")
            parts.extend(inflection_parts)

    # Reframe
    reframe = analysis_result.get("reframe", {})
    if reframe and isinstance(reframe, dict):
        reframe_text = reframe.get("text", "")
        if reframe_text:
            parts.append(f"\nREFRAME NARRATIVE:\n  {reframe_text[:400]}")

    return "\n".join(parts)


# ── Target Moment Selection ──────────────────────────────────────────

def _collect_inflection_points(trajectory: dict) -> list[dict]:
    """Extract all inflection points from trajectory, sorted by intensity delta (largest first)."""
    all_inflections = []
    speakers_data = trajectory.get("speakers", trajectory)
    if not isinstance(speakers_data, dict):
        return []

    for speaker, data in speakers_data.items():
        if not isinstance(data, dict):
            continue
        for ip in data.get("inflection_points", []):
            all_inflections.append({
                "speaker": speaker,
                "turn_number": ip.get("turn_number"),
                "intensity_delta": abs(ip.get("intensity_delta", 0)),
                "direction": ip.get("direction", "unknown"),
                "cause": ip.get("cause", ""),
            })

    all_inflections.sort(key=lambda x: x["intensity_delta"], reverse=True)
    return all_inflections


def _select_target_moment(
    analysis_result: dict,
    explored_turns: set[int] | None = None,
) -> dict | None:
    """
    Pick the most relevant unexplored inflection point or pattern moment.

    Returns a dict with speaker, turn_number, direction, cause, intensity_delta
    or None if nothing suitable found.
    """
    explored = explored_turns or set()

    # First try inflection points (most emotionally significant moments)
    trajectory = analysis_result.get("trajectory", {})
    inflections = _collect_inflection_points(trajectory)
    for ip in inflections:
        if ip["turn_number"] not in explored:
            return ip

    # Fallback: pick the highest-intensity signal not yet explored
    signals = analysis_result.get("signals", [])
    sorted_signals = sorted(signals, key=lambda s: s.get("intensity", 0), reverse=True)
    for s in sorted_signals:
        if s.get("turn_number") not in explored:
            return {
                "speaker": s.get("speaker", "someone"),
                "turn_number": s.get("turn_number"),
                "intensity_delta": s.get("intensity", 0),
                "direction": "high_intensity",
                "cause": s.get("trigger_action", {}).get("action_text", ""),
            }

    return None


# ── Core Reflection Engine ───────────────────────────────────────────

class ReflectionEngine:
    """
    Generates AI reflection questions informed by conversation analysis data.

    Usage:
        engine = ReflectionEngine()
        opening = engine.generate_opening_question(analysis_result)
        followup = engine.generate_followup(user_response, analysis_result, history, count)
    """

    def generate_opening_question(self, analysis_result: dict) -> dict:
        """
        Generate the first reflection question targeting the biggest emotional moment.

        Args:
            analysis_result: Full pipeline output (conversation, signals, patterns, trajectory, reframe)

        Returns:
            {"question": str, "reasoning": str, "target_turn": int | None}
        """
        target = _select_target_moment(analysis_result)
        analysis_summary = _build_analysis_summary(analysis_result)

        target_context = ""
        target_turn = None
        if target:
            target_turn = target["turn_number"]
            target_context = (
                f"\nFocus on this moment: {target['speaker']} at Turn {target['turn_number']} "
                f"experienced a {target['direction']} (intensity shift: {target['intensity_delta']:.1f}). "
                f"The cause was: \"{target['cause']}\""
            )

        user_prompt = (
            f"Here is the full analysis of a conversation:\n\n"
            f"{analysis_summary}\n\n"
            f"{target_context}\n\n"
            f"Generate your opening reflection question. This is the very first question — "
            f"it should be warm, inviting, and focused on a specific moment from the conversation. "
            f"Ask about the person's internal experience at that moment."
        )

        try:
            result = llm_client.analyze_with_retry(
                llm_client.REFLECTION_SYSTEM,
                user_prompt,
            )
            return {
                "question": result.get("question", self._fallback_opening(target)),
                "reasoning": result.get("reasoning", ""),
                "target_turn": target_turn,
            }
        except Exception as e:
            logger.error(f"Opening question generation failed: {e}")
            return {
                "question": self._fallback_opening(target),
                "reasoning": "fallback — LLM call failed",
                "target_turn": target_turn,
            }

    def generate_followup(
        self,
        user_response: str,
        analysis_result: dict,
        reflection_history: list[dict[str, str]],
        exchange_count: int,
    ) -> dict:
        """
        Generate a follow-up question based on user response + signal data.

        Args:
            user_response: What the user just said
            analysis_result: Full pipeline output
            reflection_history: List of {"role": "ai"|"user", "text": str}
            exchange_count: How many exchanges have occurred

        Returns:
            {"question": str, "question_type": str, "should_suggest_reveal": bool}
        """
        question_type = _question_type_for_exchange(exchange_count)
        analysis_summary = _build_analysis_summary(analysis_result)

        # Build history context for the LLM
        history_text = ""
        if reflection_history:
            history_lines = []
            for entry in reflection_history[-8:]:  # Last 8 messages to keep context manageable
                role_label = "Reflection Guide" if entry["role"] == "ai" else "User"
                history_lines.append(f"{role_label}: {entry['text']}")
            history_text = "\n".join(history_lines)

        # Find explored turns so we target something new
        explored_turns = set()
        for entry in reflection_history:
            if entry["role"] == "ai":
                # Extract any turn numbers mentioned in AI questions
                import re
                turn_refs = re.findall(r"[Tt]urn\s+(\d+)", entry["text"])
                explored_turns.update(int(t) for t in turn_refs)

        next_target = _select_target_moment(analysis_result, explored_turns)

        target_hint = ""
        if next_target:
            target_hint = (
                f"\nSuggested next focus: {next_target['speaker']} at Turn {next_target['turn_number']} "
                f"({next_target['direction']}, intensity shift: {next_target['intensity_delta']:.1f}). "
                f"But adapt based on what the user just shared."
            )

        type_guidance = {
            "anchoring": (
                "Ask an ANCHORING question — explore a specific moment or emotional shift. "
                "Help the person describe what they were experiencing internally."
            ),
            "pattern_recognition": (
                "Ask a PATTERN RECOGNITION question — help the person see the recurring dynamic. "
                "Reference one of the detected patterns and ask if they recognize it."
            ),
            "intervention": (
                "Ask an INTERVENTION question — explore where the cycle could break. "
                "Ask what would need to change for a different outcome. "
                "Reference the reframe narrative if relevant."
            ),
        }

        user_prompt = (
            f"Analysis data:\n{analysis_summary}\n\n"
            f"Reflection conversation so far:\n{history_text}\n\n"
            f"The user just responded:\n\"{user_response}\"\n\n"
            f"Question type: {question_type}\n"
            f"{type_guidance.get(question_type, '')}\n"
            f"{target_hint}\n\n"
            f"Generate your next reflection question. Build on what the user shared "
            f"while connecting it to the signal data."
        )

        should_suggest_reveal = exchange_count >= 4

        try:
            result = llm_client.analyze_with_retry(
                llm_client.REFLECTION_SYSTEM,
                user_prompt,
            )
            return {
                "question": result.get("question", self._fallback_followup(question_type)),
                "question_type": question_type,
                "should_suggest_reveal": should_suggest_reveal,
            }
        except Exception as e:
            logger.error(f"Followup question generation failed: {e}")
            return {
                "question": self._fallback_followup(question_type),
                "question_type": question_type,
                "should_suggest_reveal": should_suggest_reveal,
            }

    # ── Deterministic Fallbacks ──────────────────────────────────────

    def _fallback_opening(self, target: dict | None) -> str:
        if target:
            return (
                f"I noticed an interesting moment around Turn {target['turn_number']}. "
                f"There was a shift in the emotional tone there. "
                f"What was going through your mind at that point in the conversation?"
            )
        return (
            "As you think back on this conversation, what moment stands out to you "
            "as the most emotionally significant? What was that like for you?"
        )

    def _fallback_followup(self, question_type: str) -> str:
        fallbacks = {
            "anchoring": (
                "Can you tell me more about what you were feeling in that moment? "
                "What was happening internally for you?"
            ),
            "pattern_recognition": (
                "Does this dynamic feel familiar? Is this something that tends to "
                "happen in conversations like this one?"
            ),
            "intervention": (
                "If you could go back to that moment and respond differently, "
                "what would you want to change? What would that look like?"
            ),
        }
        return fallbacks.get(question_type, fallbacks["anchoring"])
