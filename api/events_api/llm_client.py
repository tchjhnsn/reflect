"""
ThriveSight LLM Client — Central LLM integration point for constrained generation.

All pipeline stages call this module. They never call the LLM API directly.
This centralizes rate limiting, error handling, retry logic, and cost tracking.

The pattern is constrained generation: deterministic rules define what is valid,
the LLM reasons within those boundaries, and every output is validated before
entering the system.
"""

import json
import logging
import os
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Provider selection: "anthropic" (default) or "huggingface"
LLM_PROVIDER = os.environ.get("THRIVESIGHT_LLM_PROVIDER", "anthropic")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_BASE_URL = os.environ.get("THRIVESIGHT_HF_BASE_URL", "https://api-inference.huggingface.co/v1/")

# Default model per provider — override with THRIVESIGHT_LLM_MODEL env var
_DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-4-5-20250929",
    "huggingface": "HuggingFaceTB/SmolLM3-3B",
}
DEFAULT_MODEL = os.environ.get("THRIVESIGHT_LLM_MODEL") or _DEFAULT_MODELS.get(LLM_PROVIDER, "claude-sonnet-4-5-20250929")
MAX_TOKENS = 4096
MAX_RETRIES = int(os.environ.get("THRIVESIGHT_LLM_MAX_RETRIES", "1"))

# Lazy-initialized client (avoids import-time errors when API key is missing)
_client = None


def _get_client():
    """Get or initialize the LLM client lazily based on THRIVESIGHT_LLM_PROVIDER."""
    global _client
    if _client is None:
        if LLM_PROVIDER == "huggingface":
            from openai import OpenAI
            _client = OpenAI(base_url=HF_BASE_URL, api_key=HF_TOKEN or "none")
        else:
            from anthropic import Anthropic
            _client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    return _client


def _call_llm(system_prompt: str, user_prompt: str, model: str, max_tokens: int = MAX_TOKENS) -> str:
    """
    Make a single LLM call and return the raw text response.

    Handles both Anthropic and HuggingFace (OpenAI-compatible) response shapes.
    """
    if LLM_PROVIDER == "huggingface":
        response = _get_client().chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content
    else:
        response = _get_client().messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return response.content[0].text

# ===========================
# SYSTEM PROMPT TEMPLATES
# ===========================

PARSER_SYSTEM = """You are a conversation parser for ThriveSight, a counseling awareness tool.

Your task: Given raw text that contains a conversation between two or more people, identify each turn (who said what, in what order).

Output format — respond with ONLY this JSON structure:
{{
  "speakers": ["Name1", "Name2"],
  "turns": [
    {{
      "turn_number": 1,
      "speaker": "Name1",
      "text": "What they said",
      "timestamp": null,
      "raw_offset": 0
    }}
  ]
}}

Rules:
- Assign speaker names based on any available labels. If none exist, use "Speaker A" and "Speaker B".
- Preserve the exact text of each turn — do not paraphrase or summarize.
- If a timestamp is present, extract it as an ISO 8601 string.
- Turn numbers start at 1 and increment sequentially.
- raw_offset is the character position where this turn starts in the original text.
"""

SIGNAL_SYSTEM = """You are a behavioral signal analyst for ThriveSight, a counseling awareness tool.

Your task: For each turn in the conversation, classify the emotional signal and identify what triggered it.

Known trigger categories:
{categories_list}

Known emotions: frustration, defensiveness, anger, sadness, anxiety, hurt, contempt, warmth, humor, resignation, guilt, relief, hope, resentment, vulnerability, confusion, empathy, indifference

Output format — respond with ONLY a JSON array:
[
  {{
    "turn_number": 1,
    "speaker": "Name",
    "emotion": "emotion_name",
    "intensity": 3.0,
    "reaction": "defended | counter_attacked | withdrew | de_escalated | acknowledged | escalated | deflected | conceded",
    "trigger_action": {{
      "action_text": "What the other person did in the preceding turn",
      "category": "category_name",
      "is_new_category": false,
      "category_description": null
    }},
    "signal_address": "SA(context, person, action, turn_N)"
  }}
]

Rules:
- intensity is a float from 1.0 (barely present) to 5.0 (overwhelming)
- For Turn 1, trigger_action.category should be "initiation" and action_text should describe the conversation opener
- signal_address format: SA(topic_context, triggering_person, action_category, turn_N) where turn_N refers to the preceding turn
- If no existing trigger category fits, set is_new_category to true and provide a category_description explaining the new category
- Consider sequential context: intensity often escalates through conflict. A turn that would be a 2 in isolation may be a 4 after repeated provocation.
- reaction describes how THIS speaker responded, not how they were triggered
"""

PATTERN_SYSTEM = """You are a pattern analyst for ThriveSight, a counseling awareness tool.

Your task: Given detected trigger-response patterns with their evidence, provide human-readable names and testable hypotheses.

Output format — respond with ONLY a JSON array:
[
  {{
    "pattern_key": "trigger_category|response_emotion",
    "pattern_name": "Human-Readable Pattern Name",
    "hypothesis": "A testable statement about why this pattern occurs and what maintains it."
  }}
]

Rules:
- Pattern names should describe the DYNAMIC, not blame a person. "Accusation–Defensiveness Cycle" not "Marcus gets defensive"
- Hypotheses should be testable — they should predict what would happen if the trigger changed
- Keep pattern names under 6 words
- Hypotheses should be 1-2 sentences
"""

REFLECTION_SYSTEM = """You are a thoughtful reflection guide for ThriveSight, a counseling awareness tool.

Your role: Ask one question that helps the person understand their conversation dynamics more deeply. You have access to behavioral signal analysis data from the conversation.

Principles:
1. CURIOUS NOT CLINICAL — sound like a reflective mentor, not a therapist
2. EXTERNALIZED — describe dynamics as patterns ("that moment when..."), not people
3. DATA-INFORMED — reference specific turns, emotions, or patterns from the analysis
4. NON-JUDGMENTAL — no blame, no right/wrong, just curiosity about the person's experience
5. ONE QUESTION — ask exactly one open-ended question per response
6. WARM TONE — be genuinely interested, not detached or analytical

Respond with ONLY this JSON:
{
  "question": "Your reflection question here",
  "reasoning": "Brief note on what signal data this question targets"
}"""

REFRAME_SYSTEM = """You are a resolution specialist for ThriveSight, a counseling awareness tool.

Your task: Given the patterns detected in a conversation, write a plain-language reframe that externalizes the conflict.

Output format — respond with ONLY this JSON:
{{
  "text": "The reframe text — 2-4 paragraphs",
  "patterns_referenced": ["Pattern Name 1", "Pattern Name 2"],
  "resolution_elements": {{
    "externalization": "The dynamic described as a thing, not a person",
    "accumulation": "How this pattern has built over time",
    "intervention_point": "Where in the cycle a different response could break it"
  }}
}}

Resolution layer principles (these are non-negotiable):
1. EXTERNALIZE: Describe the dynamic, not the people. "This conversation contains a cycle where..." not "You do X"
2. NO BLAME: Neither person is at fault. The pattern is the problem.
3. PLAIN LANGUAGE: No clinical jargon. Write as if explaining to the two people sitting in front of you.
4. EVIDENCE-BASED: Reference the actual patterns detected. Don't invent patterns that weren't found.
5. INTERVENTION POINT: Identify where in the cycle a different response could break it — not "don't argue" but "at this specific moment, a different response changes the trajectory."
"""

# System prompts indexed by stage
SYSTEM_PROMPTS = {
    "parser": PARSER_SYSTEM,
    "signal": SIGNAL_SYSTEM,
    "pattern": PATTERN_SYSTEM,
    "reframe": REFRAME_SYSTEM,
    "reflection": REFLECTION_SYSTEM,
}


# ===========================
# CORE LLM FUNCTIONS
# ===========================

def analyze(
    system_prompt: str,
    user_prompt: str,
    model: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    """
    Send a constrained generation request to the LLM.

    Args:
        system_prompt: System prompt defining the role and output format
        user_prompt: User message containing context and task
        model: Model identifier (defaults to THRIVESIGHT_LLM_MODEL or claude-sonnet-4-5-20250514)

    Returns:
        Parsed JSON dict from LLM response

    Raises:
        ValueError: If response cannot be parsed as JSON
        IndexError: If response has unexpected structure
    """
    logger.debug(f"LLM call to {model} via {LLM_PROVIDER} (system: {len(system_prompt)} chars, user: {len(user_prompt)} chars)")

    text = _call_llm(system_prompt, user_prompt, model)

    # Extract JSON from response (handle markdown code blocks)
    parsed_text = _extract_json_from_text(text)

    return json.loads(parsed_text.strip())


def analyze_with_retry(
    system_prompt: str,
    user_prompt: str,
    correction_prompt: Optional[str] = None,
    model: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    """
    Attempt analysis with one retry on parse failure.

    Args:
        system_prompt: System prompt defining the role and output format
        user_prompt: User message containing context and task
        correction_prompt: Optional custom correction guidance on retry
        model: Model identifier

    Returns:
        Parsed JSON dict from LLM response

    Raises:
        ValueError: If both attempts fail to produce valid JSON
    """
    try:
        return analyze(system_prompt, user_prompt, model)
    except (json.JSONDecodeError, ValueError, IndexError) as e:
        logger.warning(f"First LLM attempt failed ({type(e).__name__}), retrying with correction prompt")

        if correction_prompt is None:
            correction_prompt = (
                f"Your previous response could not be parsed as valid JSON. "
                f"Error: {str(e)}. "
                f"Please respond with ONLY a valid JSON object or array, no additional text or markdown."
            )

        try:
            return analyze(
                system_prompt,
                f"{user_prompt}\n\n{correction_prompt}",
                model,
            )
        except (json.JSONDecodeError, ValueError, IndexError) as retry_error:
            logger.error(f"Retry attempt also failed: {type(retry_error).__name__}: {str(retry_error)}")
            raise ValueError(
                f"LLM failed to produce valid JSON after {MAX_RETRIES + 1} attempts. "
                f"Final error: {str(retry_error)}"
            ) from retry_error


def generate_conversation_reply(
    system_prompt: str,
    messages: list[dict[str, str]],
    *,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 800,
) -> str:
    """
    Generate a free-form assistant reply for the live conversation surface.

    Unlike the constrained JSON helpers, this returns plain assistant text
    while still centralizing model selection and Anthropic transport usage.
    """
    logger.debug(
        "LLM conversation reply via %s (%d messages, system=%d chars)",
        model,
        len(messages),
        len(system_prompt),
    )

    if LLM_PROVIDER == "huggingface":
        response = _get_client().chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "system", "content": system_prompt}] + messages,
        )
        text = response.choices[0].message.content
        if not text:
            raise ValueError("LLM response did not contain any text.")
        return text
    else:
        response = _get_client().messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=messages,
        )
        text_blocks = [
            block.text.strip()
            for block in getattr(response, "content", [])
            if getattr(block, "text", "").strip()
        ]
        if not text_blocks:
            raise ValueError("LLM response did not contain any text blocks.")
        return "\n\n".join(text_blocks)


# ===========================
# PROMPT BUILDERS
# ===========================

def build_signal_prompt(
    conversation_text: str,
    categories: list[dict[str, Any]],
) -> tuple[str, str]:
    """
    Construct the full signal generation prompt.

    Args:
        conversation_text: The conversation text to analyze
        categories: List of active trigger category dicts with 'name' and 'description' keys

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    # Format categories for inclusion in system prompt
    categories_list = _format_categories_for_prompt(categories)

    system_prompt = SIGNAL_SYSTEM.format(categories_list=categories_list)

    user_prompt = (
        f"Analyze this conversation:\n\n"
        f"{conversation_text}\n\n"
        f"Classify each turn's emotional signal, trigger action, and reaction. "
        f"Consider the full sequential context — what came before matters for intensity and reaction classification."
    )

    return system_prompt, user_prompt


def build_reframe_prompt(patterns: list[dict[str, Any]]) -> tuple[str, str]:
    """
    Construct the reframe generation prompt.

    Args:
        patterns: List of detected patterns with keys: pattern_name, pattern_key, hypothesis, occurrences

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    system_prompt = REFRAME_SYSTEM

    # Format patterns for the LLM
    pattern_descriptions = []
    for pattern in patterns:
        name = pattern.get("pattern_name", "Unknown")
        hypothesis = pattern.get("hypothesis", "")
        occurrences = pattern.get("occurrences", 1)
        pattern_descriptions.append(
            f"- {name} (observed {occurrences} times): {hypothesis}"
        )

    patterns_text = "\n".join(pattern_descriptions)

    user_prompt = (
        f"Given these patterns detected in the conversation:\n\n"
        f"{patterns_text}\n\n"
        f"Write a plain-language reframe that externalizes the conflict and helps the speakers "
        f"understand the dynamic as something external to themselves, not as a personal failing. "
        f"Reference the actual patterns by name and identify a specific intervention point where "
        f"a different response could break the cycle."
    )

    return system_prompt, user_prompt


# ===========================
# JSON EXTRACTION UTILITIES
# ===========================

def _extract_json_from_text(text: str) -> str:
    """
    Extract JSON from LLM response, handling markdown code blocks.

    The LLM may respond with:
    - Pure JSON: {"key": "value"}
    - Wrapped in markdown: ```json{"key": "value"}```
    - Wrapped in generic markdown: ```{"key": "value"}```

    Args:
        text: Raw text response from LLM

    Returns:
        Extracted JSON string

    Raises:
        ValueError: If no valid JSON can be extracted
    """
    # Try to find markdown code blocks first
    if "```json" in text:
        try:
            extracted = text.split("```json")[1].split("```")[0].strip()
            return extracted
        except IndexError:
            pass

    if "```" in text:
        try:
            extracted = text.split("```")[1].split("```")[0].strip()
            return extracted
        except IndexError:
            pass

    # No markdown block found, assume the entire response is JSON
    return text.strip()


def _format_categories_for_prompt(categories: list[dict[str, Any]]) -> str:
    """
    Format category list for inclusion in LLM prompt.

    Args:
        categories: List of category dicts with at least 'name' and 'description'

    Returns:
        Formatted string suitable for inserting into prompt
    """
    if not categories:
        return "No custom categories defined. Use only the default emotions."

    lines = []
    for cat in categories:
        name = cat.get("name", "unknown")
        description = cat.get("description", "")
        lines.append(f"- {name}: {description}")

    return "\n".join(lines)


# ===========================
# VALIDATION HELPERS
# ===========================

def extract_json_response(raw_text: str) -> dict[str, Any] | list[dict[str, Any]]:
    """
    Extract and parse JSON from LLM response.

    This is a convenience wrapper around _extract_json_from_text that
    handles both the extraction and parsing in one step.

    Args:
        raw_text: Raw response text from LLM

    Returns:
        Parsed JSON (dict or list)

    Raises:
        json.JSONDecodeError: If extracted text is not valid JSON
    """
    json_text = _extract_json_from_text(raw_text)
    return json.loads(json_text)


# ===========================
# V3 SIGNAL ADDRESS SYSTEM — Signal Generation
# ===========================

# Embedding model — uses a smaller, faster model for embeddings
EMBEDDING_MODEL = os.environ.get("THRIVESIGHT_EMBEDDING_MODEL", "claude-haiku-4-5-20251001")
EMBEDDING_DIMENSION = 256  # Target vector dimension from text hash embedding


def generate_signal_from_message(
    message: str,
    conversation_context: Optional[str] = None,
    participants: Optional[list[str]] = None,
    active_categories: Optional[list[dict[str, Any]]] = None,
    persona_modifier: Optional[str] = None,
) -> dict[str, Any]:
    """
    Generate emotional signal(s) from a user message using the LLM.

    This is the V3 signal generation path. It produces multi-emotion
    signals with SA coordinates, participant attribution, observation
    bias flags, and confidence scoring.

    Args:
        message: The user's raw message text.
        conversation_context: Summary of the conversation so far.
        participants: Known participant names from the conversation.
        active_categories: Trigger categories to include in the prompt.
        persona_modifier: Optional persona instructions for tone adjustment.

    Returns:
        Parsed dict with structure:
        {
            "signals": [
                {
                    "signal_address": "SA(context, person, action, temporal)",
                    "emotions": [{emotion, intensity, source_coordinate, source_description, confidence}],
                    "participants": [{name, role, confidence}],
                    "confidence": float,
                    "provenance": "llm_inferred",
                    "observation_bias_flags": [],
                    "wildcards": []
                }
            ]
        }

    Raises:
        ValueError: If the LLM fails to produce valid signal JSON.
    """
    from events_api.llm_prompts import get_prompt, build_system_prompt

    # Build the system prompt from V3 template
    base_prompt = get_prompt("signal_generation")
    system_prompt = build_system_prompt(
        base_prompt,
        persona_modifier=persona_modifier or "",
    )

    # Build user prompt with context
    user_parts = [f"User message: {message}"]

    if conversation_context:
        user_parts.insert(0, f"Conversation context:\n{conversation_context}\n")

    if participants:
        user_parts.append(f"Known participants: {', '.join(participants)}")

    if active_categories:
        cats_text = _format_categories_for_prompt(active_categories)
        user_parts.append(f"Known trigger categories:\n{cats_text}")

    user_prompt = "\n\n".join(user_parts)

    result = analyze_with_retry(system_prompt, user_prompt)

    # Normalize the response shape
    if isinstance(result, list):
        signals = result
    elif isinstance(result, dict) and "signals" in result:
        signals = result["signals"]
    else:
        signals = [result]

    # Validate and sanitize each signal
    validated = []
    for sig in signals:
        validated.append(_validate_signal_response(sig))

    return {"signals": validated}


def assess_signal_confidence(
    signal_data: dict[str, Any],
    conversation_context: Optional[str] = None,
) -> dict[str, Any]:
    """
    Assess observation bias and confidence for a signal.

    Runs a separate LLM call focused on detecting projection,
    rumination amplification, confirmation bias, and narrative
    construction patterns.

    Args:
        signal_data: The signal dict to assess.
        conversation_context: Optional conversation summary.

    Returns:
        dict with keys: confidence (float), observation_bias_flags (list)
    """
    from events_api.llm_prompts import get_prompt

    system_prompt = get_prompt(
        "confidence_assessment",
        signal_data=json.dumps(signal_data, indent=2),
        conversation_context=conversation_context or "No additional context.",
    )

    # For confidence assessment, use a simpler prompt structure
    user_prompt = (
        "Assess the confidence and observation biases for this signal. "
        "Return JSON with 'confidence' (0.0-1.0) and 'observation_bias_flags' (array of strings)."
    )

    try:
        result = analyze(system_prompt, user_prompt)
        return {
            "confidence": float(result.get("confidence", 0.7)),
            "observation_bias_flags": result.get("observation_bias_flags", []),
        }
    except (ValueError, KeyError, TypeError) as e:
        logger.warning(f"Confidence assessment failed, using defaults: {e}")
        return {
            "confidence": 0.7,
            "observation_bias_flags": [],
        }


def compute_text_embedding(text: str) -> list[float]:
    """
    Compute a lightweight embedding vector for text.

    Uses a deterministic hash-based approach for MVP. This avoids
    external embedding API calls while still providing useful
    semantic similarity for the cluster engine and graph agent.

    Post-hackathon: replace with a real embedding model (sentence-transformers
    or Anthropic's embedding API when available).

    Args:
        text: The text to embed.

    Returns:
        List of floats representing the embedding vector.
    """
    import hashlib
    import struct

    # Normalize text
    normalized = text.lower().strip()

    # Generate deterministic pseudo-embedding from text hash
    # Uses SHA-512 to get enough bytes for our target dimension
    vector = []
    seed = normalized.encode("utf-8")

    while len(vector) < EMBEDDING_DIMENSION:
        h = hashlib.sha512(seed + struct.pack("I", len(vector))).digest()
        # Convert each 4-byte chunk to a float in [-1, 1]
        for i in range(0, len(h), 4):
            if len(vector) >= EMBEDDING_DIMENSION:
                break
            val = struct.unpack("f", h[i : i + 4])[0]
            # Clamp to [-1, 1] range
            clamped = max(-1.0, min(1.0, val / 1e30)) if abs(val) > 1e30 else val
            # Normalize to [-1, 1]
            clamped = max(-1.0, min(1.0, clamped))
            vector.append(clamped)

    return vector[:EMBEDDING_DIMENSION]


def _safe_float(value, default: float) -> float:
    """Safely convert a value to float, returning default on None or error."""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _validate_signal_response(signal: dict) -> dict:
    """
    Validate and normalize a signal response from the LLM.

    Ensures all required fields are present and have correct types.
    Fills in defaults for missing optional fields.

    Args:
        signal: Raw signal dict from LLM response.

    Returns:
        Validated signal dict with all required fields.
    """
    if not isinstance(signal, dict):
        return {
            "signal_address": "SA(*, *, *, *)",
            "emotions": [],
            "participants": [],
            "confidence": 0.5,
            "provenance": "llm_inferred",
            "observation_bias_flags": [],
            "wildcards": [],
        }

    validated = {
        "signal_address": signal.get("signal_address", "SA(*, *, *, *)"),
        "emotions": signal.get("emotions") or [],
        "participants": signal.get("participants") or [],
        "confidence": _safe_float(signal.get("confidence"), 0.7),
        "provenance": "llm_inferred",
        "observation_bias_flags": signal.get("observation_bias_flags") or [],
        "wildcards": signal.get("wildcards") or [],
    }

    # Validate emotions array
    valid_emotions = []
    for emo in validated["emotions"]:
        emotion_name = None
        if isinstance(emo, dict):
            emotion_name = emo.get("emotion", emo.get("name"))

        if emotion_name is not None:
            valid_emotions.append({
                "emotion": str(emotion_name),
                "intensity": _safe_float(emo.get("intensity"), 5.0),
                "source_coordinate": str(emo.get("source_coordinate", "unknown")),
                "source_description": str(emo.get("source_description", "")),
                "confidence": _safe_float(emo.get("confidence"), 0.7),
            })
    validated["emotions"] = valid_emotions

    # Validate participants array
    valid_participants = []
    for p in validated["participants"]:
        if isinstance(p, dict) and "name" in p:
            valid_participants.append({
                "name": str(p["name"]),
                "role": str(p.get("role", "mentioned")),
                "confidence": _safe_float(p.get("confidence"), 0.7),
            })
    validated["participants"] = valid_participants

    # Validate observation bias flags
    # LLM may return strings ("projection") or dicts ({"type": "projection", ...})
    valid_flags = {
        "projection", "rumination_amplification",
        "confirmation_bias", "narrative_construction",
    }
    normalized_flags = []
    for f in validated["observation_bias_flags"]:
        flag_name = f.get("type", "") if isinstance(f, dict) else f
        if isinstance(flag_name, str) and flag_name in valid_flags:
            normalized_flags.append(flag_name)
    validated["observation_bias_flags"] = normalized_flags

    return validated
