"""
ThriveSight LLM Prompt Templates — Signal Address System.

Centralized prompt registry for all LLM interactions. Prompts are versioned:
- V2: Legacy screenplay analysis prompts (retained for backward compatibility)
- V3: Signal Address System prompts for emotion detection, wildcard exploration,
  confidence assessment, entity extraction, and cluster reasoning.

All prompts support string formatting with keyword arguments via get_prompt().
"""


# ──────────────────────────────────────────────────────────────────────────────
# V2 Prompts (Legacy — Screenplay Analysis)
# ──────────────────────────────────────────────────────────────────────────────

V2_PROMPTS = {
    "parser": (
        "You are a screenplay parser. Given raw screenplay text, extract "
        "structured scene data including scene headings, action lines, "
        "character names, and dialogue. Return the result as JSON."
    ),
    "signal": (
        "You are an emotional signal detector. Given a dialogue turn, "
        "identify the emotional signal present: the trigger action, "
        "the emotion expressed, and the intensity on a scale of 1-10."
    ),
    "pattern": (
        "You are a pattern analyst. Given a set of emotional signals "
        "grouped by trigger category and emotion, identify recurring "
        "patterns and generate a hypothesis about the underlying dynamic."
    ),
    "reflection": (
        "You are a reflective partner. Given an emotional signal and "
        "its context, generate a thoughtful reflection that helps the "
        "user understand the pattern at play."
    ),
    "reframe": (
        "You are a cognitive reframing assistant. Given a dialogue turn "
        "and its emotional signal, generate an alternative perspective "
        "that the speaker might not have considered."
    ),
}


# ──────────────────────────────────────────────────────────────────────────────
# V3 Prompts (Signal Address System)
# ──────────────────────────────────────────────────────────────────────────────

V3_PROMPTS = {
    "signal_generation": (
        "You are a Signal Address System analyst. Given a user's message "
        "and conversation context, generate one or more emotional signals.\n\n"
        "For each signal, produce:\n"
        "1. A Signal Address: SA(context, person, action, temporal) where each "
        "coordinate is a specific value or * for unknown.\n"
        "2. An emotions array: each emotion has a name, intensity (1-10), "
        "source_coordinate (which SA dimension it arises from), "
        "source_description (why), and confidence (0.0-1.0).\n"
        "3. A participants list: each person mentioned with their role "
        "(subject, primary_actor, amplifier, witness, mentioned).\n"
        "4. observation_bias_flags: any detected biases "
        "(projection, rumination_amplification, confirmation_bias, "
        "narrative_construction).\n"
        "5. wildcards: which coordinates could not be determined.\n"
        "6. confidence: overall confidence score (0.0-1.0).\n\n"
        "Return JSON."
    ),
    "wildcard_exploration": (
        "You are exploring an incomplete Signal Address. The current address "
        "is: {signal_address}\n\n"
        "The following coordinates are wildcards (unknown): {wildcards}\n\n"
        "Context from the conversation: {context}\n\n"
        "Generate a thoughtful question for each wildcard coordinate that "
        "would help the user identify the missing dimension. The question "
        "should feel natural, not clinical. It should build on what we "
        "already know from the resolved coordinates."
    ),
    "emotion_attribution": (
        "You are an emotion attribution analyst. Given a signal with "
        "multiple emotions, trace each emotion back to the specific "
        "SA coordinate or participant that generated it.\n\n"
        "For example, if someone feels both anger and shame in a work "
        "meeting, anger might trace to the person (manager's dismissal) "
        "while shame traces to the context (public setting).\n\n"
        "Signal data: {signal_data}\n"
        "Participants: {participants}\n\n"
        "Return a JSON array mapping each emotion to its source."
    ),
    "confidence_assessment": (
        "You are an observation bias detector. Given a signal and its "
        "context, assess the confidence level and identify any potential "
        "observation biases.\n\n"
        "Bias types to check:\n"
        "- projection: attributing your own feelings to others\n"
        "- rumination_amplification: repeated thinking inflating intensity\n"
        "- confirmation_bias: selectively noticing pattern-confirming signals\n"
        "- narrative_construction: building a story that may not match reality\n\n"
        "Signal: {signal_data}\n"
        "Context: {conversation_context}\n\n"
        "Return confidence (0.0-1.0) and any bias flags detected."
    ),
    "entity_extraction": (
        "You are an entity extractor for emotional signals. Given a user "
        "message, extract:\n"
        "1. Persons mentioned (by name or role)\n"
        "2. Contexts (settings: work, home, social, etc.)\n"
        "3. Actions (what happened: dismissal, praise, etc.)\n"
        "4. Temporal references (when: yesterday, Monday, etc.)\n\n"
        "Message: {message}\n\n"
        "Return structured JSON with persons, contexts, actions, temporal arrays."
    ),
    "cluster_reasoning": (
        "You are a cluster interpreter. Given a cluster of related signals, "
        "generate a human-readable interpretation of what this grouping "
        "means emotionally.\n\n"
        "Cluster type: {cluster_type}\n"
        "Shared coordinates: {shared_coordinates}\n"
        "Divergent dimensions: {divergent_dimensions}\n"
        "Member signals: {member_signals}\n\n"
        "Provide a brief interpretation and a suggested hypothesis for "
        "why these signals cluster together."
    ),
}


# Combined registry
ALL_PROMPTS = {**V2_PROMPTS, **V3_PROMPTS}


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────


def get_prompt(prompt_id: str, **kwargs) -> str:
    """
    Get a prompt template by ID, optionally with formatting.

    Args:
        prompt_id: Key in the prompt registry.
        **kwargs: Format arguments to interpolate into the template.

    Returns:
        The formatted prompt string.

    Raises:
        KeyError: If prompt_id is not found.
    """
    if prompt_id not in ALL_PROMPTS:
        raise KeyError(f"Unknown prompt: '{prompt_id}'")

    template = ALL_PROMPTS[prompt_id]

    if kwargs:
        return template.format(**kwargs)

    return template


def build_system_prompt(
    base_prompt: str,
    persona_modifier: str = "",
    context_packet: str = "",
) -> str:
    """
    Assemble a complete system prompt from components.

    Args:
        base_prompt: The core instruction prompt.
        persona_modifier: Persona-specific tone instructions.
        context_packet: Assembled graph context for the LLM.

    Returns:
        Complete system prompt string.
    """
    parts = [base_prompt]

    if persona_modifier:
        parts.append(f"\n\n## Persona\n{persona_modifier}")

    if context_packet:
        parts.append(f"\n\n## Context from Graph\n{context_packet}")

    return "\n".join(parts)
