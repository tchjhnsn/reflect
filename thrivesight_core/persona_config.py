"""
ThriveSight Persona Configuration — Pre-built AI personas with configurable thresholds.

Each persona defines how the system behaves during Mode 2 (Conversational AI)
interactions. Personas control:
- System prompt tone and approach
- How much context to include from the graph
- Minimum confidence threshold for citing signals
- Minimum cluster strength to surface
- How aggressively to flag observation bias

Users can:
1. Choose a pre-built persona
2. Override any individual setting
3. The persona provides defaults; user preferences always win

All persona feedback is signal — choosing a persona, switching, or overriding
settings creates graph data about user preferences.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PersonaConfig:
    """
    Configuration for an AI persona.

    Attributes:
        name: Human-readable persona name
        description: What this persona is like
        system_prompt_modifier: Additional system prompt text
        context_depth: How many related signals to include (5/10/20)
        confidence_threshold: Minimum signal confidence to cite (0.0-1.0)
        cluster_surfacing_threshold: Minimum cluster member count to mention
        cluster_strength_threshold: Minimum cluster strength to surface
        observation_bias_aggressiveness: How eagerly to flag bias
        max_context_tokens: Token budget for context packet
        include_reasoning_traces: Whether to include Insight/Reflection history
        include_pending_insights: Whether to include background detections
    """

    name: str
    description: str
    system_prompt_modifier: str

    # Context assembly parameters
    context_depth: int = 10
    confidence_threshold: float = 0.5
    cluster_surfacing_threshold: int = 3
    cluster_strength_threshold: float = 2.0
    observation_bias_aggressiveness: str = "medium"  # "low" | "medium" | "high"

    # Token management
    max_context_tokens: int = 2000

    # Feature toggles
    include_reasoning_traces: bool = True
    include_pending_insights: bool = True


# ──────────────────────────────────────────────────────────────────────────────
# PRE-BUILT PERSONAS
# ──────────────────────────────────────────────────────────────────────────────


DIRECT_CHALLENGER = PersonaConfig(
    name="Direct Challenger",
    description=(
        "A bold, honest companion that challenges assumptions and pushes "
        "you to examine blind spots. Won't sugarcoat observations."
    ),
    system_prompt_modifier=(
        "You are direct and honest. When you notice patterns, contradictions, or "
        "potential blind spots, point them out clearly. Don't hedge or soften — "
        "the user has chosen you because they want to be challenged. Use phrases "
        "like 'I want to push back on that' or 'Have you considered that you might "
        "be...' Reference specific signals and patterns from their history. "
        "Challenge projections directly — if the data suggests they might be "
        "avoiding accountability, say so warmly but clearly."
    ),
    context_depth=20,
    confidence_threshold=0.3,  # Lower threshold → surfaces more
    cluster_surfacing_threshold=2,
    cluster_strength_threshold=1.5,
    observation_bias_aggressiveness="high",
    max_context_tokens=3000,
    include_reasoning_traces=True,
    include_pending_insights=True,
)


GENTLE_EXPLORER = PersonaConfig(
    name="Gentle Explorer",
    description=(
        "A warm, patient guide that helps you explore emotions at your "
        "own pace. Asks questions rather than making assertions."
    ),
    system_prompt_modifier=(
        "You are warm, patient, and curious. Guide the user to explore their "
        "emotions through questions rather than assertions. Use phrases like "
        "'I'm curious about...' or 'What was that like for you?' Only surface "
        "patterns and clusters when you're quite confident about them. Never "
        "rush the user toward conclusions — let them arrive at insights "
        "naturally. When observation bias is detected, frame it as a gentle "
        "question: 'I wonder if there's another way to see this?'"
    ),
    context_depth=10,
    confidence_threshold=0.7,  # Higher threshold → only confident signals
    cluster_surfacing_threshold=5,
    cluster_strength_threshold=3.0,
    observation_bias_aggressiveness="low",
    max_context_tokens=1500,
    include_reasoning_traces=True,
    include_pending_insights=False,  # Don't surface background detections
)


NEUTRAL_OBSERVER = PersonaConfig(
    name="Neutral Observer",
    description=(
        "A balanced, analytical companion that presents observations "
        "without judgment. Focuses on data and patterns, not emotions."
    ),
    system_prompt_modifier=(
        "You are analytical and balanced. Present observations as data points, "
        "not emotional conclusions. Use phrases like 'The data suggests...' or "
        "'A pattern appears in...' Reference specific signal addresses and cluster "
        "IDs when discussing patterns. Present multiple interpretations when data "
        "is ambiguous. Don't push the user toward any particular interpretation — "
        "lay out the evidence and let them decide what it means."
    ),
    context_depth=15,
    confidence_threshold=0.5,
    cluster_surfacing_threshold=3,
    cluster_strength_threshold=2.0,
    observation_bias_aggressiveness="medium",
    max_context_tokens=2500,
    include_reasoning_traces=True,
    include_pending_insights=True,
)


# ──────────────────────────────────────────────────────────────────────────────
# PERSONA REGISTRY
# ──────────────────────────────────────────────────────────────────────────────


PERSONAS = {
    "direct_challenger": DIRECT_CHALLENGER,
    "gentle_explorer": GENTLE_EXPLORER,
    "neutral_observer": NEUTRAL_OBSERVER,
}

DEFAULT_PERSONA = "neutral_observer"


def get_persona(persona_id: str, overrides: Optional[dict] = None) -> PersonaConfig:
    """
    Get a persona configuration, optionally with user overrides.

    Args:
        persona_id: Persona identifier (e.g., "direct_challenger")
        overrides: Optional dict of settings to override

    Returns:
        PersonaConfig with overrides applied

    Raises:
        KeyError: If persona_id not found
    """
    if persona_id not in PERSONAS:
        raise KeyError(
            f"Unknown persona: {persona_id}. "
            f"Available: {list(PERSONAS.keys())}"
        )

    base = PERSONAS[persona_id]

    if not overrides:
        return base

    # Create a new config with overrides
    config_dict = {
        "name": base.name,
        "description": base.description,
        "system_prompt_modifier": base.system_prompt_modifier,
        "context_depth": base.context_depth,
        "confidence_threshold": base.confidence_threshold,
        "cluster_surfacing_threshold": base.cluster_surfacing_threshold,
        "cluster_strength_threshold": base.cluster_strength_threshold,
        "observation_bias_aggressiveness": base.observation_bias_aggressiveness,
        "max_context_tokens": base.max_context_tokens,
        "include_reasoning_traces": base.include_reasoning_traces,
        "include_pending_insights": base.include_pending_insights,
    }

    # Apply overrides (only for known fields)
    for key, value in overrides.items():
        if key in config_dict:
            config_dict[key] = value

    return PersonaConfig(**config_dict)


def list_personas() -> list[dict]:
    """
    List all available personas with their descriptions.

    Returns:
        List of dicts with name, id, and description
    """
    return [
        {
            "id": pid,
            "name": persona.name,
            "description": persona.description,
        }
        for pid, persona in PERSONAS.items()
    ]
