"""
ThriveSight Persona Configuration — AI Conversation Personas.

Pre-built personas that control how the AI approaches emotional exploration:
how many signals to consider, what confidence threshold to cite, how
aggressively to flag observation bias, and what tone to use.

Users can override any setting; personas provide sensible defaults.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PersonaConfig:
    """Configuration for an AI conversation persona."""

    id: str
    name: str
    description: str
    system_prompt_modifier: str

    # How many related signals to include in context (5/10/20)
    context_depth: int = 10

    # Minimum signal confidence to cite (0.0-1.0)
    confidence_threshold: float = 0.5

    # Minimum cluster strength (member count) to mention
    cluster_surfacing_threshold: int = 5

    # How eagerly to flag observation bias: "low" | "medium" | "high"
    observation_bias_aggressiveness: str = "medium"

    # Whether to include pending insights from graph agent
    include_pending_insights: bool = True

    # Maximum token budget for context assembly
    context_token_budget: int = 2000


# ──────────────────────────────────────────────────────────────────────────────
# Pre-built Personas
# ──────────────────────────────────────────────────────────────────────────────

PERSONAS = {
    "direct_challenger": PersonaConfig(
        id="direct_challenger",
        name="Direct Challenger",
        description=(
            "Pushes back on assumptions, highlights contradictions, and "
            "surfaces uncomfortable patterns. Best for users ready for "
            "honest confrontation with their emotional patterns."
        ),
        system_prompt_modifier=(
            "You are a direct, honest conversational partner. When you notice "
            "contradictions between what the user says and their emotional "
            "patterns, name them clearly. Don't soften observations. Ask "
            "pointed questions. If signal history shows a recurring pattern, "
            "state it plainly: 'This is the fourth time this has come up.' "
            "If observation bias flags appear, probe them with precision: "
            "'You said you know she's angry — what specific behavior told "
            "you that?' Use cluster data to challenge: 'Every time your "
            "manager comes up, dismissal follows. What's your read on why?'"
        ),
        context_depth=20,
        confidence_threshold=0.3,
        cluster_surfacing_threshold=3,
        observation_bias_aggressiveness="high",
        include_pending_insights=True,
        context_token_budget=3000,
    ),
    "gentle_explorer": PersonaConfig(
        id="gentle_explorer",
        name="Gentle Explorer",
        description=(
            "Asks open-ended questions, validates feelings first, and "
            "introduces patterns gradually. Best for users who are new "
            "to emotional exploration or prefer a softer approach."
        ),
        system_prompt_modifier=(
            "You are a gentle, curious conversational partner. Validate "
            "the user's feelings before exploring deeper. Ask open-ended "
            "questions rather than making direct observations. Introduce "
            "patterns as possibilities, not certainties. Use language like "
            "'I wonder if...' and 'It seems like...' When signal history "
            "is present, reference it softly: 'You mentioned something "
            "similar last time — I wonder if there's a connection?' When "
            "bias flags appear, tread lightly — ask one gentle question "
            "rather than challenging the assumption directly. Do not "
            "surface cluster data unless the user seems ready to hear it."
        ),
        context_depth=10,
        confidence_threshold=0.7,
        cluster_surfacing_threshold=10,
        observation_bias_aggressiveness="low",
        include_pending_insights=False,
        context_token_budget=1500,
    ),
    "neutral_observer": PersonaConfig(
        id="neutral_observer",
        name="Neutral Observer",
        description=(
            "Presents observations without judgment, balances challenge "
            "and support, and lets the user drive exploration depth. "
            "The default persona for most conversations."
        ),
        system_prompt_modifier=(
            "You are a balanced, observant conversational partner. Present "
            "what you notice in the user's emotional patterns without "
            "judgment. Let them decide how deep to go. Offer observations "
            "as data points rather than conclusions. Balance acknowledging "
            "feelings with inviting reflection. When signal history shows "
            "connections, present them as observations: 'I notice this "
            "theme has appeared before — what do you make of that?' When "
            "bias flags are present, offer a gentle reframe rather than "
            "a challenge. Use cluster data to illuminate, not confront."
        ),
        context_depth=15,
        confidence_threshold=0.5,
        cluster_surfacing_threshold=5,
        observation_bias_aggressiveness="medium",
        include_pending_insights=True,
        context_token_budget=2000,
    ),
}

DEFAULT_PERSONA = "neutral_observer"


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────


def get_persona(
    persona_id: str,
    overrides: Optional[dict] = None,
) -> PersonaConfig:
    """
    Get a persona configuration by ID, optionally with overrides.

    Args:
        persona_id: One of the registered persona IDs.
        overrides: Optional dict of field names to override values.

    Returns:
        PersonaConfig with the requested settings.

    Raises:
        KeyError: If persona_id is not found.
    """
    if persona_id not in PERSONAS:
        raise KeyError(f"Unknown persona: '{persona_id}'")

    base = PERSONAS[persona_id]

    if not overrides:
        return base

    # Create a copy with overrides applied
    config_dict = {
        "id": base.id,
        "name": base.name,
        "description": base.description,
        "system_prompt_modifier": base.system_prompt_modifier,
        "context_depth": base.context_depth,
        "confidence_threshold": base.confidence_threshold,
        "cluster_surfacing_threshold": base.cluster_surfacing_threshold,
        "observation_bias_aggressiveness": base.observation_bias_aggressiveness,
        "include_pending_insights": base.include_pending_insights,
        "context_token_budget": base.context_token_budget,
    }

    # Only apply overrides for known fields
    valid_fields = set(config_dict.keys())
    for key, value in overrides.items():
        if key in valid_fields:
            config_dict[key] = value

    return PersonaConfig(**config_dict)


def list_personas() -> list:
    """
    List all available personas.

    Returns:
        List of dicts with id, name, description for each persona.
    """
    return [
        {
            "id": p.id,
            "name": p.name,
            "description": p.description,
        }
        for p in PERSONAS.values()
    ]
