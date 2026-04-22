"""
Format Journey profile data for injection into Reflect's AI context.

Produces a compact natural-language summary of the user's civic identity
that gets appended to the LLM context packet. Designed to be concise
(~100-150 tokens) so it fits within the 2000-token context budget.
"""

import logging

logger = logging.getLogger(__name__)

# Maps for human-readable display
_PATH_NAMES = {
    "wanderer": "Wanderer (exploring freely)",
    "sovereign": "Sovereign (self-directed)",
    "philosopher": "Philosopher (guided inquiry)",
}

_REGIME_NAMES = {
    "aristocratic": "Aristocratic (Reason rules)",
    "timocratic": "Timocratic (Spirit rules)",
    "oligarchic": "Oligarchic (Appetite rules)",
    "democratic": "Democratic (equal parts)",
    "tyrannical": "Tyrannical (one part dominates destructively)",
}

_ORDERING_LABELS = {
    "one-rules": "clear hierarchy",
    "co-rulers": "shared governance",
    "equal": "balanced/equal",
}


def format_journey_context(
    journey_state: dict | None,
    value_profile: dict | None,
    soul_profile: dict | None,
) -> str | None:
    """
    Format the user's Journey data as a natural-language context block.

    Returns a string to be appended to the LLM context_packet, or None
    if the user has no meaningful Journey data yet.
    """
    parts = []

    # Gate: only produce context if the user has done something meaningful
    has_journey = journey_state and journey_state.get("journey_phase")
    has_values = value_profile and value_profile.get("hierarchy")
    has_soul = soul_profile and soul_profile.get("revealedOrdering")

    if not (has_journey or has_values or has_soul):
        return None

    parts.append("User's civic identity (from Journey discovery):")

    # --- Journey state ---
    if has_journey:
        path_id = journey_state.get("path_id")
        phase = journey_state.get("journey_phase")
        if path_id:
            path_label = _PATH_NAMES.get(path_id, path_id)
            parts.append(f"  Path: {path_label}")
            # Add mode detail
            if path_id == "philosopher":
                mode = journey_state.get("philosopher_mode")
                if mode:
                    parts.append(f"  Mode: {mode}")
            elif path_id == "sovereign":
                mode = journey_state.get("sovereign_mode")
                if mode:
                    parts.append(f"  Mode: {mode}")
                statement = journey_state.get("sovereign_end_statement")
                if statement:
                    # Truncate long statements
                    stmt = statement[:120] + "..." if len(statement) > 120 else statement
                    parts.append(f"  Goal: \"{stmt}\"")
        if phase:
            parts.append(f"  Phase: {phase}")

    # --- Soul profile ---
    if has_soul:
        revealed = soul_profile["revealedOrdering"]
        regime = soul_profile.get("regime", "unknown")
        virtues = soul_profile.get("virtuesPresent", [])

        ordering_type = revealed.get("type", "equal")
        ordering_label = _ORDERING_LABELS.get(ordering_type, ordering_type)

        if ordering_type == "one-rules":
            ruler = revealed.get("ruler", "?")
            second = revealed.get("second", "?")
            third = revealed.get("third", "?")
            parts.append(
                f"  Soul ordering: {ruler.title()} rules, {second.title()} second, "
                f"{third.title()} third ({ordering_label})"
            )
        elif ordering_type == "co-rulers":
            rulers = revealed.get("rulers", [])
            sub = revealed.get("subordinate", "?")
            ruler_names = " & ".join(r.title() for r in rulers)
            parts.append(f"  Soul ordering: {ruler_names} co-rule, {sub.title()} subordinate")
        else:
            parts.append("  Soul ordering: balanced/equal across all three parts")

        regime_label = _REGIME_NAMES.get(regime, regime)
        parts.append(f"  Regime type: {regime_label}")

        if virtues:
            parts.append(f"  Virtues present: {', '.join(virtues)}")

        # Stated vs revealed comparison
        stated = soul_profile.get("statedOrdering")
        if stated and stated.get("type") != revealed.get("type"):
            parts.append(
                "  Note: Revealed ordering differs from the user's stated ordering "
                "(their actions suggest a different internal hierarchy than what they believe)"
            )

    # --- Value profile ---
    if has_values:
        hierarchy = value_profile["hierarchy"]
        top_5 = [h["valueId"] for h in hierarchy[:5]]
        bottom_3 = [h["valueId"] for h in hierarchy[-3:]]
        parts.append(f"  Top values: {', '.join(top_5)}")
        parts.append(f"  Lowest priority: {', '.join(bottom_3)}")

        # Deliberation insight — find the value with highest avg deliberation
        scores = value_profile.get("scores", {})
        max_delib_id = None
        max_delib_ms = 0
        for v_id, s in scores.items():
            avg = s.get("avgDeliberationMs", 0)
            if avg > max_delib_ms:
                max_delib_ms = avg
                max_delib_id = v_id
        if max_delib_id and max_delib_ms > 0:
            parts.append(
                f"  Most conflicted about: {max_delib_id} "
                f"(avg {max_delib_ms / 1000:.1f}s deliberation)"
            )

    return "\n".join(parts)
