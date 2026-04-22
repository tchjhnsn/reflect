"""
Backend scoring for the civic journey.

Ports the algorithms from apps/journey/lib/scoring.ts to Python,
using graph queries where the ontology is seeded, and falling back
to flat-property reads when it isn't.

Two profiles are computed:
  - ValueProfile: ranked hierarchy of values from provocation responses
  - SoulProfile: revealed psyche ordering vs stated ordering, regime type
"""

import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# All 12 value IDs (canonical order matches ontology_data.py)
ALL_VALUE_IDS = [
    "dignity", "liberty", "justice",
    "order", "authority", "sovereignty", "equality",
    "prosperity", "solidarity", "pluralism", "merit", "stewardship",
]


# =============================================================================
# ValueProfile
# =============================================================================

def compute_value_profile(*, workspace_id: str, owner_user_id: int) -> dict | None:
    """
    Compute a ValueProfile from the user's provocation responses.

    Algorithm (mirrors scoring.ts computeProfile):
      1. For each response, tally which values were protected/sacrificed.
      2. netScore = timesProtected - timesSacrificed.
      3. Rank descending by netScore.
      4. Confidence = timesAppeared / totalResponses.
      5. Track average deliberation time per value.

    Returns a dict matching the TypeScript ValueProfile shape, or None
    if the user has no responses.
    """
    try:
        from neomodel import db

        results, _ = db.cypher_query(
            """
            MATCH (u:UserProfile {workspace_id: $ws, owner_user_id: $uid})
                  -[:HAS_PROVOCATION_RESPONSE]->(r:ProvocationResponseNode)
            RETURN r.protected_values  AS protected,
                   r.sacrificed_values AS sacrificed,
                   r.deliberation_time_ms AS delib_ms
            ORDER BY r.timestamp ASC
            """,
            {"ws": workspace_id, "uid": owner_user_id},
        )

        if not results:
            return None

        # Initialize scores
        scores = {}
        delib_totals = {}
        for v_id in ALL_VALUE_IDS:
            scores[v_id] = {"timesProtected": 0, "timesSacrificed": 0, "avgDeliberationMs": 0}
            delib_totals[v_id] = {"total": 0, "count": 0}

        total_responses = len(results)

        for protected, sacrificed, delib_ms in results:
            protected = protected or []
            sacrificed = sacrificed or []
            delib_ms = delib_ms or 0

            for v_id in protected:
                if v_id in scores:
                    scores[v_id]["timesProtected"] += 1
                    delib_totals[v_id]["total"] += delib_ms
                    delib_totals[v_id]["count"] += 1

            for v_id in sacrificed:
                if v_id in scores:
                    scores[v_id]["timesSacrificed"] += 1
                    delib_totals[v_id]["total"] += delib_ms
                    delib_totals[v_id]["count"] += 1

        # Compute avg deliberation
        for v_id in ALL_VALUE_IDS:
            dt = delib_totals[v_id]
            scores[v_id]["avgDeliberationMs"] = (
                round(dt["total"] / dt["count"]) if dt["count"] > 0 else 0
            )

        # Rank by netScore descending
        ranked_items = []
        for v_id in ALL_VALUE_IDS:
            s = scores[v_id]
            times_appeared = s["timesProtected"] + s["timesSacrificed"]
            net = s["timesProtected"] - s["timesSacrificed"]
            confidence = round(times_appeared / max(total_responses, 1), 2)
            ranked_items.append((v_id, net, confidence))

        ranked_items.sort(key=lambda x: x[1], reverse=True)

        hierarchy = [
            {"valueId": v_id, "rank": i + 1, "confidence": conf}
            for i, (v_id, _, conf) in enumerate(ranked_items)
        ]

        return {
            "hierarchy": hierarchy,
            "scores": scores,
            "computedAt": datetime.now(timezone.utc).isoformat(),
            "scenarioCount": total_responses,
        }

    except Exception as exc:
        logger.warning(
            "Failed to compute value profile for user %s in workspace %s: %s",
            owner_user_id, workspace_id, exc,
        )
        return None


# =============================================================================
# SoulProfile
# =============================================================================

def compute_soul_profile(*, workspace_id: str, owner_user_id: int) -> dict | None:
    """
    Compute a SoulProfile from the user's provocation responses and
    stated soul ordering.

    Algorithm (mirrors scoring.ts computeSoulProfile):
      1. For each response, the servedSoulPart tells us who won.
      2. Count wins across three pairings: reason-spirit, reason-appetite,
         spirit-appetite.
      3. Derive a revealed ordering from pairwise winners.
      4. Classify the regime type.
      5. Identify cardinal virtues present.

    Returns a dict matching the TypeScript SoulProfile shape, or None
    if the user has no responses.
    """
    try:
        from neomodel import db

        # Fetch responses and stated ordering
        results, _ = db.cypher_query(
            """
            MATCH (u:UserProfile {workspace_id: $ws, owner_user_id: $uid})
            OPTIONAL MATCH (u)-[:HAS_PROVOCATION_RESPONSE]->(r:ProvocationResponseNode)
            RETURN u.soul_ordering AS stated,
                   collect(r.served_soul_part) AS served_parts
            """,
            {"ws": workspace_id, "uid": owner_user_id},
        )

        if not results or not results[0]:
            return None

        stated_ordering_raw = results[0][0]
        served_parts = [p for p in (results[0][1] or []) if p is not None]

        if not served_parts:
            return None

        # Parse stated ordering (stored as JSON or dict)
        stated_ordering = _parse_soul_ordering(stated_ordering_raw)

        # Count wins
        frequencies = {
            "reasonVsSpirit": {"reasonWins": 0, "spiritWins": 0, "total": 0},
            "reasonVsAppetite": {"reasonWins": 0, "appetiteWins": 0, "total": 0},
            "spiritVsAppetite": {"spiritWins": 0, "appetiteWins": 0, "total": 0},
        }

        for part in served_parts:
            if part == "reason":
                frequencies["reasonVsSpirit"]["total"] += 1
                frequencies["reasonVsSpirit"]["reasonWins"] += 1
                frequencies["reasonVsAppetite"]["total"] += 1
                frequencies["reasonVsAppetite"]["reasonWins"] += 1
            elif part == "spirit":
                frequencies["reasonVsSpirit"]["total"] += 1
                frequencies["reasonVsSpirit"]["spiritWins"] += 1
                frequencies["spiritVsAppetite"]["total"] += 1
                frequencies["spiritVsAppetite"]["spiritWins"] += 1
            elif part == "appetite":
                frequencies["reasonVsAppetite"]["total"] += 1
                frequencies["reasonVsAppetite"]["appetiteWins"] += 1
                frequencies["spiritVsAppetite"]["total"] += 1
                frequencies["spiritVsAppetite"]["appetiteWins"] += 1

        # Determine pairwise winners
        pair_winners = {
            "reasonVsSpirit": (
                "reason"
                if frequencies["reasonVsSpirit"]["reasonWins"]
                >= frequencies["reasonVsSpirit"]["spiritWins"]
                else "spirit"
            ),
            "reasonVsAppetite": (
                "reason"
                if frequencies["reasonVsAppetite"]["reasonWins"]
                >= frequencies["reasonVsAppetite"]["appetiteWins"]
                else "appetite"
            ),
            "spiritVsAppetite": (
                "spirit"
                if frequencies["spiritVsAppetite"]["spiritWins"]
                >= frequencies["spiritVsAppetite"]["appetiteWins"]
                else "appetite"
            ),
        }

        # Count total cross-pair wins
        total_wins = {"reason": 0, "spirit": 0, "appetite": 0}
        if pair_winners["reasonVsSpirit"] == "reason":
            total_wins["reason"] += 1
        else:
            total_wins["spirit"] += 1
        if pair_winners["reasonVsAppetite"] == "reason":
            total_wins["reason"] += 1
        else:
            total_wins["appetite"] += 1
        if pair_winners["spiritVsAppetite"] == "spirit":
            total_wins["spirit"] += 1
        else:
            total_wins["appetite"] += 1

        revealed_ordering = _derive_ordering(total_wins)
        regime = _classify_regime(revealed_ordering)
        virtues = _identify_virtues(revealed_ordering)

        return {
            "revealedOrdering": revealed_ordering,
            "statedOrdering": stated_ordering,
            "frequencies": frequencies,
            "regime": regime,
            "virtuesPresent": virtues,
        }

    except Exception as exc:
        logger.warning(
            "Failed to compute soul profile for user %s in workspace %s: %s",
            owner_user_id, workspace_id, exc,
        )
        return None


# =============================================================================
# Helpers (mirror scoring.ts internal functions)
# =============================================================================

def _parse_soul_ordering(raw) -> dict:
    """Parse the stored soul ordering into a standard dict."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        import json
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            pass
    return {"type": "equal"}


def _derive_ordering(wins: dict) -> dict:
    """Derive a SoulOrdering from pairwise win counts."""
    parts = sorted(wins.keys(), key=lambda p: wins[p], reverse=True)

    if wins[parts[0]] == wins[parts[1]] == wins[parts[2]]:
        return {"type": "equal"}

    if wins[parts[0]] == wins[parts[1]]:
        return {
            "type": "co-rulers",
            "rulers": [parts[0], parts[1]],
            "subordinate": parts[2],
        }

    return {
        "type": "one-rules",
        "ruler": parts[0],
        "second": parts[1],
        "third": parts[2],
    }


def _classify_regime(ordering: dict) -> str:
    """Classify the regime type from a revealed ordering."""
    if ordering["type"] == "equal":
        return "democratic"

    if ordering["type"] == "one-rules":
        ruler = ordering["ruler"]
        if ruler == "reason":
            return "aristocratic"
        if ruler == "spirit":
            return "timocratic"
        if ruler == "appetite":
            return "oligarchic"

    if ordering["type"] == "co-rulers":
        rulers = ordering.get("rulers", [])
        if "reason" in rulers:
            return "aristocratic"
        if "spirit" in rulers:
            return "timocratic"
        return "oligarchic"

    return "democratic"


def _identify_virtues(ordering: dict) -> list:
    """Identify cardinal virtues present in the ordering."""
    virtues = []

    o_type = ordering["type"]
    ruler = ordering.get("ruler")
    rulers = ordering.get("rulers", [])
    second = ordering.get("second")

    # Wisdom: Reason is in command or co-ruling
    if (o_type == "one-rules" and ruler == "reason") or (
        o_type == "co-rulers" and "reason" in rulers
    ):
        virtues.append("wisdom")

    # Courage: Spirit serves reason (reason leads and spirit follows)
    if o_type == "one-rules" and ruler == "reason":
        virtues.append("courage")
    elif o_type == "co-rulers" and "reason" in rulers and "spirit" in rulers:
        virtues.append("courage")

    # Moderation: Appetite is governed (not ruling)
    if o_type == "one-rules" and ruler != "appetite":
        virtues.append("moderation")
    elif o_type == "co-rulers" and "appetite" not in rulers:
        virtues.append("moderation")

    # Justice: All three properly ordered (reason rules, spirit second)
    if o_type == "one-rules" and ruler == "reason" and second == "spirit":
        virtues.append("justice")

    return virtues
