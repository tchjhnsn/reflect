"""
Graph synchronisation helpers.

Ensures Django relational-DB users are mirrored as UserProfile nodes
in Neo4j so the knowledge graph is queryable per-user.

Zone 1: Proprietary — sync strategy is internal.
"""

import logging

from .identity import build_self_alias_names
from .neo4j_client import cypher_query

logger = logging.getLogger(__name__)


def _delete_graph_nodes(*, workspace_ids: list[str] | None = None, owner_user_id: int | None = None) -> dict:
    workspace_ids = [str(workspace_id) for workspace_id in (workspace_ids or []) if workspace_id]
    where_clauses = []
    params = {}

    if workspace_ids:
        where_clauses.append("n.workspace_id IN $workspace_ids")
        params["workspace_ids"] = workspace_ids
    if owner_user_id is not None:
        where_clauses.append("n.owner_user_id = $owner_user_id")
        params["owner_user_id"] = owner_user_id

    if not where_clauses:
        return {"deleted_nodes": 0}

    where_sql = " OR ".join(where_clauses)
    count_query = f"""
    MATCH (n)
    WHERE {where_sql}
    RETURN count(DISTINCT n) AS deleted_nodes
    """
    delete_query = f"""
    MATCH (n)
    WHERE {where_sql}
    DETACH DELETE n
    """

    rows, _ = cypher_query(count_query, params)
    deleted_nodes = rows[0][0] if rows and rows[0] else 0
    if deleted_nodes:
        cypher_query(delete_query, params)

    return {"deleted_nodes": deleted_nodes}


def delete_workspace_graph_data(*, workspace_id: str) -> dict:
    """Delete all graph nodes scoped to a single workspace."""
    return _delete_graph_nodes(workspace_ids=[workspace_id])


def delete_user_graph_data(*, workspace_ids: list[str] | None = None, owner_user_id: int) -> dict:
    """Delete all graph nodes tied to a user's owned workspaces and user id."""
    return _delete_graph_nodes(workspace_ids=workspace_ids or [], owner_user_id=owner_user_id)


def ensure_user_profile_in_graph(user, workspace):
    """
    Create or update a UserProfile node in Neo4j for the given Django user.

    Idempotent — safe to call on every login/signup. Uses workspace_id +
    owner_user_id as the canonical identity key.

    Parameters
    ----------
    user : django.contrib.auth.models.User
        The authenticated Django user.
    workspace : events_api.models.Workspace
        The user's personal workspace (provides workspace_id).

    Returns
    -------
    dict | None
        The UserProfile node properties if created/updated, None on failure.
    """
    try:
        from neomodel import db

        workspace_id = str(workspace.id)
        owner_user_id = user.id

        query = """
        MERGE (u:UserProfile {workspace_id: $workspace_id, owner_user_id: $owner_user_id})
        ON CREATE SET
            u.uid                        = randomUUID(),
            u.username                   = $username,
            u.email                      = $email,
            u.total_messages             = 0,
            u.created_at                 = datetime(),
            u.current_provocation_index  = 0,
            u.socratic_chariot_revealed  = false,
            u.socratic_tier_revealed     = false
        ON MATCH SET
            u.username        = $username,
            u.email           = $email
        RETURN u {.*} AS profile
        """
        params = {
            "owner_user_id": owner_user_id,
            "workspace_id": workspace_id,
            "username": user.username,
            "email": user.email or "",
        }

        results, _ = db.cypher_query(query, params)

        db.cypher_query(
            """
            MATCH (p:Person {workspace_id: $workspace_id})
            WHERE toLower(trim(p.name)) IN $self_aliases
            DETACH DELETE p
            """,
            {
                "workspace_id": workspace_id,
                "self_aliases": build_self_alias_names(current_username=user.username),
            },
        )

        if results and results[0]:
            profile = results[0][0]
            logger.info(
                "UserProfile synced in Neo4j for user %s (id=%s, workspace=%s)",
                user.username,
                owner_user_id,
                workspace_id,
            )
            return profile

        logger.warning("UserProfile MERGE returned no rows for user %s", user.username)
        return None

    except Exception as exc:
        # Never block auth flow — graph sync is best-effort
        logger.warning(
            "Failed to sync UserProfile to Neo4j for user %s: %s",
            user.username,
            exc,
        )
        return None


def update_journey_state_in_graph(*, workspace_id: str, owner_user_id: int, updates: dict) -> dict | None:
    """
    Update journey state properties on a UserProfile node.

    Only updates the properties specified in `updates`, leaving everything
    else untouched. Returns the updated profile properties or None on failure.

    Allowed keys: journey_phase, path_id, philosopher_mode, sovereign_mode,
    sovereign_end_statement, soul_ordering, value_ordering,
    current_provocation_index, socratic_chariot_revealed, socratic_tier_revealed.
    """
    ALLOWED_JOURNEY_FIELDS = {
        "journey_phase", "path_id", "philosopher_mode", "sovereign_mode",
        "sovereign_end_statement", "soul_ordering", "value_ordering",
        "current_provocation_index", "socratic_chariot_revealed", "socratic_tier_revealed",
    }

    filtered = {k: v for k, v in updates.items() if k in ALLOWED_JOURNEY_FIELDS}
    if not filtered:
        return None

    try:
        from neomodel import db

        set_clauses = ", ".join(f"u.{key} = ${key}" for key in filtered)
        query = f"""
        MATCH (u:UserProfile {{workspace_id: $workspace_id, owner_user_id: $owner_user_id}})
        SET {set_clauses}
        RETURN u {{.*}} AS profile
        """
        params = {
            "workspace_id": workspace_id,
            "owner_user_id": owner_user_id,
            **filtered,
        }

        results, _ = db.cypher_query(query, params)
        if results and results[0]:
            return results[0][0]
        return None

    except Exception as exc:
        logger.warning(
            "Failed to update journey state in Neo4j for user %s in workspace %s: %s",
            owner_user_id, workspace_id, exc,
        )
        return None


def get_journey_state_from_graph(*, workspace_id: str, owner_user_id: int) -> dict | None:
    """
    Read journey state properties from a UserProfile node.

    Returns a dict with the journey-related fields, or None if the profile
    doesn't exist or on failure.
    """
    try:
        from neomodel import db

        query = """
        MATCH (u:UserProfile {workspace_id: $workspace_id, owner_user_id: $owner_user_id})
        RETURN u.journey_phase AS journey_phase,
               u.path_id AS path_id,
               u.philosopher_mode AS philosopher_mode,
               u.sovereign_mode AS sovereign_mode,
               u.sovereign_end_statement AS sovereign_end_statement,
               u.soul_ordering AS soul_ordering,
               u.value_ordering AS value_ordering,
               u.current_provocation_index AS current_provocation_index,
               u.socratic_chariot_revealed AS socratic_chariot_revealed,
               u.socratic_tier_revealed AS socratic_tier_revealed
        """
        results, columns = db.cypher_query(
            query,
            {"workspace_id": workspace_id, "owner_user_id": owner_user_id},
        )

        if not results or not results[0]:
            return None

        row = results[0]
        return dict(zip(columns, row))

    except Exception as exc:
        logger.warning(
            "Failed to read journey state from Neo4j for user %s in workspace %s: %s",
            owner_user_id, workspace_id, exc,
        )
        return None


def create_provocation_response_in_graph(
    *, workspace_id: str, owner_user_id: int, response_data: dict
) -> dict | None:
    """
    Create a ProvocationResponseNode, link it to the UserProfile, and
    (if the ontology is seeded) link it to the corresponding
    ProvocationOntologyNode and ProvocationChoiceNode.

    Returns the created node properties or None on failure.
    """
    try:
        from neomodel import db

        query = """
        MATCH (u:UserProfile {workspace_id: $workspace_id, owner_user_id: $owner_user_id})
        CREATE (r:ProvocationResponseNode {
            uid: randomUUID(),
            workspace_id: $workspace_id,
            owner_user_id: $owner_user_id,
            provocation_id: $provocation_id,
            choice_id: $choice_id,
            served_soul_part: $served_soul_part,
            protected_values: $protected_values,
            sacrificed_values: $sacrificed_values,
            deliberation_time_ms: $deliberation_time_ms,
            was_instinctive: $was_instinctive,
            timestamp: datetime()
        })
        CREATE (u)-[:HAS_PROVOCATION_RESPONSE]->(r)
        RETURN r {.*} AS response
        """
        params = {
            "workspace_id": workspace_id,
            "owner_user_id": owner_user_id,
            "provocation_id": response_data.get("provocationId", ""),
            "choice_id": response_data.get("choiceId", ""),
            "served_soul_part": response_data.get("servedSoulPart", ""),
            "protected_values": response_data.get("protectedValues", []),
            "sacrificed_values": response_data.get("sacrificedValues", []),
            "deliberation_time_ms": response_data.get("deliberationTimeMs", 0),
            "was_instinctive": response_data.get("wasInstinctive", False),
        }

        results, _ = db.cypher_query(query, params)
        if not (results and results[0]):
            return None

        response_node = results[0][0]

        # Best-effort: link to ontology nodes if they exist
        _link_response_to_ontology(
            db,
            provocation_id=params["provocation_id"],
            choice_id=params["choice_id"],
        )

        return response_node

    except Exception as exc:
        logger.warning(
            "Failed to create provocation response in Neo4j for user %s: %s",
            owner_user_id, exc,
        )
        return None


def _link_response_to_ontology(db, *, provocation_id: str, choice_id: str) -> None:
    """
    Best-effort: create TO_PROVOCATION and CHOSE edges from the most
    recent ProvocationResponseNode with matching IDs to the ontology nodes.
    Silently skips if ontology is not seeded.
    """
    try:
        db.cypher_query(
            """
            MATCH (r:ProvocationResponseNode {provocation_id: $prov_id, choice_id: $choice_id})
            WHERE NOT (r)-[:TO_PROVOCATION]->(:ProvocationOntologyNode)
            WITH r ORDER BY r.timestamp DESC LIMIT 1
            OPTIONAL MATCH (p:ProvocationOntologyNode {provocation_id: $prov_id})
            OPTIONAL MATCH (c:ProvocationChoiceNode {choice_id: $choice_id})
            FOREACH (_ IN CASE WHEN p IS NOT NULL THEN [1] ELSE [] END |
                MERGE (r)-[:TO_PROVOCATION]->(p)
            )
            FOREACH (_ IN CASE WHEN c IS NOT NULL THEN [1] ELSE [] END |
                MERGE (r)-[:CHOSE]->(c)
            )
            """,
            {"prov_id": provocation_id, "choice_id": choice_id},
        )
    except Exception:
        pass  # Ontology not seeded yet — that's fine


def backfill_provocation_ontology_links() -> dict:
    """
    Backfill TO_PROVOCATION and CHOSE relationships for all existing
    ProvocationResponseNode nodes that don't have them yet.

    Returns a summary dict with counts.
    """
    try:
        from neomodel import db

        # Link responses to provocations
        results_prov, _ = db.cypher_query(
            """
            MATCH (r:ProvocationResponseNode)
            WHERE NOT (r)-[:TO_PROVOCATION]->(:ProvocationOntologyNode)
            MATCH (p:ProvocationOntologyNode {provocation_id: r.provocation_id})
            MERGE (r)-[:TO_PROVOCATION]->(p)
            RETURN count(r) AS linked
            """
        )

        # Link responses to choices
        results_choice, _ = db.cypher_query(
            """
            MATCH (r:ProvocationResponseNode)
            WHERE NOT (r)-[:CHOSE]->(:ProvocationChoiceNode)
            MATCH (c:ProvocationChoiceNode {choice_id: r.choice_id})
            MERGE (r)-[:CHOSE]->(c)
            RETURN count(r) AS linked
            """
        )

        return {
            "provocation_links": results_prov[0][0] if results_prov else 0,
            "choice_links": results_choice[0][0] if results_choice else 0,
        }

    except Exception as exc:
        logger.warning("Failed to backfill provocation links: %s", exc)
        return {"provocation_links": 0, "choice_links": 0}


def list_provocation_responses_from_graph(
    *, workspace_id: str, owner_user_id: int
) -> list[dict]:
    """
    List all ProvocationResponseNode nodes for a user, ordered by timestamp.
    """
    try:
        from neomodel import db

        query = """
        MATCH (u:UserProfile {workspace_id: $workspace_id, owner_user_id: $owner_user_id})
              -[:HAS_PROVOCATION_RESPONSE]->(r:ProvocationResponseNode)
        RETURN r {.*} AS response
        ORDER BY r.timestamp ASC
        """
        results, _ = db.cypher_query(
            query,
            {"workspace_id": workspace_id, "owner_user_id": owner_user_id},
        )

        return [row[0] for row in results] if results else []

    except Exception as exc:
        logger.warning(
            "Failed to list provocation responses from Neo4j for user %s: %s",
            owner_user_id, exc,
        )
        return []


def link_user_to_conversations(user, workspace):
    """
    Back-link existing Conversation nodes to the UserProfile node.

    Finds any Conversation nodes owned by this user (via owner_user_id)
    that aren't yet connected and creates OWNS_CONVERSATION edges.
    """
    try:
        from neomodel import db

        query = """
        MATCH (u:UserProfile {workspace_id: $workspace_id, owner_user_id: $owner_user_id})
        MATCH (c:Conversation {workspace_id: $workspace_id, owner_user_id: $owner_user_id})
        WHERE NOT (u)-[:OWNS_CONVERSATION]->(c)
        CREATE (u)-[:OWNS_CONVERSATION]->(c)
        RETURN count(c) AS linked
        """
        results, _ = db.cypher_query(
            query, {"owner_user_id": user.id, "workspace_id": str(workspace.id)}
        )

        linked = results[0][0] if results and results[0] else 0
        if linked > 0:
            logger.info(
                "Linked %d existing conversations to UserProfile for user %s",
                linked,
                user.username,
            )
        return linked

    except Exception as exc:
        logger.warning(
            "Failed to link conversations for user %s: %s",
            user.username,
            exc,
        )
        return 0


def full_user_graph_sync(user, workspace):
    """
    Complete sync: ensure UserProfile exists + back-link all owned conversations.

    Call this on signup, login, or from the backfill management command.
    Signals are reachable via UserProfile → OWNS_CONVERSATION → Conversation → Signal.
    """
    profile = ensure_user_profile_in_graph(user, workspace)
    if profile is not None:
        link_user_to_conversations(user, workspace)
    return profile
