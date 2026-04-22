"""
ThriveSight Insight Engine — Reasoning Trace Construction and Storage.

Manages the creation and lifecycle of Insight and Reflection nodes in the
graph. Insights are LLM-generated reasoning traces; Reflections are
user-authored responses to those traces.

Key operations:
- create_insight: Generate and persist an Insight node
- create_reflection: Persist a user Reflection node
- validate_insight: Record user feedback on an insight
- get_reasoning_trace: Reconstruct the insight/reflection chain
"""

import json
import logging
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Insight Creation
# ──────────────────────────────────────────────────────────────────────────────

class InsightEngine:
    """
    Creates and manages reasoning traces in the graph.

    Insights are LLM observations stored as graph nodes. They reference
    the signals and clusters that informed them, forming a navigable
    reasoning chain that can be replayed without additional API calls.
    """

    def __init__(self, workspace_id: Optional[str] = None,
                 owner_user_id: Optional[int] = None):
        self.workspace_id = workspace_id
        self.owner_user_id = owner_user_id

    def create_insight(
        self,
        reasoning_text: str,
        persona: str = "neutral_observer",
        confidence: float = 0.5,
        signal_uids: Optional[list] = None,
        cluster_uids: Optional[list] = None,
        conversation_id: Optional[str] = None,
    ) -> Optional[str]:
        """
        Create an Insight node in the graph.

        Args:
            reasoning_text: The LLM's reasoning/observation text.
            persona: Which AI persona generated this insight.
            confidence: LLM's self-assessed confidence (0.0-1.0).
            signal_uids: UIDs of signals this insight draws upon.
            cluster_uids: UIDs of clusters this insight draws upon.
            conversation_id: Optional conversation context.

        Returns:
            The UID of the created Insight, or None on failure.
        """
        try:
            from neomodel import db

            result, _ = db.cypher_query(
                """
                CREATE (i:Insight {
                    uid: randomUUID(),
                    workspace_id: $workspace_id,
                    owner_user_id: $owner_user_id,
                    reasoning_text: $text,
                    persona: $persona,
                    confidence: $confidence,
                    validation_status: 'pending',
                    generated_at: datetime()
                })
                RETURN i.uid
                """,
                {
                    "workspace_id": self.workspace_id or "",
                    "owner_user_id": self.owner_user_id or 0,
                    "text": reasoning_text,
                    "persona": persona,
                    "confidence": confidence,
                },
            )

            if not result:
                return None

            insight_uid = result[0][0]

            # Link to source signals
            for sig_uid in (signal_uids or []):
                db.cypher_query(
                    """
                    MATCH (i:Insight {uid: $insight_uid})
                    MATCH (s:Signal {uid: $sig_uid})
                    MERGE (i)-[:INFORMED_BY {relevance: 1.0}]->(s)
                    """,
                    {"insight_uid": insight_uid, "sig_uid": sig_uid},
                )

            # Link to source clusters
            for clu_uid in (cluster_uids or []):
                db.cypher_query(
                    """
                    MATCH (i:Insight {uid: $insight_uid})
                    MATCH (c:Cluster {uid: $clu_uid})
                    MERGE (i)-[:INFORMED_BY]->(c)
                    """,
                    {"insight_uid": insight_uid, "clu_uid": clu_uid},
                )

            # Link to conversation if provided
            if conversation_id:
                db.cypher_query(
                    """
                    MATCH (i:Insight {uid: $insight_uid})
                    MATCH (c:Conversation {conversation_id: $conv_id})
                    MERGE (i)-[:GENERATED_DURING]->(c)
                    """,
                    {"insight_uid": insight_uid, "conv_id": conversation_id},
                )

            logger.info(f"Created insight {insight_uid} (persona={persona})")
            return insight_uid

        except Exception as e:
            logger.error(f"Insight creation failed: {e}")
            return None

    def create_insight_from_llm(
        self,
        context_packet: str,
        signal_uids: Optional[list] = None,
        cluster_uids: Optional[list] = None,
        persona: str = "neutral_observer",
        conversation_id: Optional[str] = None,
    ) -> Optional[str]:
        """
        Generate an insight using the LLM, then persist it.

        Args:
            context_packet: Assembled context for the LLM.
            signal_uids: Source signal UIDs.
            cluster_uids: Source cluster UIDs.
            persona: Persona to use for generation.
            conversation_id: Optional conversation context.

        Returns:
            The UID of the created Insight, or None on failure.
        """
        try:
            from events_api.llm_prompts import get_prompt, build_system_prompt
            from events_api.persona_config import get_persona
            from events_api.llm_client import analyze_with_retry

            persona_config = get_persona(persona)
            base_prompt = get_prompt("cluster_reasoning")
            system_prompt = build_system_prompt(
                base_prompt,
                persona_modifier=persona_config.system_prompt_modifier,
                context_packet=context_packet,
            )

            user_prompt = (
                "Based on the context provided, generate an observation or insight "
                "about the emotional patterns you see. Return JSON: "
                '{"reasoning_text": "...", "confidence": 0.0-1.0}'
            )

            result = analyze_with_retry(system_prompt, user_prompt)

            reasoning_text = result.get("reasoning_text", "")
            confidence = float(result.get("confidence", 0.5))

            if not reasoning_text:
                logger.warning("LLM returned empty reasoning text")
                return None

            return self.create_insight(
                reasoning_text=reasoning_text,
                persona=persona,
                confidence=confidence,
                signal_uids=signal_uids,
                cluster_uids=cluster_uids,
                conversation_id=conversation_id,
            )

        except Exception as e:
            logger.error(f"LLM insight generation failed: {e}")
            return None

    def validate_insight(
        self,
        insight_uid: str,
        validation_status: str,
        feedback: Optional[str] = None,
    ) -> bool:
        """
        Record user feedback on an insight.

        Args:
            insight_uid: The UID of the insight to validate.
            validation_status: "validated", "disputed", or "ignored".
            feedback: Optional text explaining the user's reasoning.

        Returns:
            True if update succeeded.
        """
        valid_statuses = {"validated", "disputed", "ignored"}
        if validation_status not in valid_statuses:
            logger.warning(f"Invalid validation status: {validation_status}")
            return False

        try:
            from neomodel import db

            now = datetime.now(timezone.utc).isoformat()

            db.cypher_query(
                """
                MATCH (i:Insight {uid: $uid})
                SET i.validation_status = $status,
                    i.validated_at = datetime($now),
                    i.validation_feedback = $feedback
                """,
                {
                    "uid": insight_uid,
                    "status": validation_status,
                    "now": now,
                    "feedback": feedback or "",
                },
            )

            logger.info(f"Insight {insight_uid} validated as {validation_status}")
            return True

        except Exception as e:
            logger.error(f"Insight validation failed: {e}")
            return False

    def get_reasoning_trace(
        self,
        insight_uid: Optional[str] = None,
        signal_uid: Optional[str] = None,
        limit: int = 20,
    ) -> list:
        """
        Reconstruct the insight/reflection reasoning chain.

        Starting from an insight or signal, traverse the chain of
        Insight ↔ Reflection nodes to build the full trace.

        Args:
            insight_uid: Start from this insight.
            signal_uid: Start from insights linked to this signal.
            limit: Maximum trace entries.

        Returns:
            List of trace entries [{type, uid, text, persona, timestamp, ...}]
        """
        trace = []

        try:
            from neomodel import db

            if insight_uid:
                # Get this insight and its reflections
                results, _ = db.cypher_query(
                    """
                    MATCH (i:Insight {uid: $uid})
                    OPTIONAL MATCH (r:Reflection)-[:RESPONDS_TO]->(i)
                    RETURN i.uid, i.reasoning_text, i.persona, i.confidence,
                           i.validation_status, i.generated_at,
                           collect({uid: r.uid, text: r.text,
                                    type: r.reflection_type,
                                    created_at: r.created_at}) AS reflections
                    """,
                    {"uid": insight_uid},
                )

                for row in results:
                    trace.append({
                        "type": "insight",
                        "uid": row[0],
                        "text": row[1],
                        "persona": row[2],
                        "confidence": row[3],
                        "validation_status": row[4],
                        "timestamp": str(row[5]) if row[5] else None,
                    })
                    for ref in (row[6] or []):
                        if ref.get("uid"):
                            trace.append({
                                "type": "reflection",
                                "uid": ref["uid"],
                                "text": ref["text"],
                                "reflection_type": ref["type"],
                                "timestamp": str(ref["created_at"]) if ref.get("created_at") else None,
                            })

            elif signal_uid:
                # Find all insights linked to this signal
                results, _ = db.cypher_query(
                    """
                    MATCH (i:Insight)-[:INFORMED_BY]->(s:Signal {uid: $uid})
                    OPTIONAL MATCH (r:Reflection)-[:RESPONDS_TO]->(i)
                    RETURN i.uid, i.reasoning_text, i.persona, i.confidence,
                           i.validation_status, i.generated_at,
                           collect({uid: r.uid, text: r.text,
                                    type: r.reflection_type}) AS reflections
                    ORDER BY i.generated_at
                    LIMIT $limit
                    """,
                    {"uid": signal_uid, "limit": limit},
                )

                for row in results:
                    trace.append({
                        "type": "insight",
                        "uid": row[0],
                        "text": row[1],
                        "persona": row[2],
                        "confidence": row[3],
                        "validation_status": row[4],
                        "timestamp": str(row[5]) if row[5] else None,
                    })
                    for ref in (row[6] or []):
                        if ref.get("uid"):
                            trace.append({
                                "type": "reflection",
                                "uid": ref["uid"],
                                "text": ref["text"],
                                "reflection_type": ref["type"],
                            })

        except Exception as e:
            logger.error(f"Reasoning trace retrieval failed: {e}")

        return trace[:limit]


# ──────────────────────────────────────────────────────────────────────────────
# Reflection Creation
# ──────────────────────────────────────────────────────────────────────────────

def create_reflection(
    text: str,
    reflection_type: str = "elaboration",
    insight_uid: Optional[str] = None,
    signal_uid: Optional[str] = None,
    workspace_id: Optional[str] = None,
    owner_user_id: Optional[int] = None,
) -> Optional[str]:
    """
    Create a Reflection node — user-authored reasoning trace.

    Args:
        text: The user's reflection text.
        reflection_type: "agreement", "dispute", "elaboration", "realization", "question".
        insight_uid: UID of insight this responds to.
        signal_uid: UID of signal this refines.
        workspace_id: Workspace scoping.
        owner_user_id: Owner user ID.

    Returns:
        The UID of the created Reflection, or None on failure.
    """
    valid_types = {"agreement", "dispute", "elaboration", "realization", "question"}
    if reflection_type not in valid_types:
        reflection_type = "elaboration"

    try:
        from neomodel import db

        result, _ = db.cypher_query(
            """
            CREATE (r:Reflection {
                uid: randomUUID(),
                workspace_id: $ws,
                owner_user_id: $owner,
                text: $text,
                reflection_type: $rtype,
                created_at: datetime()
            })
            RETURN r.uid
            """,
            {
                "ws": workspace_id or "",
                "owner": owner_user_id or 0,
                "text": text,
                "rtype": reflection_type,
            },
        )

        if not result:
            return None

        ref_uid = result[0][0]

        # Link to insight
        if insight_uid:
            db.cypher_query(
                """
                MATCH (r:Reflection {uid: $ref_uid})
                MATCH (i:Insight {uid: $insight_uid})
                MERGE (r)-[:RESPONDS_TO]->(i)
                """,
                {"ref_uid": ref_uid, "insight_uid": insight_uid},
            )

        # Link to signal
        if signal_uid:
            db.cypher_query(
                """
                MATCH (r:Reflection {uid: $ref_uid})
                MATCH (s:Signal {uid: $sig_uid})
                MERGE (r)-[:REFINES]->(s)
                """,
                {"ref_uid": ref_uid, "sig_uid": signal_uid},
            )

        logger.info(f"Created reflection {ref_uid} (type={reflection_type})")
        return ref_uid

    except Exception as e:
        logger.error(f"Reflection creation failed: {e}")
        return None


def create_insight(
    reasoning_text: str,
    workspace_id: Optional[str] = None,
    owner_user_id: Optional[int] = None,
    **kwargs,
) -> Optional[str]:
    """Convenience function for insight creation."""
    engine = InsightEngine(workspace_id=workspace_id, owner_user_id=owner_user_id)
    return engine.create_insight(reasoning_text=reasoning_text, **kwargs)
