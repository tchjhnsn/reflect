"""
ThriveSight Insight Engine — Reasoning traces, replay, and validation.

The insight engine manages the reasoning layer of the graph:
- Insight nodes: Store LLM reasoning about signals and clusters
- Reflection nodes: Store user reasoning about insights
- Reasoning replay: Revisit previous reasoning without new API calls
- Validation workflow: Users validate, dispute, or ignore insights

Key principle: "All user feedback is signal"
- Validating an insight creates data about what resonates
- Disputing an insight creates data about user perspective
- Ignoring an insight creates data about engagement patterns

Architecture:
    LLM response → InsightManager.create_insight()
        → Insight node in graph
        → Linked to source signals/clusters
        → Tagged with persona that generated it

    User response → InsightManager.create_reflection()
        → Reflection node in graph
        → Linked to Insight it responds to
        → Optionally refines a Signal

    Replay → InsightManager.get_reasoning_trace()
        → Returns ordered Insight + Reflection chain
        → No LLM call needed
"""

import logging
from datetime import datetime
from typing import Any, Optional

from neomodel import db

from .graph_models import Cluster, Conversation, Insight, Reflection, Signal

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# VALIDATION STATUS VALUES
# ──────────────────────────────────────────────────────────────────────────────

STATUS_PENDING = "pending"
STATUS_VALIDATED = "validated"
STATUS_DISPUTED = "disputed"
STATUS_IGNORED = "ignored"

VALID_STATUSES = {STATUS_PENDING, STATUS_VALIDATED, STATUS_DISPUTED, STATUS_IGNORED}


class InsightManager:
    """
    Manages Insight and Reflection nodes — the reasoning trace layer.

    Insights store LLM reasoning about emotional patterns. Reflections
    store user reasoning about those insights. Together, they create
    a navigable trace that can be replayed without API calls.

    Usage:
        manager = InsightManager()

        # After LLM response:
        insight = manager.create_insight(
            reasoning_text="The pattern of frustration when your manager...",
            signal_uids=["sig-001", "sig-002"],
            cluster_ids=["CLU-abc123"],
            persona="gentle_explorer",
            conversation_id="conv-123",
        )

        # When user responds:
        reflection = manager.create_reflection(
            user_text="That makes sense, I hadn't thought about it that way",
            insight_uid=insight.uid,
        )

        # Replay previous reasoning:
        trace = manager.get_reasoning_trace(signal_uid="sig-001")
    """

    def create_insight(
        self,
        reasoning_text: str,
        signal_uids: Optional[list[str]] = None,
        cluster_ids: Optional[list[str]] = None,
        persona: str = "neutral_observer",
        conversation_id: Optional[str] = None,
        confidence: float = 0.7,
    ) -> Insight:
        """
        Create an Insight node capturing LLM reasoning.

        Called automatically after every Mode 2 LLM response that
        contains reasoning about signals or patterns.

        Args:
            reasoning_text: The LLM's reasoning about signals/patterns
            signal_uids: UIDs of signals this insight references
            cluster_ids: IDs of clusters this insight references
            persona: Which persona generated this insight
            conversation_id: Conversation context
            confidence: How confident the reasoning is

        Returns:
            Created Insight node
        """
        insight = Insight(
            reasoning_text=reasoning_text,
            persona=persona,
            confidence=confidence,
            validation_status=STATUS_PENDING,
        )
        insight.save()

        # Link to source signals
        if signal_uids:
            for uid in signal_uids:
                try:
                    db.cypher_query(
                        """
                        MATCH (i:Insight {uid: $insight_uid})
                        MATCH (s:Signal {uid: $signal_uid})
                        MERGE (i)-[:INFORMED_BY {relevance: 1.0}]->(s)
                        """,
                        {"insight_uid": insight.uid, "signal_uid": uid},
                    )
                except Exception as e:
                    logger.debug(f"Could not link insight to signal {uid}: {e}")

        # Link to source clusters
        if cluster_ids:
            for cid in cluster_ids:
                try:
                    db.cypher_query(
                        """
                        MATCH (i:Insight {uid: $insight_uid})
                        MATCH (c:Cluster {cluster_id: $cluster_id})
                        MERGE (i)-[:INFORMED_BY {relevance: 1.0}]->(c)
                        """,
                        {"insight_uid": insight.uid, "cluster_id": cid},
                    )
                except Exception as e:
                    logger.debug(f"Could not link insight to cluster {cid}: {e}")

        # Link to conversation
        if conversation_id:
            try:
                db.cypher_query(
                    """
                    MATCH (i:Insight {uid: $insight_uid})
                    MATCH (c:Conversation {conversation_id: $conv_id})
                    MERGE (i)-[:GENERATED_DURING]->(c)
                    """,
                    {"insight_uid": insight.uid, "conv_id": conversation_id},
                )
            except Exception as e:
                logger.debug(f"Could not link insight to conversation: {e}")

        logger.info(
            f"Created insight {insight.uid} (persona={persona}, "
            f"signals={len(signal_uids or [])}, clusters={len(cluster_ids or [])})"
        )

        return insight

    def create_reflection(
        self,
        user_text: str,
        insight_uid: Optional[str] = None,
        signal_uid: Optional[str] = None,
        reflection_type: str = "response",
    ) -> Reflection:
        """
        Create a Reflection node capturing user reasoning.

        All user feedback is signal. Reflections capture:
        - Responses to AI insights
        - Disputes of patterns or clusters
        - Self-corrections ("actually, I think...")
        - Validations ("yes, that resonates")

        Args:
            user_text: The user's reasoning text
            insight_uid: UID of insight being responded to (if any)
            signal_uid: UID of signal being refined (if any)
            reflection_type: "response" | "dispute" | "validation" | "self_correction" | "cluster_dispute" | "cluster_validation"

        Returns:
            Created Reflection node
        """
        reflection = Reflection(
            text=user_text,
            reflection_type=reflection_type,
        )
        reflection.save()

        # Link to insight
        if insight_uid:
            try:
                db.cypher_query(
                    """
                    MATCH (r:Reflection {uid: $refl_uid})
                    MATCH (i:Insight {uid: $insight_uid})
                    MERGE (r)-[:RESPONDS_TO]->(i)
                    """,
                    {"refl_uid": reflection.uid, "insight_uid": insight_uid},
                )
            except Exception as e:
                logger.debug(f"Could not link reflection to insight: {e}")

        # Link to signal (refinement)
        if signal_uid:
            try:
                db.cypher_query(
                    """
                    MATCH (r:Reflection {uid: $refl_uid})
                    MATCH (s:Signal {uid: $signal_uid})
                    MERGE (r)-[:REFINES]->(s)
                    """,
                    {"refl_uid": reflection.uid, "signal_uid": signal_uid},
                )
            except Exception as e:
                logger.debug(f"Could not link reflection to signal: {e}")

        logger.info(
            f"Created reflection {reflection.uid} (type={reflection_type}, "
            f"insight={insight_uid}, signal={signal_uid})"
        )

        return reflection

    def get_reasoning_trace(
        self,
        signal_uid: Optional[str] = None,
        cluster_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        limit: int = 20,
    ) -> list[dict]:
        """
        Get the full reasoning trace for a signal, cluster, or conversation.

        Returns an ordered list of Insight + Reflection pairs, allowing
        the user to replay previous reasoning without API calls.

        Args:
            signal_uid: Filter by signal UID
            cluster_id: Filter by cluster ID
            conversation_id: Filter by conversation ID
            limit: Maximum number of trace entries

        Returns:
            Ordered list of dicts with insight and optional reflection data
        """
        if signal_uid:
            return self._trace_for_signal(signal_uid, limit)
        elif cluster_id:
            return self._trace_for_cluster(cluster_id, limit)
        elif conversation_id:
            return self._trace_for_conversation(conversation_id, limit)
        else:
            return []

    def validate_insight(
        self,
        insight_uid: str,
        user_feedback: str = "",
        new_status: str = STATUS_VALIDATED,
    ) -> dict:
        """
        Update an insight's validation status based on user feedback.

        Args:
            insight_uid: UID of the insight
            user_feedback: Optional user text
            new_status: New validation status

        Returns:
            Dict with update summary
        """
        if new_status not in VALID_STATUSES:
            raise ValueError(f"Invalid status: {new_status}. Valid: {VALID_STATUSES}")

        try:
            insight = Insight.nodes.get(uid=insight_uid)
        except Exception:
            raise ValueError(f"Insight not found: {insight_uid}")

        old_status = insight.validation_status
        insight.validation_status = new_status
        insight.save()

        # Create reflection if user provided feedback
        reflection = None
        if user_feedback:
            reflection_type = "validation" if new_status == STATUS_VALIDATED else "dispute"
            reflection = self.create_reflection(
                user_text=user_feedback,
                insight_uid=insight_uid,
                reflection_type=reflection_type,
            )

        logger.info(
            f"Insight {insight_uid} status: {old_status} → {new_status}"
        )

        return {
            "insight_uid": insight_uid,
            "old_status": old_status,
            "new_status": new_status,
            "reflection_uid": reflection.uid if reflection else None,
        }

    def replay_exploration(self, signal_uid: str) -> dict:
        """
        Replay the full exploration history for a signal.

        Returns everything the user needs to revisit their exploration
        without making new API calls:
        - The signal itself
        - All derived child signals
        - All insights about this signal
        - All user reflections
        - The emotional trajectory

        Args:
            signal_uid: UID of the signal to replay

        Returns:
            Dict with full exploration history
        """
        # Get the signal
        results, _ = db.cypher_query(
            """
            MATCH (s:Signal {uid: $uid})
            RETURN s.uid, s.signal_address, s.emotions, s.confidence_score,
                   s.observation_bias_flags, s.exploration_geometry,
                   s.is_resolved, s.wildcard_coordinates
            """,
            {"uid": signal_uid},
        )

        if not results:
            return {"error": f"Signal not found: {signal_uid}"}

        row = results[0]
        signal_data = {
            "uid": row[0],
            "signal_address": row[1],
            "emotions": row[2],
            "confidence_score": row[3],
            "bias_flags": row[4],
            "geometry": row[5],
            "is_resolved": row[6],
            "wildcards": row[7],
        }

        # Get derived signals
        derived_results, _ = db.cypher_query(
            """
            MATCH (parent:Signal {uid: $uid})<-[:DERIVED_FROM]-(child:Signal)
            RETURN child.uid, child.signal_address, child.emotions,
                   child.confidence_score, child.is_resolved
            ORDER BY child.created_at
            """,
            {"uid": signal_uid},
        )

        derived = [
            {
                "uid": r[0],
                "signal_address": r[1],
                "emotions": r[2],
                "confidence_score": r[3],
                "is_resolved": r[4],
            }
            for r in derived_results
        ]

        # Get reasoning trace
        trace = self._trace_for_signal(signal_uid, limit=50)

        return {
            "signal": signal_data,
            "derived_signals": derived,
            "reasoning_trace": trace,
            "exploration_depth": len(derived),
            "total_insights": len([t for t in trace if t.get("type") == "insight"]),
            "total_reflections": len([t for t in trace if t.get("type") == "reflection"]),
        }

    # ──────────────────────────────────────────────────────────────────────
    # INTERNAL: Trace Queries
    # ──────────────────────────────────────────────────────────────────────

    def _trace_for_signal(self, signal_uid: str, limit: int) -> list[dict]:
        """Get reasoning trace for a specific signal."""
        results, _ = db.cypher_query(
            """
            MATCH (s:Signal {uid: $uid})<-[:INFORMED_BY]-(i:Insight)
            OPTIONAL MATCH (r:Reflection)-[:RESPONDS_TO]->(i)
            RETURN i.uid, i.reasoning_text, i.persona, i.confidence,
                   i.validation_status, i.generated_at,
                   r.uid, r.text, r.reflection_type, r.created_at
            ORDER BY i.generated_at
            LIMIT $limit
            """,
            {"uid": signal_uid, "limit": limit},
        )

        return self._format_trace_results(results)

    def _trace_for_cluster(self, cluster_id: str, limit: int) -> list[dict]:
        """Get reasoning trace for a specific cluster."""
        results, _ = db.cypher_query(
            """
            MATCH (c:Cluster {cluster_id: $cluster_id})<-[:INFORMED_BY]-(i:Insight)
            OPTIONAL MATCH (r:Reflection)-[:RESPONDS_TO]->(i)
            RETURN i.uid, i.reasoning_text, i.persona, i.confidence,
                   i.validation_status, i.generated_at,
                   r.uid, r.text, r.reflection_type, r.created_at
            ORDER BY i.generated_at
            LIMIT $limit
            """,
            {"cluster_id": cluster_id, "limit": limit},
        )

        return self._format_trace_results(results)

    def _trace_for_conversation(self, conversation_id: str, limit: int) -> list[dict]:
        """Get reasoning trace for a specific conversation."""
        results, _ = db.cypher_query(
            """
            MATCH (conv:Conversation {conversation_id: $conv_id})<-[:GENERATED_DURING]-(i:Insight)
            OPTIONAL MATCH (r:Reflection)-[:RESPONDS_TO]->(i)
            RETURN i.uid, i.reasoning_text, i.persona, i.confidence,
                   i.validation_status, i.generated_at,
                   r.uid, r.text, r.reflection_type, r.created_at
            ORDER BY i.generated_at
            LIMIT $limit
            """,
            {"conv_id": conversation_id, "limit": limit},
        )

        return self._format_trace_results(results)

    def _format_trace_results(self, results) -> list[dict]:
        """Format Cypher results into a trace list."""
        trace = []

        for row in results:
            entry = {
                "type": "insight",
                "insight_uid": row[0],
                "reasoning_text": row[1],
                "persona": row[2],
                "confidence": row[3],
                "validation_status": row[4],
                "generated_at": str(row[5]) if row[5] else None,
            }
            trace.append(entry)

            # Add reflection if present
            if row[6]:  # reflection uid exists
                trace.append({
                    "type": "reflection",
                    "reflection_uid": row[6],
                    "text": row[7],
                    "reflection_type": row[8],
                    "created_at": str(row[9]) if row[9] else None,
                    "responds_to_insight": row[0],
                })

        return trace


# ──────────────────────────────────────────────────────────────────────────────
# CONVENIENCE FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────


def create_insight_from_response(
    reasoning_text: str,
    signal_uids: Optional[list[str]] = None,
    cluster_ids: Optional[list[str]] = None,
    persona: str = "neutral_observer",
    conversation_id: Optional[str] = None,
) -> Insight:
    """Convenience function for creating an insight after LLM response."""
    manager = InsightManager()
    return manager.create_insight(
        reasoning_text=reasoning_text,
        signal_uids=signal_uids,
        cluster_ids=cluster_ids,
        persona=persona,
        conversation_id=conversation_id,
    )


def create_user_reflection(
    user_text: str,
    insight_uid: Optional[str] = None,
    signal_uid: Optional[str] = None,
    reflection_type: str = "response",
) -> Reflection:
    """Convenience function for creating a user reflection."""
    manager = InsightManager()
    return manager.create_reflection(
        user_text=user_text,
        insight_uid=insight_uid,
        signal_uid=signal_uid,
        reflection_type=reflection_type,
    )


def replay_signal_exploration(signal_uid: str) -> dict:
    """Convenience function for replaying a signal's exploration history."""
    manager = InsightManager()
    return manager.replay_exploration(signal_uid)
