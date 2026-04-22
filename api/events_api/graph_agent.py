"""
ThriveSight Graph Agent — Background Detection Cycle.

The graph agent is an autonomous background process that maintains the
knowledge graph by:

1. Detecting new clusters from recent signals
2. Updating cluster trajectories (strength, status, lifecycle)
3. Generating PendingInsight nodes for significant detections
4. Pruning stale pending insights past their expiration
5. Identifying trajectory shifts (strengthening, weakening, dissolution)

The agent runs periodically (triggered by a management command or
scheduled task) and writes results to the graph for the context
assembly layer to pick up during the next user conversation.
"""

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# Agent configuration
DEFAULT_SIGNAL_LOOKBACK_DAYS = 7
DEFAULT_INSIGHT_EXPIRY_DAYS = 14
MINIMUM_CLUSTER_STRENGTH_FOR_INSIGHT = 0.3
NEW_CLUSTER_CONFIDENCE = 0.5
TRAJECTORY_SHIFT_CONFIDENCE = 0.6
CLUSTER_STRENGTHENED_CONFIDENCE = 0.55


# ──────────────────────────────────────────────────────────────────────────────
# GraphAgent
# ──────────────────────────────────────────────────────────────────────────────

class GraphAgent:
    """
    Autonomous background agent for graph maintenance and detection.

    The agent does not interact with the user directly. It writes
    PendingInsight nodes that the context assembly layer picks up
    and includes in the LLM context during the next conversation.
    """

    def __init__(
        self,
        workspace_id: Optional[str] = None,
        lookback_days: int = DEFAULT_SIGNAL_LOOKBACK_DAYS,
        insight_expiry_days: int = DEFAULT_INSIGHT_EXPIRY_DAYS,
    ):
        self.workspace_id = workspace_id
        self.lookback_days = lookback_days
        self.insight_expiry_days = insight_expiry_days

    def run_detection_cycle(self) -> dict:
        """
        Execute a full background detection cycle.

        Steps:
        1. Find recent unprocessed signals
        2. Run cluster detection for each
        3. Update all cluster trajectories
        4. Generate pending insights for significant events
        5. Prune expired pending insights

        Returns:
            Summary dict with counts and details.
        """
        summary = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "signals_processed": 0,
            "clusters_created": 0,
            "clusters_updated": 0,
            "insights_generated": 0,
            "insights_pruned": 0,
            "trajectory_shifts": [],
            "errors": [],
        }

        # Step 1: Process recent signals for cluster detection
        try:
            recent_signals = self._get_recent_unprocessed_signals()
            summary["signals_processed"] = len(recent_signals)

            from events_api.cluster_engine import ClusterEngine
            engine = ClusterEngine(workspace_id=self.workspace_id)

            for sig in recent_signals:
                try:
                    actions = engine.detect_clusters_for_signal(sig)
                    results = engine.execute_cluster_actions(actions)

                    for action in actions:
                        if action["action"] == "create":
                            summary["clusters_created"] += 1

                            # Generate pending insight for new cluster
                            self._create_pending_insight(
                                detection_type="new_cluster",
                                description=self._describe_new_cluster(action),
                                confidence=NEW_CLUSTER_CONFIDENCE,
                                cluster_id=action.get("cluster_type", ""),
                            )
                            summary["insights_generated"] += 1

                except Exception as e:
                    summary["errors"].append(f"Signal {sig.get('uid', '?')}: {str(e)}")

        except Exception as e:
            summary["errors"].append(f"Signal processing: {str(e)}")

        # Step 2: Update cluster trajectories
        try:
            from events_api.cluster_engine import ClusterEngine
            engine = ClusterEngine(workspace_id=self.workspace_id)
            updates = engine.update_cluster_trajectories()
            summary["clusters_updated"] = len(updates)

            # Generate insights for significant trajectory shifts
            for update in updates:
                old = update.get("old_status", "")
                new = update.get("new_status", "")
                strength = update.get("strength", 0)

                if old != new:
                    summary["trajectory_shifts"].append(update)

                    # Determine detection type
                    if new == "dissolved":
                        dtype = "cluster_dissolving"
                        conf = TRAJECTORY_SHIFT_CONFIDENCE
                    elif old == "weakening" and new == "active":
                        dtype = "cluster_strengthened"
                        conf = CLUSTER_STRENGTHENED_CONFIDENCE
                    elif new == "weakening":
                        dtype = "trajectory_shift"
                        conf = TRAJECTORY_SHIFT_CONFIDENCE
                    elif old == "dissolved" and new == "active":
                        dtype = "cluster_strengthened"
                        conf = CLUSTER_STRENGTHENED_CONFIDENCE
                    else:
                        dtype = "trajectory_shift"
                        conf = TRAJECTORY_SHIFT_CONFIDENCE

                    self._create_pending_insight(
                        detection_type=dtype,
                        description=self._describe_trajectory_shift(update),
                        confidence=conf,
                        cluster_id=update.get("cluster_id", ""),
                    )
                    summary["insights_generated"] += 1

        except Exception as e:
            summary["errors"].append(f"Trajectory update: {str(e)}")

        # Step 3: Prune expired pending insights
        try:
            pruned = self._prune_expired_insights()
            summary["insights_pruned"] = pruned
        except Exception as e:
            summary["errors"].append(f"Insight pruning: {str(e)}")

        logger.info(
            f"Detection cycle complete: "
            f"{summary['signals_processed']} signals, "
            f"{summary['clusters_created']} new clusters, "
            f"{summary['clusters_updated']} updated, "
            f"{summary['insights_generated']} insights, "
            f"{summary['insights_pruned']} pruned"
        )

        return summary

    def _get_recent_unprocessed_signals(self) -> list:
        """Get signals from the lookback window that haven't been clustered."""
        try:
            from neomodel import db

            ws_filter = "AND s.workspace_id = $ws" if self.workspace_id else ""
            cutoff = (
                datetime.now(timezone.utc) - timedelta(days=self.lookback_days)
            ).isoformat()

            results, _ = db.cypher_query(
                f"""
                MATCH (s:Signal)
                WHERE s.created_at >= datetime($cutoff)
                  AND NOT (s)-[:MEMBER_OF]->(:Cluster)
                  {ws_filter}
                RETURN s.uid, s.signal_address, s.emotions, s.confidence_score
                ORDER BY s.created_at DESC
                LIMIT 100
                """,
                {
                    "cutoff": cutoff,
                    "ws": self.workspace_id,
                },
            )

            return [
                {
                    "uid": r[0],
                    "signal_address": r[1],
                    "emotions": r[2],
                    "confidence_score": r[3],
                }
                for r in results
            ]

        except Exception as e:
            logger.warning(f"Failed to fetch recent signals: {e}")
            return []

    def _create_pending_insight(
        self,
        detection_type: str,
        description: str,
        confidence: float,
        cluster_id: str = "",
    ) -> Optional[str]:
        """Create a PendingInsight node for context assembly to surface."""
        try:
            from neomodel import db

            expires = (
                datetime.now(timezone.utc) + timedelta(days=self.insight_expiry_days)
            ).isoformat()

            result, _ = db.cypher_query(
                """
                CREATE (pi:PendingInsight {
                    uid: randomUUID(),
                    workspace_id: $ws,
                    owner_user_id: $owner,
                    detection_type: $dtype,
                    description: $desc,
                    confidence: $conf,
                    status: 'pending',
                    created_at: datetime(),
                    expires_at: datetime($expires)
                })
                RETURN pi.uid
                """,
                {
                    "ws": self.workspace_id or "",
                    "owner": 0,
                    "dtype": detection_type,
                    "desc": description,
                    "conf": confidence,
                    "expires": expires,
                },
            )

            if not result:
                return None

            pi_uid = result[0][0]

            # Link to cluster if available
            if cluster_id:
                db.cypher_query(
                    """
                    MATCH (pi:PendingInsight {uid: $pi_uid})
                    OPTIONAL MATCH (c:Cluster {cluster_id: $cid})
                    WITH pi, c WHERE c IS NOT NULL
                    MERGE (pi)-[:AWAITING_REVIEW]->(c)
                    """,
                    {"pi_uid": pi_uid, "cid": cluster_id},
                )

            logger.info(f"Created pending insight: {detection_type} ({pi_uid})")
            return pi_uid

        except Exception as e:
            logger.warning(f"Pending insight creation failed: {e}")
            return None

    def _prune_expired_insights(self) -> int:
        """Mark expired pending insights as expired."""
        try:
            from neomodel import db

            ws_filter = "AND pi.workspace_id = $ws" if self.workspace_id else ""
            now = datetime.now(timezone.utc).isoformat()

            result, _ = db.cypher_query(
                f"""
                MATCH (pi:PendingInsight)
                WHERE pi.status = 'pending'
                  AND pi.expires_at IS NOT NULL
                  AND pi.expires_at < datetime($now)
                  {ws_filter}
                SET pi.status = 'expired'
                RETURN count(pi)
                """,
                {"now": now, "ws": self.workspace_id},
            )

            count = result[0][0] if result else 0
            if count > 0:
                logger.info(f"Pruned {count} expired pending insights")
            return count

        except Exception as e:
            logger.warning(f"Insight pruning failed: {e}")
            return 0

    def get_pending_insights(self, limit: int = 10) -> list:
        """
        Retrieve pending insights for context assembly.

        Args:
            limit: Maximum insights to return.

        Returns:
            List of pending insight dicts.
        """
        try:
            from neomodel import db

            ws_filter = "AND pi.workspace_id = $ws" if self.workspace_id else ""

            results, _ = db.cypher_query(
                f"""
                MATCH (pi:PendingInsight)
                WHERE pi.status = 'pending'
                  {ws_filter}
                OPTIONAL MATCH (pi)-[:AWAITING_REVIEW]->(c:Cluster)
                RETURN pi.uid, pi.detection_type, pi.description,
                       pi.confidence, pi.created_at,
                       c.cluster_id, c.cluster_type
                ORDER BY pi.confidence DESC, pi.created_at DESC
                LIMIT $limit
                """,
                {"ws": self.workspace_id, "limit": limit},
            )

            return [
                {
                    "uid": r[0],
                    "detection_type": r[1],
                    "description": r[2],
                    "confidence": r[3],
                    "created_at": str(r[4]) if r[4] else None,
                    "cluster_id": r[5],
                    "cluster_type": r[6],
                }
                for r in results
            ]

        except Exception as e:
            logger.warning(f"Pending insight retrieval failed: {e}")
            return []

    def mark_insight_surfaced(self, insight_uid: str) -> bool:
        """Mark a pending insight as surfaced (shown to user)."""
        try:
            from neomodel import db

            now = datetime.now(timezone.utc).isoformat()
            db.cypher_query(
                """
                MATCH (pi:PendingInsight {uid: $uid})
                SET pi.status = 'surfaced',
                    pi.surfaced_at = datetime($now)
                """,
                {"uid": insight_uid, "now": now},
            )
            return True
        except Exception as e:
            logger.warning(f"Insight surfacing failed: {e}")
            return False

    # ──────────────────────────────────────────────────────────────────────
    # Description Generators
    # ──────────────────────────────────────────────────────────────────────

    def _describe_new_cluster(self, action: dict) -> str:
        """Generate a brief description for a new cluster detection."""
        cluster_type = action.get("cluster_type", "unknown")
        shared = action.get("shared_coordinates", {})

        type_desc = {
            "same_time_diff_emotion": "emotional range within a relationship",
            "same_person_diff_time": "recurring pattern with a person",
            "same_context_diff_person": "emotion attached to a setting",
            "same_action_diff_everything": "structural response to an action",
            "same_emotion_diff_source": "same feeling from different sources",
            "cross_dimensional": "complex multi-dimensional pattern",
        }

        desc = type_desc.get(cluster_type, "pattern grouping")

        # Add shared dimension details
        details = []
        for dim, values in shared.items():
            if values:
                details.append(f"{dim}: {', '.join(values[:3])}")

        if details:
            desc += f" ({'; '.join(details)})"

        return f"New cluster detected: {desc}"

    def _describe_trajectory_shift(self, update: dict) -> str:
        """Generate a brief description for a trajectory shift."""
        cluster_id = update.get("cluster_id", "unknown")
        old = update.get("old_status", "?")
        new = update.get("new_status", "?")
        strength = update.get("strength", 0)

        if new == "dissolved":
            return (
                f"Cluster {cluster_id} has dissolved "
                f"(strength dropped to {strength:.2f}). "
                f"The pattern may no longer be active."
            )
        elif new == "weakening":
            return (
                f"Cluster {cluster_id} is weakening "
                f"(strength: {strength:.2f}). "
                f"Recent signals haven't reinforced this pattern."
            )
        elif old in ("dissolved", "weakening") and new == "active":
            return (
                f"Cluster {cluster_id} has re-strengthened "
                f"(strength: {strength:.2f}). "
                f"This pattern is emerging again."
            )
        else:
            return (
                f"Cluster {cluster_id} shifted from {old} to {new} "
                f"(strength: {strength:.2f})."
            )


# ──────────────────────────────────────────────────────────────────────────────
# Convenience Functions
# ──────────────────────────────────────────────────────────────────────────────

def run_detection_cycle(workspace_id: Optional[str] = None) -> dict:
    """Run a full graph agent detection cycle."""
    agent = GraphAgent(workspace_id=workspace_id)
    return agent.run_detection_cycle()
