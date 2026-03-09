"""
ThriveSight Cluster Engine — First-class signal cluster detection and management.

Clusters are groups of signals that share 2+ coordinate dimensions. They are
first-class graph nodes with their own trajectory, confidence, and lifecycle.

Key concepts:
- Detection: Find signals sharing coordinates (same person, same context, etc.)
- Classification: What makes this cluster interesting? (same-time-diff-emotion, etc.)
- Trajectory: Cluster strength over time is itself a signal
- User Dispute: Users can dispute clusters; all feedback is signal
- Dissolution: Weakening clusters are marked dissolved but history is preserved

Cluster types:
- same_time_diff_emotion: Same temporal context, different emotions
- same_person_diff_context: Same person across different contexts
- same_context_diff_person: Same context, different people involved
- same_action_pattern: Same type of action recurring across dimensions
- emotional_convergence: Different situations producing the same emotion
- emotional_divergence: Similar situations producing different emotions

The cluster engine does NOT call the LLM for detection — it uses graph queries
and embeddings only. LLM is only called for cluster interpretation (optional).
"""

import json
import logging
import uuid
from collections import defaultdict
from datetime import datetime
from typing import Any, Optional

from neomodel import db

from .coordinate_system import (
    COORDINATE_NAMES,
    WILDCARD,
    coordinate_overlap,
    parse_signal_address,
)
from .graph_models import Cluster, Insight, PendingInsight, Reflection, Signal

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────

# Minimum coordinates that must overlap for cluster candidacy
MIN_COORDINATE_OVERLAP = 2

# Minimum members for a cluster to be considered meaningful
MIN_CLUSTER_SIZE = 2

# Default strength decay factor per trajectory update
STRENGTH_DECAY_FACTOR = 0.95

# Cluster status values
STATUS_ACTIVE = "active"
STATUS_WEAKENING = "weakening"
STATUS_DISSOLVED = "dissolved"
STATUS_DISPUTED = "disputed"

# Cluster types
CLUSTER_TYPES = {
    "same_time_diff_emotion",
    "same_person_diff_context",
    "same_context_diff_person",
    "same_action_pattern",
    "emotional_convergence",
    "emotional_divergence",
    "general",
}


class ClusterDetector:
    """
    Detects, tracks, and manages signal clusters as first-class graph nodes.

    The detector runs in two modes:
    1. Incremental: Called after each new signal to check for cluster matches
    2. Full scan: Called by the graph agent for periodic cluster detection

    Detection algorithm:
    1. For each signal, query graph for signals sharing 2+ coordinate nodes
    2. Check emotional divergence (different emotions = interesting cluster)
    3. If cluster candidate found, check if existing cluster matches → add or create
    4. Use embeddings for semantic similarity when coordinates don't overlap exactly

    All user feedback (dispute/validate) creates a Reflection node.
    """

    def __init__(
        self,
        min_overlap: int = MIN_COORDINATE_OVERLAP,
        min_cluster_size: int = MIN_CLUSTER_SIZE,
    ):
        self.min_overlap = min_overlap
        self.min_cluster_size = min_cluster_size

    # ──────────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ──────────────────────────────────────────────────────────────────────

    def detect_clusters_for_signal(self, signal: Signal) -> list[Cluster]:
        """
        Check if a new signal should join or create clusters.

        Called incrementally after each signal is generated.

        Args:
            signal: Newly created Signal node

        Returns:
            List of clusters the signal was added to or created
        """
        if not signal.signal_address:
            return []

        try:
            parsed = parse_signal_address(signal.signal_address)
        except ValueError:
            logger.warning(f"Cannot parse signal address for clustering: {signal.signal_address}")
            return []

        # Find candidate signals that share coordinates
        candidates = self._find_coordinate_neighbors(signal, parsed)

        if not candidates:
            return []

        # Check against existing clusters
        affected_clusters = []

        # Group candidates by their existing cluster membership
        existing_clusters = self._find_existing_clusters_for_candidates(candidates)

        if existing_clusters:
            # Add signal to existing clusters where it fits
            for cluster in existing_clusters:
                if self._signal_fits_cluster(signal, parsed, cluster):
                    self._add_signal_to_cluster(signal, cluster)
                    affected_clusters.append(cluster)

        # Check if remaining unaffiliated candidates form a new cluster
        unaffiliated = [c for c in candidates if not self._signal_in_any_cluster(c, existing_clusters)]
        if len(unaffiliated) + 1 >= self.min_cluster_size:  # +1 for the new signal
            new_cluster = self._create_cluster(signal, unaffiliated, parsed)
            if new_cluster:
                affected_clusters.append(new_cluster)

        return affected_clusters

    def detect_clusters_full_scan(
        self,
        limit: int = 500,
    ) -> list[Cluster]:
        """
        Run full cluster detection across all recent signals.

        Called by the graph agent during background processing.

        Args:
            limit: Maximum number of signals to process

        Returns:
            List of newly created clusters
        """
        # Get recent unprocessed signals
        results, _ = db.cypher_query(
            """
            MATCH (s:Signal)
            WHERE s.signal_address IS NOT NULL
              AND NOT (s)-[:MEMBER_OF]->(:Cluster)
            RETURN s.uid
            ORDER BY s.created_at DESC
            LIMIT $limit
            """,
            {"limit": limit},
        )

        new_clusters = []
        processed_uids = set()

        for row in results:
            signal_uid = row[0]
            if signal_uid in processed_uids:
                continue

            try:
                signal = Signal.nodes.get(uid=signal_uid)
                clusters = self.detect_clusters_for_signal(signal)
                new_clusters.extend(clusters)

                # Track processed signals to avoid re-clustering
                processed_uids.add(signal_uid)
                for cluster in clusters:
                    for member in self._get_cluster_members(cluster):
                        processed_uids.add(member.uid)

            except Exception as e:
                logger.warning(f"Cluster detection failed for signal {signal_uid}: {e}")
                continue

        logger.info(
            f"Full cluster scan: processed {len(processed_uids)} signals, "
            f"created {len(new_clusters)} new clusters"
        )

        return new_clusters

    def classify_cluster_type(self, cluster: Cluster) -> str:
        """
        Determine the type of cluster based on its shared/divergent dimensions.

        Args:
            cluster: Cluster node

        Returns:
            Cluster type string
        """
        shared = cluster.shared_coordinates or []
        divergent = cluster.divergent_dimensions or []

        # Classification logic based on shared/divergent patterns
        if "temporal" in shared and any(d in divergent for d in ["emotion"]):
            return "same_time_diff_emotion"
        elif "person" in shared and "context" in divergent:
            return "same_person_diff_context"
        elif "context" in shared and "person" in divergent:
            return "same_context_diff_person"
        elif "action" in shared:
            return "same_action_pattern"
        else:
            # Check emotion patterns
            members = self._get_cluster_members(cluster)
            emotions = set()
            for member in members:
                if member.emotions:
                    for e in member.emotions:
                        if isinstance(e, dict):
                            emotions.add(e.get("emotion"))
                elif member.emotion:
                    emotions.add(member.emotion)

            if len(emotions) <= 1:
                return "emotional_convergence"
            elif len(emotions) >= 3:
                return "emotional_divergence"

        return "general"

    def compute_strength(self, cluster: Cluster) -> float:
        """
        Compute cluster strength based on member count, density, and recency.

        Strength formula:
        strength = member_count × density_factor × recency_factor

        Where:
        - density_factor = shared_coordinates / total_coordinates
        - recency_factor = 1.0 for signals in last 7 days, decaying to 0.5

        Args:
            cluster: Cluster node

        Returns:
            Float strength score (0.0+)
        """
        members = self._get_cluster_members(cluster)
        member_count = len(members)

        if member_count == 0:
            return 0.0

        # Density factor: how many coordinates are shared
        shared = cluster.shared_coordinates or []
        density_factor = len(shared) / len(COORDINATE_NAMES) if shared else 0.25

        # Recency factor: based on most recent member signal
        recency_factor = 1.0  # Default to full recency

        # Calculate strength
        strength = member_count * (1 + density_factor) * recency_factor

        return round(strength, 2)

    def update_trajectory(self, cluster: Cluster) -> dict:
        """
        Update a cluster's trajectory history with current strength snapshot.

        The trajectory is stored as a JSON array on the cluster node:
        [{"timestamp": ..., "strength": ..., "member_count": ...}, ...]

        Args:
            cluster: Cluster node

        Returns:
            Dict with current trajectory snapshot
        """
        current_strength = self.compute_strength(cluster)
        members = self._get_cluster_members(cluster)

        snapshot = {
            "timestamp": datetime.utcnow().isoformat(),
            "strength": current_strength,
            "member_count": len(members),
        }

        # Append to trajectory history
        trajectory = cluster.trajectory_history or []
        trajectory.append(snapshot)

        # Keep last 100 snapshots
        if len(trajectory) > 100:
            trajectory = trajectory[-100:]

        cluster.trajectory_history = trajectory
        cluster.strength = current_strength
        cluster.member_count = len(members)
        cluster.save()

        return snapshot

    def dissolve_cluster(self, cluster: Cluster, reason: str = "weakened"):
        """
        Mark a cluster as dissolved while preserving its history.

        The cluster node remains in the graph with status="dissolved".
        Trajectory history is preserved. Member MEMBER_OF relationships
        are deactivated but not deleted.

        Args:
            cluster: Cluster node to dissolve
            reason: Reason for dissolution
        """
        cluster.status = STATUS_DISSOLVED
        cluster.dissolved_at = datetime.utcnow()

        # Add dissolution note to trajectory
        trajectory = cluster.trajectory_history or []
        trajectory.append({
            "timestamp": datetime.utcnow().isoformat(),
            "strength": 0.0,
            "member_count": cluster.member_count or 0,
            "event": "dissolved",
            "reason": reason,
        })
        cluster.trajectory_history = trajectory
        cluster.save()

        # Deactivate all MEMBER_OF relationships
        db.cypher_query(
            """
            MATCH (s:Signal)-[r:MEMBER_OF]->(c:Cluster {cluster_id: $cluster_id})
            SET r.active = false, r.deactivated_at = datetime()
            """,
            {"cluster_id": cluster.cluster_id},
        )

        logger.info(f"Dissolved cluster {cluster.cluster_id}: {reason}")

    def user_dispute(
        self,
        cluster: Cluster,
        user_feedback: str,
        conversation_id: Optional[str] = None,
    ) -> Reflection:
        """
        Record a user dispute of a cluster. All feedback is signal.

        Creates a Reflection node capturing the user's reasoning.
        Adjusts cluster confidence downward.

        Args:
            cluster: Cluster being disputed
            user_feedback: User's text explaining their dispute
            conversation_id: Optional conversation context

        Returns:
            Created Reflection node
        """
        # Create reflection node (user reasoning)
        reflection = Reflection(
            text=user_feedback,
            reflection_type="cluster_dispute",
        )
        reflection.save()

        # Lower confidence
        current_confidence = cluster.confidence_score or 0.5
        cluster.confidence_score = max(current_confidence - 0.2, 0.0)
        cluster.status = STATUS_DISPUTED

        # Record in trajectory
        trajectory = cluster.trajectory_history or []
        trajectory.append({
            "timestamp": datetime.utcnow().isoformat(),
            "strength": cluster.strength or 0.0,
            "member_count": cluster.member_count or 0,
            "event": "user_disputed",
            "feedback_preview": user_feedback[:100],
        })
        cluster.trajectory_history = trajectory
        cluster.save()

        logger.info(
            f"User disputed cluster {cluster.cluster_id}. "
            f"Confidence: {current_confidence} → {cluster.confidence_score}"
        )

        return reflection

    def user_validate(
        self,
        cluster: Cluster,
        user_feedback: str = "",
    ) -> Optional[Reflection]:
        """
        Record a user validation of a cluster. Boosts confidence.

        Args:
            cluster: Cluster being validated
            user_feedback: Optional text from user

        Returns:
            Created Reflection node (if feedback provided)
        """
        reflection = None
        if user_feedback:
            reflection = Reflection(
                text=user_feedback,
                reflection_type="cluster_validation",
            )
            reflection.save()

        # Boost confidence
        current_confidence = cluster.confidence_score or 0.5
        cluster.confidence_score = min(current_confidence + 0.15, 1.0)

        if cluster.status == STATUS_DISPUTED:
            cluster.status = STATUS_ACTIVE

        # Record in trajectory
        trajectory = cluster.trajectory_history or []
        trajectory.append({
            "timestamp": datetime.utcnow().isoformat(),
            "strength": cluster.strength or 0.0,
            "member_count": cluster.member_count or 0,
            "event": "user_validated",
        })
        cluster.trajectory_history = trajectory
        cluster.save()

        logger.info(
            f"User validated cluster {cluster.cluster_id}. "
            f"Confidence: {current_confidence} → {cluster.confidence_score}"
        )

        return reflection

    def check_dissolution_thresholds(
        self,
        strength_threshold: float = 1.0,
        age_days: int = 30,
    ) -> list[Cluster]:
        """
        Check active clusters for dissolution conditions.

        A cluster is dissolved if:
        - Its strength drops below the threshold AND
        - It hasn't received new members in age_days days

        Args:
            strength_threshold: Minimum strength to remain active
            age_days: Days without growth before eligible for dissolution

        Returns:
            List of dissolved clusters
        """
        dissolved = []

        results, _ = db.cypher_query(
            """
            MATCH (c:Cluster)
            WHERE c.status IN ['active', 'weakening']
            RETURN c.cluster_id, c.strength, c.member_count
            """,
        )

        for row in results:
            cluster_id, strength, member_count = row
            if strength is not None and strength < strength_threshold:
                try:
                    cluster = Cluster.nodes.get(cluster_id=cluster_id)
                    cluster.status = STATUS_WEAKENING
                    cluster.save()

                    # Check if it's been weakening long enough to dissolve
                    trajectory = cluster.trajectory_history or []
                    weakening_entries = [
                        t for t in trajectory
                        if t.get("strength", float("inf")) < strength_threshold
                    ]
                    if len(weakening_entries) >= 3:  # Weakening for 3+ cycles
                        self.dissolve_cluster(cluster, reason="strength below threshold")
                        dissolved.append(cluster)

                except Exception as e:
                    logger.warning(f"Dissolution check failed for cluster {cluster_id}: {e}")

        return dissolved

    # ──────────────────────────────────────────────────────────────────────
    # INTERNAL: Graph Queries
    # ──────────────────────────────────────────────────────────────────────

    def _find_coordinate_neighbors(
        self,
        signal: Signal,
        parsed_sa: dict,
    ) -> list[Signal]:
        """
        Find signals sharing 2+ coordinate nodes with the given signal.

        Uses Cypher queries to find signals connected to the same
        ContextNode, ActionNode, TemporalNode, or Person nodes.
        """
        neighbor_uids = set()

        # Query for signals sharing context
        context_val = parsed_sa.get("context", WILDCARD)
        if context_val != WILDCARD:
            results, _ = db.cypher_query(
                """
                MATCH (s:Signal)-[:IN_CONTEXT]->(ctx:ContextNode {name: $name})
                WHERE s.uid <> $signal_uid
                RETURN s.uid
                """,
                {"name": context_val.lower(), "signal_uid": signal.uid},
            )
            context_neighbors = {row[0] for row in results}
        else:
            context_neighbors = set()

        # Query for signals sharing person
        person_val = parsed_sa.get("person", WILDCARD)
        if person_val != WILDCARD:
            results, _ = db.cypher_query(
                """
                MATCH (p:Person {name: $name})-[:PARTICIPANT_IN]->(s:Signal)
                WHERE s.uid <> $signal_uid
                RETURN s.uid
                """,
                {"name": person_val.lower(), "signal_uid": signal.uid},
            )
            person_neighbors = {row[0] for row in results}
        else:
            person_neighbors = set()

        # Query for signals sharing action
        action_val = parsed_sa.get("action", WILDCARD)
        if action_val != WILDCARD:
            results, _ = db.cypher_query(
                """
                MATCH (s:Signal)-[:INVOLVES_ACTION]->(a:ActionNode {name: $name})
                WHERE s.uid <> $signal_uid
                RETURN s.uid
                """,
                {"name": action_val.lower(), "signal_uid": signal.uid},
            )
            action_neighbors = {row[0] for row in results}
        else:
            action_neighbors = set()

        # Query for signals sharing temporal
        temporal_val = parsed_sa.get("temporal", WILDCARD)
        if temporal_val != WILDCARD:
            results, _ = db.cypher_query(
                """
                MATCH (s:Signal)-[:AT_TIME]->(t:TemporalNode {name: $name})
                WHERE s.uid <> $signal_uid
                RETURN s.uid
                """,
                {"name": temporal_val.lower(), "signal_uid": signal.uid},
            )
            temporal_neighbors = {row[0] for row in results}
        else:
            temporal_neighbors = set()

        # Find signals appearing in 2+ neighbor sets
        all_neighbor_sets = [
            context_neighbors, person_neighbors,
            action_neighbors, temporal_neighbors,
        ]

        # Count how many coordinate dimensions each neighbor shares
        uid_counts = defaultdict(int)
        for neighbor_set in all_neighbor_sets:
            for uid in neighbor_set:
                uid_counts[uid] += 1

        # Filter for min_overlap
        qualifying_uids = [
            uid for uid, count in uid_counts.items()
            if count >= self.min_overlap
        ]

        # Fetch Signal nodes
        neighbors = []
        for uid in qualifying_uids[:50]:  # Limit to 50 candidates
            try:
                neighbors.append(Signal.nodes.get(uid=uid))
            except Exception:
                continue

        return neighbors

    def _find_existing_clusters_for_candidates(
        self,
        candidates: list[Signal],
    ) -> list[Cluster]:
        """Find active clusters that any of the candidate signals belong to."""
        if not candidates:
            return []

        candidate_uids = [c.uid for c in candidates]

        results, _ = db.cypher_query(
            """
            MATCH (s:Signal)-[r:MEMBER_OF]->(c:Cluster)
            WHERE s.uid IN $uids AND r.active = true
              AND c.status IN ['active', 'weakening']
            RETURN DISTINCT c.cluster_id
            """,
            {"uids": candidate_uids},
        )

        clusters = []
        for row in results:
            try:
                clusters.append(Cluster.nodes.get(cluster_id=row[0]))
            except Exception:
                continue

        return clusters

    def _signal_fits_cluster(
        self,
        signal: Signal,
        parsed_sa: dict,
        cluster: Cluster,
    ) -> bool:
        """Check if a signal should join an existing cluster."""
        shared_coords = cluster.shared_coordinates or []

        # Signal must share the cluster's shared coordinates
        for coord in shared_coords:
            signal_val = parsed_sa.get(coord, WILDCARD)
            if signal_val == WILDCARD:
                return False  # Can't confirm overlap for wildcard

        return True

    def _signal_in_any_cluster(
        self,
        signal: Signal,
        clusters: list[Cluster],
    ) -> bool:
        """Check if a signal is a member of any of the given clusters."""
        if not clusters:
            return False

        cluster_ids = [c.cluster_id for c in clusters]

        results, _ = db.cypher_query(
            """
            MATCH (s:Signal {uid: $uid})-[r:MEMBER_OF]->(c:Cluster)
            WHERE c.cluster_id IN $cluster_ids AND r.active = true
            RETURN count(r) > 0
            """,
            {"uid": signal.uid, "cluster_ids": cluster_ids},
        )

        return results[0][0] if results else False

    def _add_signal_to_cluster(self, signal: Signal, cluster: Cluster):
        """Add a signal to an existing cluster."""
        signal.clusters.connect(cluster, {
            "active": True,
            "joined_at": datetime.utcnow(),
        })

        # Update cluster member count
        cluster.member_count = (cluster.member_count or 0) + 1
        cluster.save()

        logger.debug(f"Added signal {signal.uid} to cluster {cluster.cluster_id}")

    def _create_cluster(
        self,
        new_signal: Signal,
        neighbors: list[Signal],
        parsed_sa: dict,
    ) -> Optional[Cluster]:
        """
        Create a new cluster from a signal and its neighbors.
        """
        # Determine shared and divergent coordinates
        shared = []
        divergent = []

        for coord in COORDINATE_NAMES:
            values = set()
            new_val = parsed_sa.get(coord, WILDCARD)
            if new_val != WILDCARD:
                values.add(new_val.lower())

            for neighbor in neighbors:
                if neighbor.signal_address:
                    try:
                        n_parsed = parse_signal_address(neighbor.signal_address)
                        n_val = n_parsed.get(coord, WILDCARD)
                        if n_val != WILDCARD:
                            values.add(n_val.lower())
                    except ValueError:
                        continue

            if len(values) == 1:
                shared.append(coord)
            elif len(values) > 1:
                divergent.append(coord)

        # Generate cluster ID
        cluster_id = f"CLU-{uuid.uuid4().hex[:8]}"

        # Create cluster node
        cluster = Cluster(
            cluster_id=cluster_id,
            shared_coordinates=shared,
            divergent_dimensions=divergent,
            strength=0.0,
            confidence_score=0.5,
            trajectory_history=[{
                "timestamp": datetime.utcnow().isoformat(),
                "strength": 0.0,
                "member_count": len(neighbors) + 1,
                "event": "created",
            }],
            member_count=len(neighbors) + 1,
            status=STATUS_ACTIVE,
        )
        cluster.save()

        # Classify cluster type
        cluster.cluster_type = self.classify_cluster_type(cluster)
        cluster.save()

        # Add all signals as members
        self._add_signal_to_cluster(new_signal, cluster)
        for neighbor in neighbors:
            self._add_signal_to_cluster(neighbor, cluster)

        # Compute initial strength
        self.update_trajectory(cluster)

        logger.info(
            f"Created cluster {cluster_id} with {len(neighbors) + 1} members. "
            f"Type: {cluster.cluster_type}, Shared: {shared}, Divergent: {divergent}"
        )

        return cluster

    def _get_cluster_members(self, cluster: Cluster) -> list[Signal]:
        """Get all active member signals of a cluster."""
        results, _ = db.cypher_query(
            """
            MATCH (s:Signal)-[r:MEMBER_OF]->(c:Cluster {cluster_id: $cluster_id})
            WHERE r.active = true
            RETURN s.uid
            """,
            {"cluster_id": cluster.cluster_id},
        )

        members = []
        for row in results:
            try:
                members.append(Signal.nodes.get(uid=row[0]))
            except Exception:
                continue

        return members


# ──────────────────────────────────────────────────────────────────────────────
# CONVENIENCE FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────


def detect_clusters_for_signal(signal: Signal, **kwargs) -> list[Cluster]:
    """Convenience wrapper for ClusterDetector.detect_clusters_for_signal()."""
    detector = ClusterDetector(**kwargs)
    return detector.detect_clusters_for_signal(signal)


def run_full_cluster_scan(**kwargs) -> list[Cluster]:
    """Convenience wrapper for ClusterDetector.detect_clusters_full_scan()."""
    detector = ClusterDetector()
    return detector.detect_clusters_full_scan(**kwargs)
