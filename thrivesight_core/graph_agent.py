"""
ThriveSight Graph Agent — Background processing for cluster detection and maintenance.

The graph agent is Mode 3 of the three interaction modes. It runs autonomously
in the background, performing:
- Cluster detection across unprocessed signals
- Cluster trajectory updates
- Dissolution threshold checks
- Pending insight generation for noteworthy patterns
- Stale insight pruning
- Missing embedding backfill

Key design principles:
- NO LLM CALLS — the agent uses graph queries + embeddings only
- Lightweight inference via semantic embeddings for similarity matching
- All detections are stored as PendingInsight nodes, surfaced in Mode 2
- Configurable schedule (default: every hour)
- Safe to run concurrently with user interactions

The agent can be run as:
1. Django management command: python manage.py run_graph_agent
2. Celery task (future)
3. Direct Python call: GraphAgent().run_detection_cycle()
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Optional

from neomodel import db

from .cluster_engine import ClusterDetector
from .graph_models import Cluster, PendingInsight, Signal
from .signal_engine import SignalGenerator

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────

# Default interval between cycles (seconds)
DEFAULT_INTERVAL = 3600  # 1 hour

# Maximum signals to process per cycle
MAX_SIGNALS_PER_CYCLE = 500

# Pending insight expiry (days)
PENDING_INSIGHT_EXPIRY_DAYS = 7

# Minimum cluster strength change to generate a pending insight
TRAJECTORY_SHIFT_THRESHOLD = 2.0


class GraphAgent:
    """
    Autonomous background agent for graph maintenance and detection.

    The graph agent runs periodic detection cycles that:
    1. Find unprocessed signals and check for cluster candidates
    2. Update trajectory for all active clusters
    3. Check dissolution thresholds for weakening clusters
    4. Generate PendingInsight nodes for noteworthy patterns
    5. Prune stale pending insights
    6. Backfill missing embeddings

    Usage:
        agent = GraphAgent()
        results = agent.run_detection_cycle()

        # Or run continuously:
        agent.run_continuous(interval=3600)  # Every hour
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.cluster_detector = ClusterDetector()
        self.signal_generator = SignalGenerator(use_llm=False)  # No LLM in background

    def run_detection_cycle(self) -> dict:
        """
        Run a complete background processing cycle.

        Returns:
            Dict with results summary:
            - new_clusters: int
            - trajectories_updated: int
            - clusters_dissolved: int
            - pending_insights_created: int
            - insights_pruned: int
            - embeddings_computed: int
            - duration_seconds: float
        """
        start_time = time.time()
        results = {
            "new_clusters": 0,
            "trajectories_updated": 0,
            "clusters_dissolved": 0,
            "pending_insights_created": 0,
            "insights_pruned": 0,
            "embeddings_computed": 0,
            "errors": [],
        }

        logger.info("Graph agent: starting detection cycle")

        # Step 1: Detect new clusters
        try:
            new_clusters = self.detect_new_clusters()
            results["new_clusters"] = len(new_clusters)
            if self.verbose:
                logger.info(f"  Detected {len(new_clusters)} new clusters")
        except Exception as e:
            logger.error(f"  Cluster detection failed: {e}")
            results["errors"].append(f"cluster_detection: {e}")

        # Step 2: Update cluster trajectories
        try:
            updated = self.update_cluster_trajectories()
            results["trajectories_updated"] = updated
            if self.verbose:
                logger.info(f"  Updated {updated} cluster trajectories")
        except Exception as e:
            logger.error(f"  Trajectory update failed: {e}")
            results["errors"].append(f"trajectory_update: {e}")

        # Step 3: Check dissolution thresholds
        try:
            dissolved = self.check_dissolution_thresholds()
            results["clusters_dissolved"] = len(dissolved)
            if self.verbose:
                logger.info(f"  Dissolved {len(dissolved)} clusters")
        except Exception as e:
            logger.error(f"  Dissolution check failed: {e}")
            results["errors"].append(f"dissolution_check: {e}")

        # Step 4: Generate pending insights
        try:
            insights = self.generate_pending_insights()
            results["pending_insights_created"] = len(insights)
            if self.verbose:
                logger.info(f"  Created {len(insights)} pending insights")
        except Exception as e:
            logger.error(f"  Pending insight generation failed: {e}")
            results["errors"].append(f"pending_insights: {e}")

        # Step 5: Prune stale insights
        try:
            pruned = self.prune_stale_insights()
            results["insights_pruned"] = pruned
            if self.verbose:
                logger.info(f"  Pruned {pruned} stale insights")
        except Exception as e:
            logger.error(f"  Insight pruning failed: {e}")
            results["errors"].append(f"prune_insights: {e}")

        # Step 6: Backfill missing embeddings
        try:
            computed = self.compute_missing_embeddings()
            results["embeddings_computed"] = computed
            if self.verbose:
                logger.info(f"  Computed {computed} missing embeddings")
        except Exception as e:
            logger.error(f"  Embedding backfill failed: {e}")
            results["errors"].append(f"embeddings: {e}")

        duration = time.time() - start_time
        results["duration_seconds"] = round(duration, 2)

        logger.info(
            f"Graph agent: cycle complete in {duration:.1f}s. "
            f"Clusters: +{results['new_clusters']}, "
            f"Trajectories: {results['trajectories_updated']}, "
            f"Dissolved: {results['clusters_dissolved']}, "
            f"Insights: +{results['pending_insights_created']}, "
            f"Embeddings: {results['embeddings_computed']}"
        )

        return results

    def run_continuous(self, interval: int = DEFAULT_INTERVAL):
        """
        Run the agent continuously at the specified interval.

        Args:
            interval: Seconds between cycles
        """
        logger.info(f"Graph agent: starting continuous mode (interval={interval}s)")

        while True:
            try:
                self.run_detection_cycle()
            except Exception as e:
                logger.error(f"Graph agent cycle failed: {e}")

            logger.info(f"Graph agent: sleeping {interval}s until next cycle")
            time.sleep(interval)

    # ──────────────────────────────────────────────────────────────────────
    # DETECTION STEPS
    # ──────────────────────────────────────────────────────────────────────

    def detect_new_clusters(self) -> list[Cluster]:
        """
        Find unprocessed signals and check for cluster candidates.

        Uses the ClusterDetector's full scan mode.
        """
        return self.cluster_detector.detect_clusters_full_scan(
            limit=MAX_SIGNALS_PER_CYCLE
        )

    def update_cluster_trajectories(self) -> int:
        """
        Recalculate strength for all active clusters.

        Returns:
            Number of clusters updated
        """
        results, _ = db.cypher_query(
            """
            MATCH (c:Cluster)
            WHERE c.status IN ['active', 'weakening']
            RETURN c.cluster_id
            """,
        )

        updated = 0
        for row in results:
            try:
                cluster = Cluster.nodes.get(cluster_id=row[0])
                self.cluster_detector.update_trajectory(cluster)
                updated += 1
            except Exception as e:
                logger.debug(f"Trajectory update failed for {row[0]}: {e}")

        return updated

    def check_dissolution_thresholds(self) -> list[Cluster]:
        """
        Mark weakening clusters as dissolved if they meet dissolution criteria.
        """
        return self.cluster_detector.check_dissolution_thresholds()

    def generate_pending_insights(self) -> list[PendingInsight]:
        """
        Create PendingInsight nodes for noteworthy patterns.

        Checks for:
        - Newly formed clusters (detection_type: "new_cluster")
        - Clusters that significantly strengthened (detection_type: "cluster_strengthened")
        - Clusters approaching dissolution (detection_type: "cluster_dissolving")
        - Trajectory shifts (detection_type: "trajectory_shift")
        """
        insights = []

        # Check for new clusters without pending insights
        new_cluster_results, _ = db.cypher_query(
            """
            MATCH (c:Cluster)
            WHERE c.status = 'active'
              AND NOT (c)<-[:AWAITING_REVIEW]-(:PendingInsight)
            RETURN c.cluster_id, c.cluster_type, c.member_count, c.strength
            LIMIT 20
            """,
        )

        for row in new_cluster_results:
            cluster_id, cluster_type, member_count, strength = row
            if member_count and member_count >= 3:  # Only for meaningful clusters
                pi = PendingInsight(
                    detection_type="new_cluster",
                    description=(
                        f"New {cluster_type or 'general'} cluster detected with "
                        f"{member_count} signals (strength: {strength or 0:.1f})"
                    ),
                    confidence=min((strength or 0) / 10.0, 0.9),
                    status="pending",
                )
                pi.save()

                # Link to cluster
                try:
                    db.cypher_query(
                        """
                        MATCH (pi:PendingInsight {uid: $pi_uid})
                        MATCH (c:Cluster {cluster_id: $cluster_id})
                        MERGE (pi)-[:AWAITING_REVIEW]->(c)
                        """,
                        {"pi_uid": pi.uid, "cluster_id": cluster_id},
                    )
                except Exception as e:
                    logger.debug(f"Could not link pending insight to cluster: {e}")

                insights.append(pi)

        # Check for trajectory shifts
        shift_results, _ = db.cypher_query(
            """
            MATCH (c:Cluster)
            WHERE c.status IN ['active', 'weakening']
              AND c.trajectory_history IS NOT NULL
            RETURN c.cluster_id, c.trajectory_history, c.strength
            """,
        )

        for row in shift_results:
            cluster_id, trajectory, current_strength = row
            if trajectory and len(trajectory) >= 3:
                # Check for significant strength change
                recent = trajectory[-3:]
                strengths = [t.get("strength", 0) for t in recent if isinstance(t, dict)]
                if len(strengths) >= 2:
                    change = abs(strengths[-1] - strengths[0])
                    if change >= TRAJECTORY_SHIFT_THRESHOLD:
                        direction = "strengthening" if strengths[-1] > strengths[0] else "weakening"
                        pi = PendingInsight(
                            detection_type="trajectory_shift",
                            description=(
                                f"Cluster {cluster_id} is {direction}: "
                                f"strength {strengths[0]:.1f} → {strengths[-1]:.1f}"
                            ),
                            confidence=0.6,
                            status="pending",
                        )
                        pi.save()

                        try:
                            db.cypher_query(
                                """
                                MATCH (pi:PendingInsight {uid: $pi_uid})
                                MATCH (c:Cluster {cluster_id: $cluster_id})
                                MERGE (pi)-[:AWAITING_REVIEW]->(c)
                                """,
                                {"pi_uid": pi.uid, "cluster_id": cluster_id},
                            )
                        except Exception:
                            pass

                        insights.append(pi)

        return insights

    def prune_stale_insights(self) -> int:
        """
        Remove old pending insights that were never surfaced.

        Returns:
            Number of insights pruned
        """
        cutoff = datetime.utcnow() - timedelta(days=PENDING_INSIGHT_EXPIRY_DAYS)

        results, _ = db.cypher_query(
            """
            MATCH (pi:PendingInsight)
            WHERE pi.status = 'pending'
              AND pi.created_at < $cutoff
            SET pi.status = 'expired'
            RETURN count(pi)
            """,
            {"cutoff": cutoff},
        )

        pruned = results[0][0] if results else 0
        return pruned

    def compute_missing_embeddings(self, limit: int = 100) -> int:
        """
        Backfill embeddings on signals that lack them.

        Returns:
            Number of embeddings computed
        """
        results, _ = db.cypher_query(
            """
            MATCH (s:Signal)
            WHERE s.embedding IS NULL
              AND s.signal_address IS NOT NULL
            RETURN s.uid
            LIMIT $limit
            """,
            {"limit": limit},
        )

        computed = 0
        for row in results:
            try:
                signal = Signal.nodes.get(uid=row[0])
                embedding = self.signal_generator.compute_embedding(signal)
                if embedding:
                    computed += 1
            except Exception as e:
                logger.debug(f"Embedding computation failed for {row[0]}: {e}")

        return computed
