"""
ThriveSight Cluster Engine — Signal Grouping with Coordinate Density.

Detects clusters of signals that share coordinate values but diverge along
other dimensions. Manages cluster lifecycle: formation, strengthening,
weakening, dissolution, and re-emergence.

Six cluster types based on shared vs divergent SA dimensions:
- same_time_diff_emotion
- same_person_diff_time
- same_context_diff_person
- same_action_diff_everything
- same_emotion_diff_source
- cross_dimensional

All cluster detection runs in Python — no GDS graph projections.
"""

import json
import logging
import math
from datetime import datetime, timezone
from typing import Optional

from events_api.coordinate_system import parse_signal_address, WILDCARD

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Cluster Type Definitions
# ──────────────────────────────────────────────────────────────────────────────

CLUSTER_TYPES = {
    "same_time_diff_emotion": {
        "shared": ["temporal", "person"],
        "divergent": ["emotion"],
        "description": "Emotional range within a relationship at a point in time",
    },
    "same_person_diff_time": {
        "shared": ["person"],
        "divergent": ["temporal", "emotion"],
        "description": "Recurring pattern with a specific person",
    },
    "same_context_diff_person": {
        "shared": ["context"],
        "divergent": ["person", "emotion"],
        "description": "Emotion attached to the setting, not individual",
    },
    "same_action_diff_everything": {
        "shared": ["action"],
        "divergent": ["context", "person", "temporal"],
        "description": "Structural relationship to the action itself",
    },
    "same_emotion_diff_source": {
        "shared": ["emotion"],
        "divergent": ["context", "person", "action"],
        "description": "Same feeling, different triggers",
    },
    "cross_dimensional": {
        "shared": [],
        "divergent": [],
        "description": "Complex multi-axis overlap (2+ shared, 2+ divergent)",
    },
}

# Detection thresholds
MIN_CLUSTER_SIZE = 2
STRENGTH_DECAY_RATE = 0.05  # Per-day decay when no new signals
DISSOLUTION_THRESHOLD = 0.15  # Below this, cluster is dissolved
WEAKENING_THRESHOLD = 0.35  # Below this, cluster is weakening
STRENGTHENING_BONUS = 0.15  # Added per new member signal
MAX_STRENGTH = 1.0
EMBEDDING_SIMILARITY_THRESHOLD = 0.6


# ──────────────────────────────────────────────────────────────────────────────
# Signal Coordinate Extraction
# ──────────────────────────────────────────────────────────────────────────────

def _extract_signal_coordinates(signal_data: dict) -> dict:
    """
    Extract resolved coordinate values from a signal.

    Args:
        signal_data: Dict with at least 'signal_address' and 'emotions'.

    Returns:
        Dict with context, person, action, temporal, emotions keys.
    """
    address = signal_data.get("signal_address", "SA(*, *, *, *)")
    try:
        parsed = parse_signal_address(address)
    except (ValueError, TypeError):
        parsed = {"context": WILDCARD, "person": WILDCARD,
                  "action": WILDCARD, "temporal": WILDCARD}

    # Extract emotion names from emotions array
    emotions_raw = signal_data.get("emotions", [])
    if isinstance(emotions_raw, str):
        try:
            emotions_raw = json.loads(emotions_raw)
        except (json.JSONDecodeError, TypeError):
            emotions_raw = []

    emotion_names = []
    for e in emotions_raw:
        if isinstance(e, dict):
            emotion_names.append(e.get("emotion", "unknown"))
        elif isinstance(e, str):
            emotion_names.append(e)

    return {
        "context": parsed.get("context", WILDCARD),
        "person": parsed.get("person", WILDCARD),
        "action": parsed.get("action", WILDCARD),
        "temporal": parsed.get("temporal", WILDCARD),
        "emotions": emotion_names,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Cluster Type Classification
# ──────────────────────────────────────────────────────────────────────────────

def classify_cluster_type(shared_dims: list, divergent_dims: list) -> str:
    """
    Classify a cluster type based on shared and divergent dimensions.

    Args:
        shared_dims: Dimension names that are shared across members.
        divergent_dims: Dimension names that differ across members.

    Returns:
        Cluster type string from CLUSTER_TYPES.
    """
    shared_set = set(shared_dims)
    divergent_set = set(divergent_dims)

    # Check specific types first
    if shared_set >= {"temporal", "person"} and "emotion" in divergent_set:
        return "same_time_diff_emotion"

    if "person" in shared_set and shared_set - {"person"} == set():
        if divergent_set >= {"temporal"}:
            return "same_person_diff_time"

    if "context" in shared_set and shared_set - {"context"} == set():
        if divergent_set >= {"person"}:
            return "same_context_diff_person"

    if "action" in shared_set and shared_set - {"action"} == set():
        return "same_action_diff_everything"

    if "emotion" in shared_set and shared_set - {"emotion"} == set():
        return "same_emotion_diff_source"

    # Fallback: cross-dimensional
    if len(shared_set) >= 2 and len(divergent_set) >= 2:
        return "cross_dimensional"

    # Best guess based on what's shared
    if "person" in shared_set:
        return "same_person_diff_time"
    if "context" in shared_set:
        return "same_context_diff_person"
    if "action" in shared_set:
        return "same_action_diff_everything"
    if "emotion" in shared_set:
        return "same_emotion_diff_source"

    return "cross_dimensional"


def _compute_shared_divergent(coords_list: list) -> tuple:
    """
    Given a list of coordinate dicts, determine shared and divergent dimensions.

    Args:
        coords_list: List of dicts from _extract_signal_coordinates.

    Returns:
        (shared_dims, divergent_dims, shared_values, divergent_values)
    """
    if not coords_list:
        return [], [], {}, {}

    dims = ["context", "person", "action", "temporal"]
    shared_dims = []
    divergent_dims = []
    shared_values = {}
    divergent_values = {}

    for dim in dims:
        values = set()
        for coords in coords_list:
            val = coords.get(dim, WILDCARD)
            if val != WILDCARD:
                values.add(val.lower())

        if len(values) <= 1 and len(values) > 0:
            shared_dims.append(dim)
            shared_values[dim] = list(values)
        elif len(values) > 1:
            divergent_dims.append(dim)
            divergent_values[dim] = list(values)
        # Wildcards don't count as either

    # Check emotion dimension
    all_emotions = set()
    for coords in coords_list:
        all_emotions.update(coords.get("emotions", []))

    if len(all_emotions) <= 1 and len(all_emotions) > 0:
        shared_dims.append("emotion")
        shared_values["emotion"] = list(all_emotions)
    elif len(all_emotions) > 1:
        divergent_dims.append("emotion")
        divergent_values["emotion"] = list(all_emotions)

    return shared_dims, divergent_dims, shared_values, divergent_values


# ──────────────────────────────────────────────────────────────────────────────
# Cluster Strength Computation
# ──────────────────────────────────────────────────────────────────────────────

def compute_cluster_strength(
    member_count: int,
    avg_confidence: float = 0.7,
    days_since_last_signal: float = 0.0,
    has_user_validation: bool = False,
) -> float:
    """
    Compute cluster strength from member count, confidence, and recency.

    Strength formula:
        base = log2(member_count + 1) / log2(max_expected + 1)
        confidence_factor = avg_confidence
        recency_penalty = exp(-decay_rate * days)
        validation_bonus = 0.1 if validated

        strength = base * confidence_factor * recency_penalty + validation_bonus

    Args:
        member_count: Number of signals in the cluster.
        avg_confidence: Average confidence of member signals.
        days_since_last_signal: Days since the most recent member was added.
        has_user_validation: Whether a user has validated this cluster.

    Returns:
        Strength value clamped to [0.0, 1.0].
    """
    if member_count < MIN_CLUSTER_SIZE:
        return 0.0

    # Log-scale base (diminishing returns above ~10 members)
    max_expected = 20
    base = math.log2(member_count + 1) / math.log2(max_expected + 1)
    base = min(base, 1.0)

    # Confidence factor
    confidence_factor = max(0.1, avg_confidence)

    # Recency decay
    recency_penalty = math.exp(-STRENGTH_DECAY_RATE * days_since_last_signal)

    # User validation bonus
    validation_bonus = 0.1 if has_user_validation else 0.0

    strength = base * confidence_factor * recency_penalty + validation_bonus
    return min(MAX_STRENGTH, max(0.0, strength))


def determine_cluster_status(strength: float, current_status: str = "active") -> str:
    """
    Determine cluster lifecycle status from strength.

    Status transitions:
        active → weakening (strength < 0.35)
        weakening → dissolved (strength < 0.15)
        dissolved → active (strength >= 0.35 — re-emergence)
        disputed (set externally by user feedback)

    Args:
        strength: Current computed strength.
        current_status: Current status string.

    Returns:
        New status string.
    """
    if current_status == "disputed":
        return "disputed"

    if strength < DISSOLUTION_THRESHOLD:
        return "dissolved"
    elif strength < WEAKENING_THRESHOLD:
        return "weakening"
    else:
        return "active"


# ──────────────────────────────────────────────────────────────────────────────
# ClusterEngine
# ──────────────────────────────────────────────────────────────────────────────

class ClusterEngine:
    """
    Detects and manages signal clusters in the graph.

    The engine runs in two modes:
    1. On-signal: When a new signal is created, check if it joins existing
       clusters or forms a new one.
    2. Background: Periodically scan all signals for cluster opportunities
       and update cluster trajectories.
    """

    def __init__(self, workspace_id: Optional[str] = None):
        self.workspace_id = workspace_id

    def detect_clusters_for_signal(self, signal_data: dict) -> list:
        """
        Check if a new signal should join or create clusters.

        Args:
            signal_data: Dict with signal_address, emotions, uid, confidence_score.

        Returns:
            List of cluster actions: [{action, cluster_id, cluster_type, ...}]
        """
        new_coords = _extract_signal_coordinates(signal_data)
        actions = []

        try:
            from neomodel import db

            # Find signals sharing at least one coordinate
            candidates = self._find_coordinate_neighbors(signal_data.get("uid", ""))

            if not candidates:
                return actions

            # Group candidates by shared dimensions
            for candidate in candidates:
                cand_coords = _extract_signal_coordinates(candidate)
                all_coords = [new_coords, cand_coords]

                shared, divergent, shared_vals, divergent_vals = _compute_shared_divergent(all_coords)

                if len(shared) >= 1 and len(divergent) >= 1:
                    cluster_type = classify_cluster_type(shared, divergent)

                    # Check if a matching cluster already exists
                    existing = self._find_existing_cluster(
                        cluster_type, shared_vals
                    )

                    if existing:
                        actions.append({
                            "action": "join",
                            "cluster_id": existing["cluster_id"],
                            "cluster_type": cluster_type,
                            "signal_uid": signal_data.get("uid", ""),
                        })
                    else:
                        # Create new cluster
                        actions.append({
                            "action": "create",
                            "cluster_type": cluster_type,
                            "shared_coordinates": shared_vals,
                            "divergent_dimensions": divergent_vals,
                            "member_uids": [
                                signal_data.get("uid", ""),
                                candidate.get("uid", ""),
                            ],
                        })

        except Exception as e:
            logger.warning(f"Cluster detection failed: {e}")

        return actions

    def execute_cluster_actions(self, actions: list) -> list:
        """
        Execute cluster creation/join actions against the graph.

        Args:
            actions: List from detect_clusters_for_signal.

        Returns:
            List of cluster UIDs that were created or updated.
        """
        cluster_uids = []

        try:
            from neomodel import db

            for action in actions:
                if action["action"] == "create":
                    uid = self._create_cluster(action)
                    if uid:
                        cluster_uids.append(uid)

                elif action["action"] == "join":
                    uid = self._join_cluster(action)
                    if uid:
                        cluster_uids.append(uid)

        except Exception as e:
            logger.warning(f"Cluster action execution failed: {e}")

        return cluster_uids

    def update_cluster_trajectories(self) -> list:
        """
        Update strength and status for all active clusters.

        This is the background maintenance operation. It:
        1. Recalculates strength for active and weakening clusters
        2. Applies time decay
        3. Updates status based on strength thresholds
        4. Records trajectory history entries
        5. Marks dissolved clusters

        Returns:
            List of cluster updates: [{cluster_id, old_status, new_status, strength}]
        """
        updates = []

        try:
            from neomodel import db

            ws_filter = "AND c.workspace_id = $ws" if self.workspace_id else ""

            # Get all non-dissolved clusters
            results, _ = db.cypher_query(
                f"""
                MATCH (c:Cluster)
                WHERE c.status IN ['active', 'weakening']
                {ws_filter}
                OPTIONAL MATCH (s:Signal)-[:MEMBER_OF]->(c)
                WITH c,
                     count(s) AS member_count,
                     avg(s.confidence_score) AS avg_conf,
                     max(s.created_at) AS last_signal_at
                RETURN c.uid, c.cluster_id, c.status, c.strength,
                       c.trajectory_history,
                       member_count, avg_conf, last_signal_at
                """,
                {"ws": self.workspace_id} if self.workspace_id else {},
            )

            now = datetime.now(timezone.utc)

            for row in results:
                (uid, cluster_id, old_status, old_strength,
                 trajectory_json, member_count, avg_conf, last_signal) = row

                # Compute days since last signal
                days_since = 0.0
                if last_signal:
                    try:
                        if hasattr(last_signal, 'to_native'):
                            last_dt = last_signal.to_native()
                        else:
                            last_dt = last_signal
                        if last_dt.tzinfo is None:
                            last_dt = last_dt.replace(tzinfo=timezone.utc)
                        days_since = (now - last_dt).total_seconds() / 86400.0
                    except Exception:
                        days_since = 1.0

                # Compute new strength
                new_strength = compute_cluster_strength(
                    member_count=member_count or 0,
                    avg_confidence=avg_conf or 0.7,
                    days_since_last_signal=days_since,
                )

                new_status = determine_cluster_status(new_strength, old_status)

                # Record trajectory entry
                trajectory = []
                if trajectory_json:
                    if isinstance(trajectory_json, str):
                        try:
                            trajectory = json.loads(trajectory_json)
                        except (json.JSONDecodeError, TypeError):
                            trajectory = []
                    elif isinstance(trajectory_json, list):
                        trajectory = trajectory_json

                event = "stable"
                if new_status != old_status:
                    event = f"{old_status}_to_{new_status}"

                trajectory.append({
                    "timestamp": now.isoformat(),
                    "strength": round(new_strength, 4),
                    "member_count": member_count or 0,
                    "event": event,
                })

                # Keep last 50 trajectory entries
                trajectory = trajectory[-50:]

                # Update cluster
                update_params = {
                    "uid": uid,
                    "strength": new_strength,
                    "status": new_status,
                    "member_count": member_count or 0,
                    "trajectory": json.dumps(trajectory),
                    "now": now.isoformat(),
                }

                dissolved_set = ""
                if new_status == "dissolved" and old_status != "dissolved":
                    dissolved_set = ", c.dissolved_at = datetime($now)"

                db.cypher_query(
                    f"""
                    MATCH (c:Cluster {{uid: $uid}})
                    SET c.strength = $strength,
                        c.status = $status,
                        c.member_count = $member_count,
                        c.trajectory_history = $trajectory,
                        c.last_updated = datetime($now)
                        {dissolved_set}
                    """,
                    update_params,
                )

                updates.append({
                    "cluster_id": cluster_id,
                    "old_status": old_status,
                    "new_status": new_status,
                    "strength": round(new_strength, 4),
                    "member_count": member_count or 0,
                })

        except Exception as e:
            logger.error(f"Cluster trajectory update failed: {e}")

        return updates

    def _find_coordinate_neighbors(self, signal_uid: str) -> list:
        """Find signals sharing coordinates with the given signal."""
        try:
            from neomodel import db

            results, _ = db.cypher_query(
                """
                MATCH (s:Signal {uid: $uid})
                OPTIONAL MATCH (s)-[:IN_CONTEXT]->(ctx)<-[:IN_CONTEXT]-(n1:Signal)
                OPTIONAL MATCH (s)-[:INVOLVES_ACTION]->(act)<-[:INVOLVES_ACTION]-(n2:Signal)
                OPTIONAL MATCH (s)<-[:PARTICIPANT_IN]-(p:Person)-[:PARTICIPANT_IN]->(n3:Signal)
                WITH collect(DISTINCT n1) + collect(DISTINCT n2) + collect(DISTINCT n3) AS neighbors
                UNWIND neighbors AS n
                WHERE n IS NOT NULL AND n.uid <> $uid
                RETURN DISTINCT n.uid, n.signal_address, n.emotions, n.confidence_score
                LIMIT 50
                """,
                {"uid": signal_uid},
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
            logger.warning(f"Coordinate neighbor search failed: {e}")
            return []

    def _find_existing_cluster(self, cluster_type: str, shared_vals: dict) -> Optional[dict]:
        """Find an existing active cluster matching the type and shared coordinates."""
        try:
            from neomodel import db

            results, _ = db.cypher_query(
                """
                MATCH (c:Cluster)
                WHERE c.cluster_type = $ctype
                  AND c.status IN ['active', 'weakening']
                RETURN c.uid, c.cluster_id, c.shared_coordinates
                """,
                {"ctype": cluster_type},
            )

            for row in results:
                uid, cid, stored_shared = row
                if stored_shared:
                    if isinstance(stored_shared, str):
                        try:
                            stored_shared = json.loads(stored_shared)
                        except (json.JSONDecodeError, TypeError):
                            continue
                    # Check if shared values match
                    if self._shared_values_match(shared_vals, stored_shared):
                        return {"uid": uid, "cluster_id": cid}

            return None
        except Exception as e:
            logger.warning(f"Existing cluster search failed: {e}")
            return None

    def _shared_values_match(self, new_vals: dict, stored_vals: dict) -> bool:
        """Check if shared coordinate values overlap meaningfully."""
        for dim, values in new_vals.items():
            stored = stored_vals.get(dim, [])
            if isinstance(stored, str):
                stored = [stored]
            new_set = {v.lower() for v in values if v != WILDCARD}
            stored_set = {v.lower() for v in stored if v != WILDCARD}
            if new_set and stored_set and new_set & stored_set:
                return True
        return False

    def _create_cluster(self, action: dict) -> Optional[str]:
        """Create a new cluster node and link member signals."""
        try:
            from neomodel import db
            import hashlib

            cluster_type = action["cluster_type"]
            shared = action.get("shared_coordinates", {})
            divergent = action.get("divergent_dimensions", {})
            member_uids = action.get("member_uids", [])

            # Generate cluster ID
            hash_input = f"{cluster_type}:{json.dumps(shared, sort_keys=True)}"
            short_hash = hashlib.md5(hash_input.encode()).hexdigest()[:8]
            cluster_id = f"CLU-{cluster_type[:12]}-{short_hash}"

            now = datetime.now(timezone.utc).isoformat()
            initial_trajectory = json.dumps([{
                "timestamp": now,
                "strength": 0.3,
                "member_count": len(member_uids),
                "event": "created",
            }])

            # Create cluster
            result, _ = db.cypher_query(
                """
                CREATE (c:Cluster {
                    uid: randomUUID(),
                    cluster_id: $cid,
                    cluster_type: $ctype,
                    shared_coordinates: $shared,
                    divergent_dimensions: $divergent,
                    strength: 0.3,
                    confidence_score: 0.5,
                    trajectory_history: $trajectory,
                    member_count: $count,
                    status: 'active',
                    created_at: datetime(),
                    last_updated: datetime()
                })
                RETURN c.uid
                """,
                {
                    "cid": cluster_id,
                    "ctype": cluster_type,
                    "shared": json.dumps(shared),
                    "divergent": json.dumps(divergent),
                    "trajectory": initial_trajectory,
                    "count": len(member_uids),
                },
            )

            if not result:
                return None

            cluster_uid = result[0][0]

            # Link member signals
            for sig_uid in member_uids:
                db.cypher_query(
                    """
                    MATCH (s:Signal {uid: $sig_uid})
                    MATCH (c:Cluster {uid: $cluster_uid})
                    MERGE (s)-[:MEMBER_OF {active: true, joined_at: datetime()}]->(c)
                    """,
                    {"sig_uid": sig_uid, "cluster_uid": cluster_uid},
                )

            logger.info(f"Created cluster {cluster_id} with {len(member_uids)} members")
            return cluster_uid

        except Exception as e:
            logger.warning(f"Cluster creation failed: {e}")
            return None

    def _join_cluster(self, action: dict) -> Optional[str]:
        """Add a signal to an existing cluster."""
        try:
            from neomodel import db

            cluster_id = action["cluster_id"]
            signal_uid = action["signal_uid"]

            # Link signal to cluster
            db.cypher_query(
                """
                MATCH (s:Signal {uid: $sig_uid})
                MATCH (c:Cluster {cluster_id: $cid})
                MERGE (s)-[:MEMBER_OF {active: true, joined_at: datetime()}]->(c)
                SET c.member_count = c.member_count + 1,
                    c.strength = CASE WHEN c.strength + $bonus > 1.0 THEN 1.0
                                      ELSE c.strength + $bonus END,
                    c.last_updated = datetime()
                """,
                {
                    "sig_uid": signal_uid,
                    "cid": cluster_id,
                    "bonus": STRENGTHENING_BONUS,
                },
            )

            logger.info(f"Signal {signal_uid} joined cluster {cluster_id}")
            return cluster_id

        except Exception as e:
            logger.warning(f"Cluster join failed: {e}")
            return None


def detect_clusters(signal_data: dict, workspace_id: Optional[str] = None) -> list:
    """
    Convenience function for cluster detection.

    Args:
        signal_data: Signal dict to check for cluster membership.
        workspace_id: Optional workspace scoping.

    Returns:
        List of cluster action results.
    """
    engine = ClusterEngine(workspace_id=workspace_id)
    actions = engine.detect_clusters_for_signal(signal_data)
    return engine.execute_cluster_actions(actions)
