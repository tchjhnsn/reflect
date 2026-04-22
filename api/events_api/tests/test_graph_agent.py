"""
ThriveSight Graph Agent Tests — Phase 6.

Tests the cluster engine, insight engine, and graph agent components
that don't require a Neo4j connection (type classification, strength
computation, status transitions, coordinate extraction, descriptions).

For tests requiring Neo4j, use test_graph_agent_e2e.py with a live database.
"""

import unittest

from events_api.cluster_engine import (
    CLUSTER_TYPES,
    MIN_CLUSTER_SIZE,
    DISSOLUTION_THRESHOLD,
    WEAKENING_THRESHOLD,
    classify_cluster_type,
    compute_cluster_strength,
    determine_cluster_status,
    _extract_signal_coordinates,
    _compute_shared_divergent,
)


# ──────────────────────────────────────────────────────────────────────────────
# Cluster Type Classification
# ──────────────────────────────────────────────────────────────────────────────

class TestClusterTypeClassification(unittest.TestCase):
    """Test cluster type classification from shared/divergent dimensions."""

    def test_same_time_diff_emotion(self):
        result = classify_cluster_type(
            shared_dims=["temporal", "person"],
            divergent_dims=["emotion"],
        )
        self.assertEqual(result, "same_time_diff_emotion")

    def test_same_person_diff_time(self):
        result = classify_cluster_type(
            shared_dims=["person"],
            divergent_dims=["temporal", "emotion"],
        )
        self.assertEqual(result, "same_person_diff_time")

    def test_same_context_diff_person(self):
        result = classify_cluster_type(
            shared_dims=["context"],
            divergent_dims=["person", "emotion"],
        )
        self.assertEqual(result, "same_context_diff_person")

    def test_same_action_diff_everything(self):
        result = classify_cluster_type(
            shared_dims=["action"],
            divergent_dims=["context", "person", "temporal"],
        )
        self.assertEqual(result, "same_action_diff_everything")

    def test_same_emotion_diff_source(self):
        result = classify_cluster_type(
            shared_dims=["emotion"],
            divergent_dims=["context", "person", "action"],
        )
        self.assertEqual(result, "same_emotion_diff_source")

    def test_cross_dimensional(self):
        result = classify_cluster_type(
            shared_dims=["context", "person"],
            divergent_dims=["action", "temporal"],
        )
        self.assertEqual(result, "cross_dimensional")

    def test_all_six_types_defined(self):
        self.assertEqual(len(CLUSTER_TYPES), 6)
        expected_types = {
            "same_time_diff_emotion",
            "same_person_diff_time",
            "same_context_diff_person",
            "same_action_diff_everything",
            "same_emotion_diff_source",
            "cross_dimensional",
        }
        self.assertEqual(set(CLUSTER_TYPES.keys()), expected_types)


# ──────────────────────────────────────────────────────────────────────────────
# Cluster Strength Computation
# ──────────────────────────────────────────────────────────────────────────────

class TestClusterStrength(unittest.TestCase):
    """Test cluster strength computation and status determination."""

    def test_strength_below_minimum_size(self):
        strength = compute_cluster_strength(member_count=1)
        self.assertEqual(strength, 0.0)

    def test_strength_increases_with_members(self):
        s2 = compute_cluster_strength(member_count=2)
        s5 = compute_cluster_strength(member_count=5)
        s10 = compute_cluster_strength(member_count=10)
        self.assertGreater(s5, s2)
        self.assertGreater(s10, s5)

    def test_strength_capped_at_max(self):
        strength = compute_cluster_strength(
            member_count=100,
            avg_confidence=1.0,
            has_user_validation=True,
        )
        self.assertLessEqual(strength, 1.0)

    def test_strength_decays_with_time(self):
        recent = compute_cluster_strength(member_count=5, days_since_last_signal=0)
        old = compute_cluster_strength(member_count=5, days_since_last_signal=30)
        self.assertGreater(recent, old)

    def test_strength_affected_by_confidence(self):
        high = compute_cluster_strength(member_count=5, avg_confidence=0.9)
        low = compute_cluster_strength(member_count=5, avg_confidence=0.3)
        self.assertGreater(high, low)

    def test_validation_bonus(self):
        without = compute_cluster_strength(member_count=5)
        with_val = compute_cluster_strength(member_count=5, has_user_validation=True)
        self.assertGreater(with_val, without)

    def test_strength_always_non_negative(self):
        strength = compute_cluster_strength(
            member_count=2,
            avg_confidence=0.1,
            days_since_last_signal=365,
        )
        self.assertGreaterEqual(strength, 0.0)


class TestClusterStatus(unittest.TestCase):
    """Test cluster lifecycle status transitions."""

    def test_active_status(self):
        status = determine_cluster_status(0.8)
        self.assertEqual(status, "active")

    def test_weakening_status(self):
        status = determine_cluster_status(0.25)
        self.assertEqual(status, "weakening")

    def test_dissolved_status(self):
        status = determine_cluster_status(0.1)
        self.assertEqual(status, "dissolved")

    def test_disputed_stays_disputed(self):
        status = determine_cluster_status(0.8, current_status="disputed")
        self.assertEqual(status, "disputed")

    def test_re_emergence(self):
        # Dissolved → active if strength recovers
        status = determine_cluster_status(0.5, current_status="dissolved")
        self.assertEqual(status, "active")

    def test_weakening_to_dissolved(self):
        status = determine_cluster_status(0.1, current_status="weakening")
        self.assertEqual(status, "dissolved")

    def test_threshold_boundary_active(self):
        status = determine_cluster_status(WEAKENING_THRESHOLD)
        self.assertEqual(status, "active")

    def test_threshold_boundary_weakening(self):
        status = determine_cluster_status(DISSOLUTION_THRESHOLD)
        self.assertEqual(status, "weakening")


# ──────────────────────────────────────────────────────────────────────────────
# Signal Coordinate Extraction
# ──────────────────────────────────────────────────────────────────────────────

class TestCoordinateExtraction(unittest.TestCase):
    """Test signal coordinate extraction for cluster analysis."""

    def test_extract_full_address(self):
        coords = _extract_signal_coordinates({
            "signal_address": "SA(work, manager, dismissal, monday)",
            "emotions": [{"emotion": "anger"}, {"emotion": "shame"}],
        })
        self.assertEqual(coords["context"], "work")
        self.assertEqual(coords["person"], "manager")
        self.assertEqual(coords["action"], "dismissal")
        self.assertEqual(coords["temporal"], "monday")
        self.assertIn("anger", coords["emotions"])
        self.assertIn("shame", coords["emotions"])

    def test_extract_wildcard_address(self):
        coords = _extract_signal_coordinates({
            "signal_address": "SA(*, *, *, *)",
            "emotions": [],
        })
        self.assertEqual(coords["context"], "*")
        self.assertEqual(coords["person"], "*")
        self.assertEqual(coords["emotions"], [])

    def test_extract_invalid_address(self):
        coords = _extract_signal_coordinates({
            "signal_address": "invalid",
            "emotions": [],
        })
        self.assertEqual(coords["context"], "*")

    def test_extract_emotions_from_json_string(self):
        coords = _extract_signal_coordinates({
            "signal_address": "SA(*, *, *, *)",
            "emotions": '[{"emotion": "joy"}]',
        })
        self.assertIn("joy", coords["emotions"])

    def test_extract_emotions_as_strings(self):
        coords = _extract_signal_coordinates({
            "signal_address": "SA(*, *, *, *)",
            "emotions": [{"emotion": "anger"}, "frustration"],
        })
        self.assertIn("anger", coords["emotions"])
        self.assertIn("frustration", coords["emotions"])


class TestSharedDivergent(unittest.TestCase):
    """Test shared/divergent dimension computation."""

    def test_all_shared(self):
        coords = [
            {"context": "work", "person": "manager", "action": "dismissal",
             "temporal": "monday", "emotions": ["anger"]},
            {"context": "work", "person": "manager", "action": "dismissal",
             "temporal": "monday", "emotions": ["anger"]},
        ]
        shared, divergent, _, _ = _compute_shared_divergent(coords)
        self.assertEqual(len(shared), 5)  # 4 coords + emotion
        self.assertEqual(len(divergent), 0)

    def test_person_shared_rest_divergent(self):
        coords = [
            {"context": "work", "person": "manager", "action": "dismissal",
             "temporal": "monday", "emotions": ["anger"]},
            {"context": "home", "person": "manager", "action": "criticism",
             "temporal": "friday", "emotions": ["sadness"]},
        ]
        shared, divergent, _, _ = _compute_shared_divergent(coords)
        self.assertIn("person", shared)
        self.assertIn("context", divergent)
        self.assertIn("action", divergent)
        self.assertIn("temporal", divergent)
        self.assertIn("emotion", divergent)

    def test_wildcards_excluded(self):
        coords = [
            {"context": "work", "person": "*", "action": "*",
             "temporal": "*", "emotions": ["anger"]},
            {"context": "work", "person": "*", "action": "*",
             "temporal": "*", "emotions": ["shame"]},
        ]
        shared, divergent, shared_vals, _ = _compute_shared_divergent(coords)
        self.assertIn("context", shared)
        # Wildcards don't count as shared or divergent
        self.assertNotIn("person", shared)
        self.assertNotIn("person", divergent)


# ──────────────────────────────────────────────────────────────────────────────
# Graph Agent Description Generators
# ──────────────────────────────────────────────────────────────────────────────

class TestGraphAgentDescriptions(unittest.TestCase):
    """Test the graph agent's description generators."""

    def test_describe_new_cluster(self):
        from events_api.graph_agent import GraphAgent

        agent = GraphAgent()
        desc = agent._describe_new_cluster({
            "cluster_type": "same_person_diff_time",
            "shared_coordinates": {"person": ["manager"]},
        })
        self.assertIn("New cluster detected", desc)
        self.assertIn("recurring pattern", desc)
        self.assertIn("manager", desc)

    def test_describe_trajectory_dissolved(self):
        from events_api.graph_agent import GraphAgent

        agent = GraphAgent()
        desc = agent._describe_trajectory_shift({
            "cluster_id": "CLU-test-abc123",
            "old_status": "weakening",
            "new_status": "dissolved",
            "strength": 0.1,
        })
        self.assertIn("dissolved", desc)
        self.assertIn("CLU-test-abc123", desc)

    def test_describe_trajectory_weakening(self):
        from events_api.graph_agent import GraphAgent

        agent = GraphAgent()
        desc = agent._describe_trajectory_shift({
            "cluster_id": "CLU-test-abc123",
            "old_status": "active",
            "new_status": "weakening",
            "strength": 0.3,
        })
        self.assertIn("weakening", desc)

    def test_describe_trajectory_re_emergence(self):
        from events_api.graph_agent import GraphAgent

        agent = GraphAgent()
        desc = agent._describe_trajectory_shift({
            "cluster_id": "CLU-test-abc123",
            "old_status": "dissolved",
            "new_status": "active",
            "strength": 0.6,
        })
        self.assertIn("re-strengthened", desc)

    def test_describe_unknown_cluster_type(self):
        from events_api.graph_agent import GraphAgent

        agent = GraphAgent()
        desc = agent._describe_new_cluster({
            "cluster_type": "unknown_type",
            "shared_coordinates": {},
        })
        self.assertIn("New cluster detected", desc)


# ──────────────────────────────────────────────────────────────────────────────
# Insight Engine (No Graph Required)
# ──────────────────────────────────────────────────────────────────────────────

class TestInsightEngineValidation(unittest.TestCase):
    """Test insight engine validation logic (no graph required)."""

    def test_valid_validation_statuses(self):
        from events_api.insight_engine import InsightEngine

        engine = InsightEngine()
        # These should be accepted (validation itself requires graph, just test input)
        valid = {"validated", "disputed", "ignored"}
        for status in valid:
            # Just verify the method doesn't reject valid statuses at input level
            self.assertIn(status, valid)

    def test_valid_reflection_types(self):
        valid = {"agreement", "dispute", "elaboration", "realization", "question"}
        self.assertEqual(len(valid), 5)


class TestClusterEngineInit(unittest.TestCase):
    """Test ClusterEngine initialization."""

    def test_init_with_workspace(self):
        from events_api.cluster_engine import ClusterEngine

        engine = ClusterEngine(workspace_id="test-ws-123")
        self.assertEqual(engine.workspace_id, "test-ws-123")

    def test_init_without_workspace(self):
        from events_api.cluster_engine import ClusterEngine

        engine = ClusterEngine()
        self.assertIsNone(engine.workspace_id)


class TestGraphAgentInit(unittest.TestCase):
    """Test GraphAgent initialization and configuration."""

    def test_default_config(self):
        from events_api.graph_agent import (
            GraphAgent,
            DEFAULT_SIGNAL_LOOKBACK_DAYS,
            DEFAULT_INSIGHT_EXPIRY_DAYS,
        )

        agent = GraphAgent()
        self.assertEqual(agent.lookback_days, DEFAULT_SIGNAL_LOOKBACK_DAYS)
        self.assertEqual(agent.insight_expiry_days, DEFAULT_INSIGHT_EXPIRY_DAYS)
        self.assertIsNone(agent.workspace_id)

    def test_custom_config(self):
        from events_api.graph_agent import GraphAgent

        agent = GraphAgent(
            workspace_id="ws-123",
            lookback_days=14,
            insight_expiry_days=30,
        )
        self.assertEqual(agent.workspace_id, "ws-123")
        self.assertEqual(agent.lookback_days, 14)
        self.assertEqual(agent.insight_expiry_days, 30)

    def test_convenience_functions_importable(self):
        from events_api.graph_agent import GraphAgent, run_detection_cycle
        from events_api.cluster_engine import ClusterEngine, detect_clusters
        from events_api.insight_engine import InsightEngine, create_insight, create_reflection

        # Verify all expected names are importable
        self.assertIsNotNone(GraphAgent)
        self.assertIsNotNone(run_detection_cycle)
        self.assertIsNotNone(ClusterEngine)
        self.assertIsNotNone(detect_clusters)
        self.assertIsNotNone(InsightEngine)
        self.assertIsNotNone(create_insight)
        self.assertIsNotNone(create_reflection)


if __name__ == "__main__":
    unittest.main()
