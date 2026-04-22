"""
ThriveSight Integration Tests — Phase 7.

Tests cross-module integration, API contract compliance, and IP zone
boundaries. These tests verify that:

1. The full pipeline works end-to-end (signal → cluster → insight → agent)
2. API response shapes match the Reflect Graph Contract
3. All proprietary logic stays in Python (no Neo4j stored procedures)
4. Module interfaces are stable and compatible

For tests requiring Neo4j, use test_integration_e2e.py with a live database.
"""

import json
import re
import os
import unittest
from unittest.mock import MagicMock, patch


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: Cross-Module Pipeline Integration
# ══════════════════════════════════════════════════════════════════════════════


class TestSignalToClusterPipeline(unittest.TestCase):
    """Test that signal generation output is compatible with cluster engine input."""

    def test_fallback_signal_shape_matches_cluster_input(self):
        """Verify keyword fallback produces signals the cluster engine can consume."""
        from events_api.signal_engine import SignalGenerator
        from events_api.cluster_engine import _extract_signal_coordinates

        generator = SignalGenerator(use_llm=False)
        result = generator.generate_from_message(
            "I'm frustrated with my manager at work today"
        )

        signals = result["signals"]
        self.assertGreater(len(signals), 0)

        signal = signals[0]
        # Cluster engine needs: signal_address, emotions, uid (optional for detection)
        self.assertIn("signal_address", signal)
        self.assertIn("emotions", signal)
        self.assertIn("confidence", signal)

        # Should be parseable by cluster engine
        coords = _extract_signal_coordinates(signal)
        self.assertIn("context", coords)
        self.assertIn("person", coords)
        self.assertIn("action", coords)
        self.assertIn("temporal", coords)
        self.assertIn("emotions", coords)

    def test_multi_emotion_signal_feeds_cluster_divergent(self):
        """Multi-emotion signals should create divergent emotion dimension for clusters."""
        from events_api.signal_engine import SignalGenerator
        from events_api.cluster_engine import _extract_signal_coordinates, _compute_shared_divergent

        gen = SignalGenerator(use_llm=False)

        # Two messages sharing context but with different emotions
        r1 = gen.generate_from_message("I'm angry about work")
        r2 = gen.generate_from_message("I feel sad about work")

        c1 = _extract_signal_coordinates(r1["signals"][0])
        c2 = _extract_signal_coordinates(r2["signals"][0])

        shared, divergent, _, _ = _compute_shared_divergent([c1, c2])

        # Both have 'work' context
        self.assertIn("context", shared)
        # Different emotions
        self.assertIn("emotion", divergent)

    def test_signal_wildcards_propagate_to_cluster(self):
        """Signals with wildcards should be handled by cluster coordinate extraction."""
        from events_api.cluster_engine import _extract_signal_coordinates

        signal = {
            "signal_address": "SA(work, *, *, *)",
            "emotions": [{"emotion": "frustration"}],
        }

        coords = _extract_signal_coordinates(signal)
        self.assertEqual(coords["context"], "work")
        self.assertEqual(coords["person"], "*")
        self.assertEqual(coords["action"], "*")
        self.assertEqual(coords["temporal"], "*")


class TestClusterToInsightPipeline(unittest.TestCase):
    """Test that cluster engine output feeds insight engine correctly."""

    def test_cluster_action_create_shape(self):
        """Verify cluster creation actions contain all fields insight engine needs."""
        from events_api.cluster_engine import classify_cluster_type

        # Simulate what detect_clusters_for_signal produces
        action = {
            "action": "create",
            "cluster_type": classify_cluster_type(["person"], ["temporal", "emotion"]),
            "shared_coordinates": {"person": ["manager"]},
            "divergent_dimensions": {"temporal": ["monday", "friday"], "emotion": ["anger", "sadness"]},
            "member_uids": ["uid-1", "uid-2"],
        }

        self.assertEqual(action["cluster_type"], "same_person_diff_time")
        self.assertIn("shared_coordinates", action)
        self.assertIn("divergent_dimensions", action)
        self.assertIn("member_uids", action)

    def test_cluster_status_triggers_insight_types(self):
        """Verify cluster status changes map to valid PendingInsight detection types."""
        from events_api.cluster_engine import determine_cluster_status

        valid_detection_types = {
            "new_cluster", "cluster_strengthened",
            "cluster_dissolving", "pattern_detected", "trajectory_shift",
        }

        # Active → weakening triggers trajectory_shift
        status = determine_cluster_status(0.25, "active")
        self.assertEqual(status, "weakening")

        # Weakening → dissolved triggers cluster_dissolving
        status = determine_cluster_status(0.1, "weakening")
        self.assertEqual(status, "dissolved")

        # Dissolved → active triggers cluster_strengthened
        status = determine_cluster_status(0.5, "dissolved")
        self.assertEqual(status, "active")


class TestContextAssemblyIntegration(unittest.TestCase):
    """Test that context assembly output feeds LLM prompts correctly."""

    def test_entity_extraction_to_prompt_assembly(self):
        """Entity extraction output should work with build_system_prompt."""
        from events_api.context_assembly import ContextAssembler, assemble_context
        from events_api.llm_prompts import build_system_prompt, get_prompt

        assembler = ContextAssembler()
        entities = assembler.extract_entities(
            "Yesterday my manager dismissed my proposal at work"
        )

        packet = assemble_context(entities)
        self.assertIsInstance(packet, str)

        # Should feed into build_system_prompt
        base = get_prompt("signal_generation")
        full_prompt = build_system_prompt(base, context_packet=packet)
        self.assertIn("Context from Graph", full_prompt)

    def test_persona_config_integrates_with_prompts(self):
        """Persona system_prompt_modifier should work with build_system_prompt."""
        from events_api.persona_config import get_persona
        from events_api.llm_prompts import build_system_prompt, get_prompt

        for persona_id in ["direct_challenger", "gentle_explorer", "neutral_observer"]:
            persona = get_persona(persona_id)
            base = get_prompt("signal_generation")
            full = build_system_prompt(base, persona_modifier=persona.system_prompt_modifier)
            self.assertIn("Persona", full)
            self.assertIn(persona.system_prompt_modifier, full)


class TestGraphAgentIntegration(unittest.TestCase):
    """Test that graph agent orchestrates cluster + insight engines correctly."""

    def test_agent_description_generators_handle_all_cluster_types(self):
        """Graph agent should generate descriptions for every cluster type."""
        from events_api.graph_agent import GraphAgent
        from events_api.cluster_engine import CLUSTER_TYPES

        agent = GraphAgent()

        for ctype in CLUSTER_TYPES:
            desc = agent._describe_new_cluster({
                "cluster_type": ctype,
                "shared_coordinates": {"context": ["work"]},
            })
            self.assertIn("New cluster detected", desc)
            self.assertIsInstance(desc, str)
            self.assertGreater(len(desc), 20)

    def test_agent_handles_all_status_transitions(self):
        """Graph agent should describe all lifecycle transitions."""
        from events_api.graph_agent import GraphAgent

        agent = GraphAgent()
        transitions = [
            ("active", "weakening"),
            ("weakening", "dissolved"),
            ("dissolved", "active"),
            ("active", "dissolved"),
        ]

        for old, new in transitions:
            desc = agent._describe_trajectory_shift({
                "cluster_id": "CLU-test-001",
                "old_status": old,
                "new_status": new,
                "strength": 0.3,
            })
            self.assertIsInstance(desc, str)
            self.assertGreater(len(desc), 10)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: API Contract Compliance
# ══════════════════════════════════════════════════════════════════════════════


class TestLabelToKindMapping(unittest.TestCase):
    """Verify that all graph model labels map to API kinds per the contract."""

    LABEL_TO_KIND = {
        "Signal": "signal",
        "Pattern": "pattern",
        "Cluster": "cluster",
        "Emotion": "emotion",
        "Person": "person",
        "ContextNode": "context",
        "ActionNode": "action",
        "Behavior": "behavior",
        "Outcome": "outcome",
    }

    def test_all_contract_labels_have_graph_models(self):
        """Every API kind in the contract should have a corresponding graph model."""
        import events_api.graph_models as gm

        for label in self.LABEL_TO_KIND:
            self.assertTrue(
                hasattr(gm, label),
                f"Graph model missing for contract label: {label}"
            )

    def test_all_graph_models_have_uid(self):
        """Every mapped model must have a uid property (contract requires stable IDs)."""
        import events_api.graph_models as gm

        for label in self.LABEL_TO_KIND:
            model_class = getattr(gm, label)
            self.assertTrue(
                hasattr(model_class, "uid"),
                f"{label} is missing uid property"
            )

    def test_all_graph_models_have_workspace_scoping(self):
        """Every mapped model must have workspace_id for multi-tenancy."""
        import events_api.graph_models as gm

        for label in self.LABEL_TO_KIND:
            model_class = getattr(gm, label)
            self.assertTrue(
                hasattr(model_class, "workspace_id"),
                f"{label} is missing workspace_id property"
            )

    def test_kind_values_are_lowercase_strings(self):
        """Contract requires kinds to be lowercase strings."""
        for label, kind in self.LABEL_TO_KIND.items():
            self.assertEqual(kind, kind.lower())
            self.assertRegex(kind, r'^[a-z_]+$')


class TestRelToKindMapping(unittest.TestCase):
    """Verify relationship types match the contract."""

    REL_TO_KIND = {
        "SOURCED_FROM": "shows_pattern",
        "MEMBER_OF": "shows_pattern",
        "EXPRESSES_EMOTION": "expresses_emotion",
        "PARTICIPANT_IN": "involves_person",
        "IN_CONTEXT": "in_context",
        "INVOLVES_ACTION": "involves_action",
        "DERIVED_FROM": "derived_from",
        "TRIGGERED_BY": "triggered_by",
        "LED_TO": "led_to",
        "SHOWS_BEHAVIOR": "shows_behavior",
    }

    def test_all_contract_relationships_exist_in_models(self):
        """Every relationship type in the contract should exist in graph models."""
        import events_api.graph_models as gm
        import inspect

        # Collect all relationship type strings from the models module
        source = inspect.getsource(gm)

        for rel_type in self.REL_TO_KIND:
            self.assertIn(
                f'"{rel_type}"',
                source,
                f"Relationship type {rel_type} not found in graph_models.py"
            )

    def test_edge_kind_values_are_lowercase(self):
        """Contract requires edge kinds to be lowercase strings."""
        for rel, kind in self.REL_TO_KIND.items():
            self.assertEqual(kind, kind.lower())
            self.assertRegex(kind, r'^[a-z_]+$')


class TestSignalNodeContractCompliance(unittest.TestCase):
    """Verify Signal node has all properties the contract requires."""

    REQUIRED_SIGNAL_PROPERTIES = [
        "uid", "signal_address", "confidence_score", "provenance",
        "created_at",
    ]

    REQUIRED_SIGNAL_RELATIONSHIPS = [
        "context",       # IN_CONTEXT → ContextNode
        "action",        # INVOLVES_ACTION → ActionNode
        "participants",  # PARTICIPANT_IN ← Person
    ]

    def test_signal_has_required_properties(self):
        from events_api.graph_models import Signal

        for prop in self.REQUIRED_SIGNAL_PROPERTIES:
            self.assertTrue(
                hasattr(Signal, prop),
                f"Signal missing required property: {prop}"
            )

    def test_signal_has_coordinate_relationships(self):
        from events_api.graph_models import Signal

        # Check relationship definitions exist (names match graph_models.py)
        self.assertTrue(hasattr(Signal, "context"))
        self.assertTrue(hasattr(Signal, "action"))
        self.assertTrue(hasattr(Signal, "participants"))

    def test_signal_has_emotion_relationships(self):
        from events_api.graph_models import Signal

        self.assertTrue(hasattr(Signal, "emotions_expressed"))


class TestClusterNodeContractCompliance(unittest.TestCase):
    """Verify Cluster node matches contract expectations."""

    def test_cluster_has_required_properties(self):
        from events_api.graph_models import Cluster

        required = [
            "uid", "cluster_id", "cluster_type",
            "shared_coordinates", "divergent_dimensions",
            "strength", "trajectory_history",
            "member_count", "status", "created_at",
        ]
        for prop in required:
            self.assertTrue(
                hasattr(Cluster, prop),
                f"Cluster missing required property: {prop}"
            )

    def test_cluster_has_member_relationship(self):
        from events_api.graph_models import Cluster

        self.assertTrue(hasattr(Cluster, "member_signals"))


class TestInsightReflectionContractCompliance(unittest.TestCase):
    """Verify Insight and Reflection nodes match contract."""

    def test_insight_has_required_properties(self):
        from events_api.graph_models import Insight

        required = [
            "uid", "reasoning_text", "persona",
            "confidence", "validation_status", "generated_at",
        ]
        for prop in required:
            self.assertTrue(
                hasattr(Insight, prop),
                f"Insight missing required property: {prop}"
            )

    def test_reflection_has_required_properties(self):
        from events_api.graph_models import Reflection

        required = ["uid", "text", "reflection_type", "created_at"]
        for prop in required:
            self.assertTrue(
                hasattr(Reflection, prop),
                f"Reflection missing required property: {prop}"
            )

    def test_pending_insight_has_required_properties(self):
        from events_api.graph_models import PendingInsight

        required = [
            "uid", "detection_type", "description",
            "confidence", "status", "created_at", "expires_at",
        ]
        for prop in required:
            self.assertTrue(
                hasattr(PendingInsight, prop),
                f"PendingInsight missing required property: {prop}"
            )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: IP Zone Compliance
# ══════════════════════════════════════════════════════════════════════════════


class TestIPZoneCompliance(unittest.TestCase):
    """Verify that proprietary logic stays in Python, not in Neo4j."""

    def test_no_stored_procedures_in_codebase(self):
        """No code should create Neo4j stored procedures."""
        import events_api.signal_engine as se
        import events_api.cluster_engine as ce
        import events_api.insight_engine as ie
        import events_api.graph_agent as ga
        import inspect

        for module in [se, ce, ie, ga]:
            source = inspect.getsource(module)
            self.assertNotIn("db.createProcedure", source)
            self.assertNotIn("CALL apoc.", source)
            self.assertNotIn("CALL gds.", source)
            self.assertNotIn("ai.embed", source)

    def test_no_gds_projections(self):
        """No code should use Neo4j GDS graph projections."""
        import events_api.cluster_engine as ce
        import inspect

        source = inspect.getsource(ce)
        self.assertNotIn("gds.graph.project", source)
        self.assertNotIn("gds.graph.create", source)
        self.assertNotIn("CALL gds", source)

    def test_embeddings_computed_in_python(self):
        """Embeddings must be computed in Python, not via Neo4j ai.embed()."""
        from events_api.llm_client import compute_text_embedding

        # Should work without any Neo4j connection
        embedding = compute_text_embedding("test text")
        self.assertIsInstance(embedding, list)
        self.assertGreater(len(embedding), 0)

    def test_all_cypher_in_python_files(self):
        """Verify Cypher queries are constructed in Python files."""
        import events_api.signal_engine as se
        import events_api.cluster_engine as ce
        import events_api.insight_engine as ie
        import events_api.graph_agent as ga
        import inspect

        for module in [se, ce, ie, ga]:
            source = inspect.getsource(module)
            # If module uses Cypher, it should go through db.cypher_query
            if "cypher_query" in source:
                self.assertIn("db.cypher_query", source)
                # Should NOT store Cypher externally
                self.assertNotIn("CALL db.", source.replace("CALL db.cypher", ""))

    def test_no_aura_agent_references(self):
        """No code should reference Neo4j Aura Agent."""
        import events_api.signal_engine as se
        import events_api.cluster_engine as ce
        import events_api.insight_engine as ie
        import events_api.graph_agent as ga
        import inspect

        for module in [se, ce, ie, ga]:
            source = inspect.getsource(module)
            self.assertNotIn("aura_agent", source.lower())
            self.assertNotIn("AuraAgent", source)

    def test_zone2_contract_has_no_zone1_details(self):
        """The Reflect Graph Contract should not contain Zone 1 algorithmic details."""
        contract_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "..",
            "Reflect_Graph_Contract_v1.md"
        )
        if not os.path.exists(contract_path):
            self.skipTest("Contract file not found at expected path")

        with open(contract_path) as f:
            content = f.read()

        # Zone 1 terms that should NOT appear in Zone 2 contract
        zone1_terms = [
            "coordinate_overlap",
            "resolution_completeness",
            "STRENGTH_DECAY_RATE",
            "DISSOLUTION_THRESHOLD",
            "compute_cluster_strength",
            "observation_bias_aggressiveness",
            "context_token_budget",
            "rumination_amplification",
            "_detect_emotions_keyword",
            "cosine_similarity",
        ]

        for term in zone1_terms:
            self.assertNotIn(
                term, content,
                f"Zone 1 term '{term}' found in Zone 2 contract"
            )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4: Module Interface Stability
# ══════════════════════════════════════════════════════════════════════════════


class TestModulePublicInterfaces(unittest.TestCase):
    """Verify all modules export their expected public interfaces."""

    def test_coordinate_system_exports(self):
        from events_api.coordinate_system import (
            COORDINATE_NAMES,
            WILDCARD,
            parse_signal_address,
            build_signal_address,
            detect_wildcards,
            is_fully_resolved,
            resolution_completeness,
            coordinate_overlap,
            CoordinateSystem,
        )
        self.assertIsNotNone(CoordinateSystem)

    def test_signal_engine_exports(self):
        from events_api.signal_engine import (
            SignalGenerator,
            SignalEngine,
            generate_signals,
        )
        self.assertIs(SignalEngine, SignalGenerator)

    def test_cluster_engine_exports(self):
        from events_api.cluster_engine import (
            CLUSTER_TYPES,
            ClusterEngine,
            classify_cluster_type,
            compute_cluster_strength,
            determine_cluster_status,
            detect_clusters,
        )
        self.assertEqual(len(CLUSTER_TYPES), 6)

    def test_insight_engine_exports(self):
        from events_api.insight_engine import (
            InsightEngine,
            create_insight,
            create_reflection,
        )
        self.assertIsNotNone(InsightEngine)

    def test_graph_agent_exports(self):
        from events_api.graph_agent import (
            GraphAgent,
            run_detection_cycle,
        )
        self.assertIsNotNone(GraphAgent)

    def test_persona_config_exports(self):
        from events_api.persona_config import (
            PERSONAS,
            DEFAULT_PERSONA,
            get_persona,
            list_personas,
        )
        self.assertEqual(len(PERSONAS), 3)

    def test_llm_prompts_exports(self):
        from events_api.llm_prompts import (
            V2_PROMPTS,
            V3_PROMPTS,
            ALL_PROMPTS,
            get_prompt,
            build_system_prompt,
        )
        self.assertEqual(len(ALL_PROMPTS), 11)

    def test_context_assembly_exports(self):
        from events_api.context_assembly import (
            ContextAssembler,
            assemble_context,
        )
        self.assertIsNotNone(ContextAssembler)

    def test_llm_client_v3_exports(self):
        from events_api.llm_client import (
            generate_signal_from_message,
            assess_signal_confidence,
            compute_text_embedding,
            EMBEDDING_DIMENSION,
        )
        self.assertEqual(EMBEDDING_DIMENSION, 256)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5: End-to-End Pipeline Simulation (No Graph)
# ══════════════════════════════════════════════════════════════════════════════


class TestEndToEndPipelineNoGraph(unittest.TestCase):
    """
    Simulate the full pipeline without Neo4j.

    Tests the data flow: message → entity extraction → signal generation
    → coordinate parsing → cluster classification → insight description.
    """

    def test_full_pipeline_keyword_path(self):
        """Walk a message through every module using keyword fallback."""
        # Step 1: Entity extraction
        from events_api.context_assembly import ContextAssembler, assemble_context

        assembler = ContextAssembler()
        entities = assembler.extract_entities(
            "Yesterday my manager dismissed my idea at work and I felt humiliated"
        )

        self.assertGreater(len(entities["persons"]), 0)
        self.assertGreater(len(entities["contexts"]), 0)
        self.assertGreater(len(entities["temporal"]), 0)
        self.assertGreater(len(entities["actions"]), 0)

        # Step 2: Signal generation (keyword fallback)
        from events_api.signal_engine import SignalGenerator

        gen = SignalGenerator(use_llm=False)
        result = gen.generate_from_message(
            "Yesterday my manager dismissed my idea at work and I felt humiliated",
            participants=["manager"],
        )

        signals = result["signals"]
        self.assertGreater(len(signals), 0)
        signal = signals[0]

        # Step 3: Coordinate parsing
        from events_api.coordinate_system import parse_signal_address, detect_wildcards

        parsed = parse_signal_address(signal["signal_address"])
        wildcards = detect_wildcards(parsed)
        self.assertIsInstance(wildcards, list)

        # Step 4: Embedding
        from events_api.llm_client import compute_text_embedding

        embedding = compute_text_embedding(
            "Yesterday my manager dismissed my idea at work"
        )
        self.assertEqual(len(embedding), 256)

        # Step 5: Cluster classification (simulate with two signals)
        from events_api.cluster_engine import (
            _extract_signal_coordinates,
            _compute_shared_divergent,
            classify_cluster_type,
            compute_cluster_strength,
        )

        coords1 = _extract_signal_coordinates(signal)
        # Simulate a second signal with same context but different emotion
        coords2 = {
            "context": coords1["context"],
            "person": coords1["person"],
            "action": "accusation",
            "temporal": "today",
            "emotions": ["anger"],
        }

        shared, divergent, _, _ = _compute_shared_divergent([coords1, coords2])

        if shared and divergent:
            ctype = classify_cluster_type(shared, divergent)
            self.assertIn(ctype, [
                "same_time_diff_emotion", "same_person_diff_time",
                "same_context_diff_person", "same_action_diff_everything",
                "same_emotion_diff_source", "cross_dimensional",
            ])

        # Step 6: Strength computation
        strength = compute_cluster_strength(member_count=3, avg_confidence=0.7)
        self.assertGreater(strength, 0.0)
        self.assertLessEqual(strength, 1.0)

        # Step 7: Context assembly for LLM
        packet = assemble_context(entities)
        self.assertIsInstance(packet, str)

        # Step 8: Prompt assembly
        from events_api.llm_prompts import build_system_prompt, get_prompt
        from events_api.persona_config import get_persona

        persona = get_persona("neutral_observer")
        base = get_prompt("signal_generation")
        full_prompt = build_system_prompt(
            base,
            persona_modifier=persona.system_prompt_modifier,
            context_packet=packet,
        )
        self.assertIn("Persona", full_prompt)
        self.assertIn("Context from Graph", full_prompt)

        # Step 9: Graph agent description
        from events_api.graph_agent import GraphAgent

        agent = GraphAgent()
        desc = agent._describe_new_cluster({
            "cluster_type": "same_context_diff_person",
            "shared_coordinates": {"context": ["work"]},
        })
        self.assertIn("New cluster detected", desc)

    def test_graceful_degradation_chain(self):
        """When LLM is unavailable, the full chain should still work."""
        from events_api.signal_engine import SignalGenerator

        # No LLM client — should use keyword fallback
        gen = SignalGenerator(use_llm=True, llm_client=None)
        result = gen.generate_from_message("I'm anxious about the meeting tomorrow")

        signals = result["signals"]
        self.assertGreater(len(signals), 0)

        # Verify fallback provenance
        self.assertEqual(signals[0]["provenance"], "system_detected")

        # Verify the signal is still usable by cluster engine
        from events_api.cluster_engine import _extract_signal_coordinates

        coords = _extract_signal_coordinates(signals[0])
        self.assertIsInstance(coords["emotions"], list)
        self.assertGreater(len(coords["emotions"]), 0)


class TestIndexDefinitionCompleteness(unittest.TestCase):
    """Verify all indexed properties match the Storage Model document."""

    def test_startup_indexes_defined(self):
        from events_api.graph_models import STARTUP_INDEXES

        self.assertIsInstance(STARTUP_INDEXES, list)
        self.assertGreater(len(STARTUP_INDEXES), 10)

    def test_key_indexes_present(self):
        from events_api.graph_models import STARTUP_INDEXES

        index_text = " ".join(STARTUP_INDEXES)

        # Core indexes from storage model
        expected_patterns = [
            "Signal", "Conversation", "Person", "Cluster",
            "ContextNode", "ActionNode", "TemporalNode",
            "Emotion", "Behavior", "Outcome",
            "workspace_id",
        ]

        for pattern in expected_patterns:
            self.assertIn(
                pattern, index_text,
                f"Expected index pattern '{pattern}' not found in STARTUP_INDEXES"
            )


class TestEmbeddingConsistency(unittest.TestCase):
    """Verify embedding behavior matches pipeline expectations."""

    def test_similar_texts_have_different_embeddings(self):
        """Hash embeddings should differ for semantically different texts."""
        from events_api.llm_client import compute_text_embedding

        e1 = compute_text_embedding("I am angry at my boss")
        e2 = compute_text_embedding("I am happy with my friend")
        self.assertNotEqual(e1, e2)

    def test_embedding_dimension_matches_constant(self):
        from events_api.llm_client import compute_text_embedding, EMBEDDING_DIMENSION

        embedding = compute_text_embedding("any text")
        self.assertEqual(len(embedding), EMBEDDING_DIMENSION)

    def test_empty_text_produces_embedding(self):
        from events_api.llm_client import compute_text_embedding

        embedding = compute_text_embedding("")
        self.assertEqual(len(embedding), 256)
        # Empty text should still produce a valid vector
        self.assertTrue(all(-1.0 <= v <= 1.0 for v in embedding))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6: User Graph Sync
# ══════════════════════════════════════════════════════════════════════════════


class TestGraphSyncModule(unittest.TestCase):
    """Verify the graph_sync module exists and has correct interfaces."""

    def test_graph_sync_exports(self):
        from events_api.graph_sync import (
            delete_user_graph_data,
            delete_workspace_graph_data,
            ensure_user_profile_in_graph,
            link_user_to_conversations,
            full_user_graph_sync,
        )
        self.assertTrue(callable(delete_workspace_graph_data))
        self.assertTrue(callable(delete_user_graph_data))
        self.assertTrue(callable(ensure_user_profile_in_graph))
        self.assertTrue(callable(link_user_to_conversations))
        self.assertTrue(callable(full_user_graph_sync))

    def test_ensure_user_profile_graceful_without_neo4j(self):
        """Sync should fail gracefully when Neo4j is unavailable."""
        from events_api.graph_sync import ensure_user_profile_in_graph

        mock_user = MagicMock()
        mock_user.id = 1
        mock_user.username = "testuser"
        mock_user.email = "test@example.com"

        mock_workspace = MagicMock()
        mock_workspace.id = "ws-123"

        # Without a live Neo4j connection, this should return None, not crash
        result = ensure_user_profile_in_graph(mock_user, mock_workspace)
        # Result may be None (no Neo4j) or a dict (if Neo4j is running)
        self.assertTrue(result is None or isinstance(result, dict))

    def test_full_sync_graceful_without_neo4j(self):
        """Full sync should not raise even without Neo4j."""
        from events_api.graph_sync import full_user_graph_sync

        mock_user = MagicMock()
        mock_user.id = 2
        mock_user.username = "testuser2"
        mock_user.email = "test2@example.com"

        mock_workspace = MagicMock()
        mock_workspace.id = "ws-456"

        # Should not raise
        result = full_user_graph_sync(mock_user, mock_workspace)
        self.assertTrue(result is None or isinstance(result, dict))

    def test_link_functions_graceful_without_neo4j(self):
        """Link functions should return 0 on failure, not crash."""
        from events_api.graph_sync import link_user_to_conversations

        mock_user = MagicMock()
        mock_user.id = 3
        mock_user.username = "testuser3"

        mock_workspace = MagicMock()
        mock_workspace.id = "ws-789"

        convs = link_user_to_conversations(mock_user, mock_workspace)
        self.assertEqual(convs, 0)

    @patch("events_api.graph_sync.cypher_query")
    def test_delete_workspace_graph_data_counts_then_deletes(self, mock_cypher_query):
        from events_api.graph_sync import delete_workspace_graph_data

        mock_cypher_query.side_effect = [
            ([[5]], ["deleted_nodes"]),
            ([], []),
        ]

        result = delete_workspace_graph_data(workspace_id="ws-123")

        self.assertEqual(result["deleted_nodes"], 5)
        self.assertEqual(mock_cypher_query.call_count, 2)

    @patch("events_api.graph_sync.cypher_query")
    def test_delete_user_graph_data_deletes_by_workspace_and_owner(self, mock_cypher_query):
        from events_api.graph_sync import delete_user_graph_data

        mock_cypher_query.side_effect = [
            ([[9]], ["deleted_nodes"]),
            ([], []),
        ]

        result = delete_user_graph_data(workspace_ids=["ws-1", "ws-2"], owner_user_id=42)

        self.assertEqual(result["deleted_nodes"], 9)
        count_query = mock_cypher_query.call_args_list[0].args[0]
        self.assertIn("n.workspace_id IN $workspace_ids", count_query)
        self.assertIn("n.owner_user_id = $owner_user_id", count_query)


class TestAuthViewsSyncIntegration(unittest.TestCase):
    """Verify auth views contain the graph sync calls (source-level check)."""

    def test_signup_view_contains_sync(self):
        """SignupView source should call graph sync on signup."""
        auth_views_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "auth_views.py",
        )
        with open(auth_views_path, "r") as f:
            source = f.read()
        self.assertIn("full_user_graph_sync", source)
        self.assertIn("graph_sync", source)

    def test_auth_views_have_delete_endpoints(self):
        auth_views_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "auth_views.py",
        )
        with open(auth_views_path, "r") as f:
            source = f.read()
        self.assertIn("class DeleteDataView", source)
        self.assertIn("class DeleteAccountView", source)
        self.assertIn("class ExportConversationsView", source)
        self.assertIn("class ExportMemoryGraphView", source)
        self.assertIn("delete_workspace_graph_data", source)
        self.assertIn("delete_user_graph_data", source)
        self.assertIn("build_conversations_export", source)
        self.assertIn("build_memory_graph_svg", source)

    def test_login_view_contains_sync(self):
        """LoginView source should call graph sync on login."""
        auth_views_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "auth_views.py",
        )
        with open(auth_views_path, "r") as f:
            source = f.read()
        # Verify sync is called in the login section (after "class LoginView")
        login_section = source[source.index("class LoginView"):]
        self.assertIn("full_user_graph_sync", login_section)

    def test_auth_urls_have_export_routes(self):
        urls_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "urls.py",
        )
        with open(urls_path, "r") as f:
            source = f.read()
        self.assertIn('path("auth/export/conversations/"', source)
        self.assertIn('path("auth/export/memory-graph/"', source)


class TestUserProfileModel(unittest.TestCase):
    """Verify UserProfile graph model has required fields."""

    def test_userprofile_has_identity_fields(self):
        from events_api.graph_models import UserProfile

        self.assertTrue(hasattr(UserProfile, "workspace_id"))
        self.assertTrue(hasattr(UserProfile, "owner_user_id"))
        self.assertTrue(hasattr(UserProfile, "username"))
        self.assertTrue(hasattr(UserProfile, "email"))

    def test_userprofile_has_relationships(self):
        from events_api.graph_models import UserProfile

        self.assertTrue(hasattr(UserProfile, "conversations"))

    def test_userprofile_has_stats_fields(self):
        from events_api.graph_models import UserProfile

        self.assertTrue(hasattr(UserProfile, "total_messages"))
        self.assertTrue(hasattr(UserProfile, "avg_message_length"))
        self.assertTrue(hasattr(UserProfile, "top_topics"))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7: Enriched Prompt Assembly (LiveConversationView Pipeline)
# ══════════════════════════════════════════════════════════════════════════════


class TestEnrichedPromptAssemblyPipeline(unittest.TestCase):
    """
    Verify that the context-enriched prompt assembly pipeline works
    end-to-end: entity extraction → context packet → persona → build_system_prompt.

    These tests validate the pipeline that LiveConversationView now uses
    instead of static system prompts.
    """

    def test_entity_extraction_feeds_context_assembly(self):
        """Entities extracted from a message produce a meaningful context packet."""
        from events_api.context_assembly import ContextAssembler, assemble_context

        assembler = ContextAssembler()
        entities = assembler.extract_entities(
            "My boss dismissed my idea in the meeting yesterday"
        )

        packet = assemble_context(entities)

        # Should contain extracted entity information
        self.assertIn("manager", packet)  # boss → manager role
        self.assertIn("work", packet)  # meeting → work context
        self.assertIn("dismissal", packet)  # dismissed → dismissal action
        self.assertIn("yesterday", packet)  # temporal reference

    def test_persona_modifier_injected_into_system_prompt(self):
        """Each persona's system_prompt_modifier appears in the assembled prompt."""
        from events_api.llm_prompts import build_system_prompt
        from events_api.persona_config import get_persona, PERSONAS

        base = "You are ThriveSight."

        for persona_id in PERSONAS:
            persona = get_persona(persona_id)
            prompt = build_system_prompt(
                base_prompt=base,
                persona_modifier=persona.system_prompt_modifier,
                context_packet="",
            )
            self.assertIn("## Persona", prompt)
            self.assertIn(persona.system_prompt_modifier, prompt)

    def test_context_packet_injected_into_system_prompt(self):
        """The context packet appears in the 'Context from Graph' section."""
        from events_api.context_assembly import ContextAssembler, assemble_context
        from events_api.llm_prompts import build_system_prompt
        from events_api.persona_config import get_persona

        assembler = ContextAssembler()
        entities = assembler.extract_entities("I felt ignored by my partner last week")
        packet = assemble_context(entities)

        persona = get_persona("neutral_observer")
        prompt = build_system_prompt(
            base_prompt="You are ThriveSight.",
            persona_modifier=persona.system_prompt_modifier,
            context_packet=packet,
        )

        self.assertIn("## Context from Graph", prompt)
        self.assertIn("partner", prompt)

    def test_full_pipeline_produces_three_section_prompt(self):
        """The assembled prompt has base + Persona + Context sections."""
        from events_api.context_assembly import ContextAssembler, assemble_context
        from events_api.llm_prompts import build_system_prompt
        from events_api.persona_config import get_persona

        assembler = ContextAssembler()
        entities = assembler.extract_entities(
            "My colleague interrupted me during the standup this morning"
        )
        packet = assemble_context(entities)
        persona = get_persona("direct_challenger")

        prompt = build_system_prompt(
            base_prompt="You are ThriveSight, a warm AI companion.",
            persona_modifier=persona.system_prompt_modifier,
            context_packet=packet,
        )

        # Three distinct sections
        self.assertIn("ThriveSight", prompt)  # base prompt
        self.assertIn("## Persona", prompt)  # persona section
        self.assertIn("## Context from Graph", prompt)  # context section

        # Persona-specific content
        self.assertIn("direct", prompt.lower())  # direct_challenger tone

        # Entity-derived context
        self.assertIn("colleague", prompt)
        self.assertIn("interruption", prompt)
        self.assertIn("work", prompt)

    def test_empty_message_produces_valid_prompt(self):
        """Even with no entities extracted, the pipeline produces a valid prompt."""
        from events_api.context_assembly import ContextAssembler, assemble_context
        from events_api.llm_prompts import build_system_prompt
        from events_api.persona_config import get_persona

        assembler = ContextAssembler()
        entities = assembler.extract_entities("I just feel off today")
        packet = assemble_context(entities)

        persona = get_persona("gentle_explorer")
        prompt = build_system_prompt(
            base_prompt="You are ThriveSight.",
            persona_modifier=persona.system_prompt_modifier,
            context_packet=packet,
        )

        # Should still have persona section
        self.assertIn("## Persona", prompt)
        self.assertIsInstance(prompt, str)
        self.assertGreater(len(prompt), 50)

    def test_persona_fallback_on_invalid_id(self):
        """Invalid persona ID should raise KeyError (LiveConversationView catches this)."""
        from events_api.persona_config import get_persona

        with self.assertRaises(KeyError):
            get_persona("nonexistent_persona")

    def test_persona_overrides_applied(self):
        """Persona overrides should modify configuration values."""
        from events_api.persona_config import get_persona

        persona = get_persona("neutral_observer", overrides={
            "context_depth": 25,
            "confidence_threshold": 0.9,
        })

        self.assertEqual(persona.context_depth, 25)
        self.assertAlmostEqual(persona.confidence_threshold, 0.9)
        # Other fields retain defaults
        self.assertEqual(persona.observation_bias_aggressiveness, "medium")

    def test_all_three_personas_produce_different_prompts(self):
        """Each persona should produce a distinguishably different system prompt."""
        from events_api.context_assembly import ContextAssembler, assemble_context
        from events_api.llm_prompts import build_system_prompt
        from events_api.persona_config import get_persona, PERSONAS

        assembler = ContextAssembler()
        entities = assembler.extract_entities("My manager criticized my work today")
        packet = assemble_context(entities)
        base = "You are ThriveSight."

        prompts = {}
        for persona_id in PERSONAS:
            persona = get_persona(persona_id)
            prompts[persona_id] = build_system_prompt(
                base_prompt=base,
                persona_modifier=persona.system_prompt_modifier,
                context_packet=packet,
            )

        # All three prompts should be different
        self.assertNotEqual(prompts["direct_challenger"], prompts["gentle_explorer"])
        self.assertNotEqual(prompts["gentle_explorer"], prompts["neutral_observer"])
        self.assertNotEqual(prompts["direct_challenger"], prompts["neutral_observer"])


class TestLiveConversationServiceSourceWiring(unittest.TestCase):
    """Source-level verification that live conversation orchestration lives in a service."""

    def _get_service_source(self):
        service_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "live_conversation_service.py",
        )
        with open(service_path, "r") as f:
            return f.read()

    def _get_view_source(self):
        views_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "views.py",
        )
        with open(views_path, "r") as f:
            return f.read()

    def test_service_defines_step_methods(self):
        source = self._get_service_source()
        for method_name in [
            "def build_prompt_context",
            "def generate_reply",
            "def persist_conversation",
            "def generate_and_link_signals",
            "def write_pipeline_trace",
            "def run",
        ]:
            self.assertIn(method_name, source)

    def test_service_owns_context_assembly_and_persona_logic(self):
        source = self._get_service_source()
        self.assertIn("ContextAssembler", source)
        self.assertIn("assemble_context", source)
        self.assertIn("build_system_prompt", source)
        self.assertIn("get_persona", source)
        self.assertIn("DEFAULT_PERSONA", source)

    def test_service_uses_shared_llm_client(self):
        source = self._get_service_source()
        self.assertIn("generate_conversation_reply", source)
        self.assertNotIn("anthropic.Anthropic", source)
        self.assertNotIn("client.messages.create(", source)

    def test_view_delegates_to_service(self):
        source = self._get_view_source()
        self.assertIn("LiveConversationService", source)
        self.assertNotIn("ContextAssembler", source)
        self.assertNotIn("assemble_context", source)
        self.assertNotIn("build_system_prompt", source)

    def test_service_includes_context_assembly_metadata(self):
        source = self._get_service_source()
        self.assertIn("context_assembly", source)
        self.assertIn("context_packet_length", source)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8: Insights Tab — Knowledge Graph Visualization
# ══════════════════════════════════════════════════════════════════════════════


class TestPipelineTraceModel(unittest.TestCase):
    """Verify PipelineTrace node is properly defined in graph_models.py."""

    def test_pipeline_trace_exists(self):
        from events_api.graph_models import PipelineTrace
        self.assertTrue(hasattr(PipelineTrace, "uid"))
        self.assertTrue(hasattr(PipelineTrace, "workspace_id"))

    def test_pipeline_trace_has_observability_fields(self):
        from events_api.graph_models import PipelineTrace
        required_fields = [
            "entities_extracted",
            "persona_used",
            "context_packet_summary",
            "signals_referenced",
            "clusters_referenced",
            "token_count",
            "created_at",
        ]
        for field in required_fields:
            self.assertTrue(
                hasattr(PipelineTrace, field),
                f"PipelineTrace missing required field: {field}",
            )

    def test_pipeline_trace_has_traces_relationship(self):
        from events_api.graph_models import PipelineTrace
        self.assertTrue(
            hasattr(PipelineTrace, "for_turn"),
            "PipelineTrace should have for_turn (TRACES) relationship to UserTurn",
        )

    def test_pipeline_trace_index_defined(self):
        from events_api.graph_models import STARTUP_INDEXES
        pipeline_trace_indexes = [
            idx for idx in STARTUP_INDEXES if "PipelineTrace" in idx
        ]
        self.assertTrue(
            len(pipeline_trace_indexes) >= 1,
            "At least one PipelineTrace index should be defined in STARTUP_INDEXES",
        )


class TestWritePipelineTraceFunction(unittest.TestCase):
    """Verify write_pipeline_trace function exists and has correct signature."""

    def test_write_pipeline_trace_exists(self):
        from events_api.live_graph import write_pipeline_trace
        self.assertTrue(callable(write_pipeline_trace))

    def test_write_pipeline_trace_accepts_required_params(self):
        """Verify the function signature accepts all expected keyword arguments."""
        import inspect
        from events_api.live_graph import write_pipeline_trace

        sig = inspect.signature(write_pipeline_trace)
        required_params = [
            "workspace_id",
            "conversation_id",
            "entities",
            "persona",
            "context_summary",
            "signals_referenced",
            "clusters_referenced",
            "token_count",
        ]
        for param_name in required_params:
            self.assertIn(
                param_name,
                sig.parameters,
                f"write_pipeline_trace missing parameter: {param_name}",
            )


class TestWriteLiveConversationToGraph(unittest.TestCase):
    """Verify live conversation writes validate that turns were actually stored."""

    @patch("events_api.live_graph.extract_live_topics", return_value=[])
    @patch("events_api.live_graph._run_write_queries")
    def test_write_live_conversation_returns_verified_success(self, mock_run_queries, _mock_topics):
        from events_api.live_graph import write_live_conversation_to_graph

        mock_run_queries.return_value = [
            ([], []),
            ([], []),
            ([], []),
            ([], []),
            ([], []),
            ([], []),
            ([[ "Live: test", 2, 2 ]], ["title", "turn_count", "stored_turns"]),
        ]

        result = write_live_conversation_to_graph(
            conversation_id="abc123",
            message="test message",
            ai_response="test response",
            workspace_id="ws-1",
            owner_user_id=42,
        )

        self.assertTrue(result["updated"])
        self.assertEqual(result["stored_turns"], 2)
        self.assertEqual(result["conversation_title"], "Live: test")

    @patch("events_api.live_graph.extract_live_topics", return_value=[])
    @patch("events_api.live_graph._run_write_queries")
    def test_write_live_conversation_returns_failure_when_turns_missing(self, mock_run_queries, _mock_topics):
        from events_api.live_graph import write_live_conversation_to_graph

        mock_run_queries.return_value = [
            ([], []),
            ([], []),
            ([], []),
            ([], []),
            ([], []),
            ([], []),
            ([[ "Live: test", 0, 0 ]], ["title", "turn_count", "stored_turns"]),
        ]

        result = write_live_conversation_to_graph(
            conversation_id="abc123",
            message="test message",
            ai_response="test response",
            workspace_id="ws-1",
            owner_user_id=42,
        )

        self.assertFalse(result["updated"])
        self.assertEqual(result["stored_turns"], 0)


class TestGraphIdentityCanonicalization(unittest.TestCase):
    """Verify graph writes reserve UserProfile as the canonical self identity."""

    def _read_backend_file(self, relative_path):
        base = os.path.dirname(os.path.dirname(__file__))
        path = os.path.join(base, relative_path)
        with open(path, "r") as f:
            return f.read()

    def test_normalize_graph_person_name_collapses_self_aliases(self):
        from events_api.identity import build_self_alias_names, normalize_graph_person_name

        for alias in ["self", "user", "me", "myself", "i", "bronzeage"]:
            self.assertIsNone(
                normalize_graph_person_name(alias, current_username="bronzeage")
            )

        self.assertIn("bronzeage", build_self_alias_names(current_username="bronzeage"))
        self.assertEqual(
            normalize_graph_person_name("Therapist", current_username="bronzeage"),
            "Therapist",
        )

    def test_signal_generator_scopes_coordinate_nodes_by_workspace(self):
        source = self._read_backend_file("signal_engine.py")
        self.assertIn("normalize_graph_person_name", source)
        self.assertIn("MERGE (p:Person {name: $name, workspace_id: $workspace_id})", source)
        self.assertIn("MERGE (c:ContextNode {name: $name, workspace_id: $workspace_id})", source)
        self.assertIn("MERGE (a:ActionNode {name: $name, workspace_id: $workspace_id})", source)
        self.assertIn("MERGE (t:TemporalNode {name: $name, workspace_id: $workspace_id})", source)
        self.assertIn("MERGE (e:Emotion {name: $name, workspace_id: $workspace_id})", source)

    def test_import_user_signals_does_not_create_self_person_node(self):
        source = self._read_backend_file("management/commands/import_user_signals.py")
        self.assertIn("normalize_graph_person_name", source)
        self.assertIn("build_self_alias_names", source)
        self.assertNotIn("people = [username]", source)
        self.assertIn("MERGE (u:UserProfile {workspace_id: $workspace_id, owner_user_id: $owner_user_id})", source)
        self.assertIn("MERGE (u)-[:OWNS_CONVERSATION]->(c)", source)
        self.assertIn("DETACH DELETE p", source)

    def test_user_profile_merges_no_longer_use_primary_slug(self):
        import_source = self._read_backend_file("import_parsers.py")
        sync_source = self._read_backend_file("graph_sync.py")
        live_source = self._read_backend_file("live_graph.py")

        self.assertNotIn("id: 'primary'", import_source)
        self.assertIn("workspace_id: $workspace_id, owner_user_id: $owner_user_id", import_source)
        self.assertIn("workspace_id: $workspace_id, owner_user_id: $owner_user_id", sync_source)
        self.assertIn("workspace_id: $workspace_id, owner_user_id: $owner_user_id", live_source)
        self.assertIn("self_aliases", sync_source)
        self.assertIn("self_aliases", live_source)

    def test_emotional_graph_reads_workspace_scoped_nodes(self):
        source = self._read_backend_file("views.py")
        self.assertIn('"(s:Signal {workspace_id: $ws})-[:EXPRESSES_EMOTION]->(n:Emotion {workspace_id: $ws})"', source)
        self.assertIn('"(s:Signal {workspace_id: $ws})-[:IN_CONTEXT]->(n:ContextNode {workspace_id: $ws})"', source)
        self.assertIn('"(s:Signal {workspace_id: $ws})-[:INVOLVES_ACTION]->(n:ActionNode {workspace_id: $ws})"', source)
        self.assertIn('"(s:Signal {workspace_id: $ws})-[:AT_TIME]->(n:TemporalNode {workspace_id: $ws})"', source)
        self.assertIn('MATCH (a:Person {{workspace_id: $ws}})-[:PARTICIPANT_IN]->(s:Signal {{workspace_id: $ws}})', source)


class TestWorkspaceGraphEndpoint(unittest.TestCase):
    """Verify workspace_graph query is registered and handler exists."""

    def _get_view_source(self):
        views_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "views.py",
        )
        with open(views_path, "r") as f:
            return f.read()

    def test_workspace_graph_registered_in_queries(self):
        source = self._get_view_source()
        self.assertIn('"workspace_graph"', source)

    def test_workspace_graph_has_handler(self):
        source = self._get_view_source()
        self.assertIn("_handle_workspace_graph", source)

    def test_handler_returns_nodes_edges_counts(self):
        """Handler method should reference nodes, edges, and counts in its return."""
        source = self._get_view_source()
        handler_start = source.find("def _handle_workspace_graph")
        self.assertNotEqual(handler_start, -1, "Handler method not found")
        # Handler is large — search enough to find the Response() at the end
        handler_section = source[handler_start:handler_start + 16000]
        self.assertIn('"nodes"', handler_section)
        self.assertIn('"edges"', handler_section)
        self.assertIn('"counts"', handler_section)

    def test_label_map_has_user_friendly_translations(self):
        source = self._get_view_source()
        # Key translations from the plan
        self.assertIn('"Signal": "Moment"', source)
        self.assertIn('"Cluster": "Recurring Pattern"', source)
        self.assertIn('"PipelineTrace": "How this was understood"', source)
        self.assertIn('"UserTurn": "Your message"', source)
        self.assertIn('"AssistantTurn": "AI response"', source)
        self.assertIn('"UserProfile": "You"', source)
        self.assertIn('"Topic": "Theme"', source)

    def test_handler_supports_filter_params(self):
        """Handler should accept node_types, since, and min_confidence filters."""
        source = self._get_view_source()
        handler_start = source.find("def _handle_workspace_graph")
        handler_section = source[handler_start:handler_start + 3000]
        self.assertIn("node_types", handler_section)
        self.assertIn("since", handler_section)
        self.assertIn("min_confidence", handler_section)


class TestPipelineTraceWiringInService(unittest.TestCase):
    """Verify the live conversation service writes PipelineTrace after conversation turns."""

    def _get_service_source(self):
        service_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "live_conversation_service.py",
        )
        with open(service_path, "r") as f:
            return f.read()

    def test_service_imports_write_pipeline_trace(self):
        source = self._get_service_source()
        self.assertIn("write_pipeline_trace", source)

    def test_pipeline_trace_write_is_non_fatal(self):
        """PipelineTrace write should be in a try/except so failures don't block conversations."""
        source = self._get_service_source()
        self.assertIn("PipelineTrace write failed (non-fatal)", source)


class TestInsightsPageFrontendWiring(unittest.TestCase):
    """Verify frontend files are wired correctly for the Insights tab."""

    def _read_frontend_file(self, relative_path):
        base = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))),
            "apps", "reflect", "src",
        )
        full_path = os.path.join(base, relative_path)
        with open(full_path, "r") as f:
            return f.read()

    def test_app_routes_to_insights_page(self):
        source = self._read_frontend_file("App.jsx")
        self.assertIn("InsightsPage", source)
        self.assertNotIn("Insights2Page", source)

    def test_api_has_fetch_workspace_graph(self):
        source = self._read_frontend_file("api.js")
        self.assertIn("fetchWorkspaceGraph", source)
        self.assertIn("workspace_graph", source)

    def test_ai_output_has_signal_address_sanitizer_and_pending_indicator(self):
        source = self._read_frontend_file("components/AIOutput.jsx")
        self.assertIn("stripSignalAddresses", source)
        self.assertIn("PendingAsterisks", source)
        self.assertIn('SA\\([^)]*\\)', source)

    def test_graph_view_model_has_workspace_builder(self):
        source = self._read_frontend_file("features/graphViewModel.js")
        self.assertIn("buildWorkspaceGraphData", source)
        self.assertIn("ALWAYS_LABEL_KINDS", source)

    def test_graph_view_model_has_new_node_types(self):
        source = self._read_frontend_file("features/graphViewModel.js")
        for kind in ["Signal", "Cluster", "PipelineTrace", "Person", "Insight"]:
            self.assertIn(
                kind,
                source,
                f"graphViewModel.js missing node type color/size: {kind}",
            )

    def test_insights_page_uses_force_graph(self):
        source = self._read_frontend_file("pages/InsightsPage.jsx")
        self.assertIn("ForceGraph2D", source)
        self.assertIn("fetchEmotionalGraph", source)

    def test_insights_sidebar_exists(self):
        source = self._read_frontend_file("components/InsightsSidebar.jsx")
        self.assertIn("InsightsSidebar", source)
        self.assertIn("insights-sidebar", source)

    def test_insights_sidebar_has_progressive_disclosure(self):
        source = self._read_frontend_file("components/InsightsSidebar.jsx")
        self.assertIn("emotional-category-group", source)
        self.assertIn("emotional-category-row", source)
        self.assertIn("emotional-category-expanded", source)

    def test_insights_sidebar_has_detail_inspector(self):
        source = self._read_frontend_file("components/InsightsSidebar.jsx")
        self.assertIn("insights-detail-card", source)
        self.assertIn("insights-node-badge", source)
        self.assertIn("insights-back-btn", source)
        self.assertIn("Recent conversations", source)
        self.assertIn("Connected signals", source)
        self.assertIn("Also appears with", source)
        self.assertIn('"workspace_id"', source)
        self.assertIn('"owner_user_id"', source)
        self.assertIn('"signal_count"', source)
        self.assertIn("formatDetailValue", source)
        self.assertIn("normalizeEmotionalNodeLabel(item.label)", source)

    def test_css_has_insights_styles(self):
        source = self._read_frontend_file("styles.css")
        expected_classes = [
            ".insights-layout",
            ".insights-sidebar",
            ".insights-canvas",
            ".insights-refresh-btn",
            ".emotional-category-group",
            ".emotional-category-row",
            ".emotional-category-expanded",
            ".emotional-item-chip",
            ".insights-detail-card",
            ".insights-node-badge",
            ".insights-detail-section",
            ".insights-detail-list",
            ".ai-output-pending-indicator",
        ]
        for cls in expected_classes:
            self.assertIn(cls, source, f"styles.css missing CSS class: {cls}")


class TestGraphValueNormalization(unittest.TestCase):
    def test_views_define_graph_value_normalizer(self):
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "views.py")
        with open(path, "r") as f:
            source = f.read()

        self.assertIn("def _normalize_graph_value", source)
        self.assertIn('getattr(value, "isoformat", None)', source)
        self.assertIn('getattr(value, "iso_format", None)', source)


# ══════════════════════════════════════════════════════════════════════════════
# Section 9 — Persistent Conversations
# ══════════════════════════════════════════════════════════════════════════════

class TestPersistentConversations(unittest.TestCase):
    """Verify full content storage, conversation listing, and history retrieval."""

    def _read_frontend_file(self, relative_path):
        base = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))),
            "apps", "reflect", "src",
        )
        full_path = os.path.join(base, relative_path)
        with open(full_path, "r") as f:
            return f.read()

    def _get_live_graph_source(self):
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "live_graph.py")
        with open(path, "r") as f:
            return f.read()

    def _get_view_source(self):
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "views.py")
        with open(path, "r") as f:
            return f.read()

    def test_full_content_stored_in_user_turn(self):
        """live_graph.py stores full content property on UserTurn nodes."""
        source = self._get_live_graph_source()
        self.assertIn("content:", source, "UserTurn Cypher should store full content")
        self.assertIn("content_preview:", source, "UserTurn Cypher should keep content_preview")

    def test_full_content_stored_in_both_turns(self):
        """live_graph.py stores full content on both UserTurn and AssistantTurn."""
        source = self._get_live_graph_source()
        content_count = source.count("content: $content")
        self.assertGreaterEqual(content_count, 2,
            "Both UserTurn and AssistantTurn should store full content")

    def test_conversation_list_query_registered(self):
        """GraphQueryView has a 'conversation_list' query."""
        source = self._get_view_source()
        self.assertIn('"conversation_list"', source)

    def test_conversation_list_returns_correct_fields(self):
        """conversation_list Cypher returns expected fields."""
        source = self._get_view_source()
        # Find the conversation_list section
        start = source.find('"conversation_list"')
        section = source[start:start + 1000]
        for field in ["conversation_id", "title", "turn_count", "last_active"]:
            self.assertIn(field, section, f"conversation_list missing field: {field}")

    def test_conversation_history_query_registered(self):
        """GraphQueryView has a 'conversation_history' query."""
        source = self._get_view_source()
        self.assertIn('"conversation_history"', source)

    def test_conversation_history_uses_coalesce(self):
        """conversation_history uses COALESCE for backward compat with old turns."""
        source = self._get_view_source()
        start = source.find('"conversation_history"')
        section = source[start:start + 1000]
        self.assertIn("coalesce", section.lower(),
            "conversation_history should use COALESCE for backward compatibility")

    def test_conversation_history_ordered_asc(self):
        """conversation_history orders turns by create_time ASC."""
        source = self._get_view_source()
        start = source.find('"conversation_history"')
        section = source[start:start + 1000]
        self.assertIn("ORDER BY", section)
        self.assertIn("ASC", section)

    def test_frontend_has_fetch_conversation_functions(self):
        """api.js exports fetchConversationList and fetchConversationHistory."""
        source = self._read_frontend_file("api.js")
        self.assertIn("fetchConversationList", source)
        self.assertIn("fetchConversationHistory", source)

    def test_reflect_page_imports_dropdown(self):
        """ReflectPage imports and uses ConversationDropdown."""
        source = self._read_frontend_file("pages/ReflectPage.jsx")
        self.assertIn("ConversationDropdown", source)
        self.assertIn("fetchConversationList", source)

    def test_reflect_page_updates_history_after_send(self):
        """ReflectPage should update and then re-sync the conversation list after send."""
        source = self._read_frontend_file("pages/ReflectPage.jsx")
        self.assertIn("upsertConversationSummary", source)
        self.assertIn("await refreshConversationList()", source)

    def test_reflect_page_warns_when_history_save_fails(self):
        """ReflectPage should surface backend save failures instead of silently hiding them."""
        source = self._read_frontend_file("pages/ReflectPage.jsx")
        self.assertIn("result.graph_updated === false", source)
        self.assertIn("did not save to history", source)

    def test_reflect_page_uses_pending_asterisks_and_sanitizes_assistant_messages(self):
        source = self._read_frontend_file("pages/ReflectPage.jsx")
        self.assertIn("PendingAsterisks", source)
        self.assertIn("stripSignalAddresses", source)
        self.assertIn("content: <PendingAsterisks />", source)

    def test_reflect_page_does_not_restore_last_saved_conversation(self):
        """ReflectPage should open blank on refresh instead of auto-loading a historical chat."""
        source = self._read_frontend_file("pages/ReflectPage.jsx")
        self.assertNotIn("LAST_CONVERSATION_KEY", source)
        self.assertNotIn("readPersistedConversation", source)

    def test_conversation_dropdown_component_exists(self):
        """ConversationDropdown.jsx exists with expected structure."""
        source = self._read_frontend_file("components/ConversationDropdown.jsx")
        self.assertIn("ConversationDropdown", source)
        self.assertIn("conversation-dropdown", source)
        self.assertIn("onSelect", source)
        self.assertIn("onNew", source)

    def test_css_has_dropdown_styles(self):
        """styles.css includes ConversationDropdown CSS classes."""
        source = self._read_frontend_file("styles.css")
        for cls in [".conversation-dropdown", ".conversation-dropdown-toggle",
                     ".conversation-dropdown-menu", ".conversation-dropdown-item"]:
            self.assertIn(cls, source, f"styles.css missing dropdown class: {cls}")

    def test_api_has_profile_delete_functions(self):
        source = self._read_frontend_file("api.js")
        self.assertIn("deleteStoredData", source)
        self.assertIn('request("/auth/data/"', source)
        self.assertIn("deleteAccount", source)
        self.assertIn('request("/auth/account/"', source)
        self.assertIn("exportConversations", source)
        self.assertIn('downloadRequest("/auth/export/conversations/"', source)
        self.assertIn("exportMemoryGraph", source)
        self.assertIn('downloadRequest("/auth/export/memory-graph/"', source)

    def test_profile_page_has_delete_confirmation_dialog(self):
        source = self._read_frontend_file("pages/ProfilePage.jsx")
        self.assertIn("Delete your stored data?", source)
        self.assertIn("Delete your account and all data?", source)
        self.assertIn("role=\"dialog\"", source)
        self.assertIn("Delete my account", source)
        self.assertIn("Delete my data", source)

    def test_profile_page_marks_ui_only_controls_with_icon_only_badge(self):
        source = self._read_frontend_file("pages/ProfilePage.jsx")
        self.assertIn("handleExport", source)
        self.assertIn("Export conversations", source)
        self.assertIn("Export memory graph", source)
        self.assertNotIn("PlaceholderBadge", source)
        self.assertNotIn("Save changes <PlaceholderBadge />", source)
        self.assertNotIn("AI Preferences", source)
        self.assertNotIn("Basic profile", source)
        self.assertNotIn("handleSave", source)


# ══════════════════════════════════════════════════════════════════════════════
# Section 10 — Emotional Graph Visualization
# ══════════════════════════════════════════════════════════════════════════════

class TestEmotionalGraphVisualization(unittest.TestCase):
    """Verify emotional graph endpoint, frontend builder, and sidebar redesign."""

    def _read_frontend_file(self, relative_path):
        base = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))),
            "apps", "reflect", "src",
        )
        full_path = os.path.join(base, relative_path)
        with open(full_path, "r") as f:
            return f.read()

    def _get_view_source(self):
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "views.py")
        with open(path, "r") as f:
            return f.read()

    def test_emotional_graph_registered(self):
        """GraphQueryView has an 'emotional_graph' query with a handler."""
        source = self._get_view_source()
        self.assertIn('"emotional_graph"', source)
        self.assertIn('"_handle_emotional_graph"', source)
        self.assertIn('"emotional_node_detail"', source)
        self.assertIn('"_handle_emotional_node_detail"', source)

    def test_emotional_graph_handler_exists(self):
        """views.py defines _handle_emotional_graph method."""
        source = self._get_view_source()
        self.assertIn("def _handle_emotional_graph", source)
        self.assertIn("def _handle_emotional_node_detail", source)

    def test_emotional_categories_defined(self):
        """EMOTIONAL_CATEGORIES defines all 5 expected categories."""
        source = self._get_view_source()
        self.assertIn("EMOTIONAL_CATEGORIES", source)
        for cat in ["Emotion", "Person", "ContextNode", "ActionNode", "TemporalNode"]:
            self.assertIn(f'"{cat}"', source,
                f"EMOTIONAL_CATEGORIES missing category: {cat}")

    def test_emotional_categories_have_required_keys(self):
        """EMOTIONAL_CATEGORIES entries have match, label_field, props."""
        source = self._get_view_source()
        start = source.find("EMOTIONAL_CATEGORIES")
        section = source[start:start + 2000]
        for key in ["match", "label_field", "props"]:
            self.assertIn(f'"{key}"', section,
                f"EMOTIONAL_CATEGORIES missing key: {key}")

    def test_emotional_graph_handler_returns_expected_shape(self):
        """_handle_emotional_graph references nodes, edges, counts in response."""
        source = self._get_view_source()
        handler_start = source.find("def _handle_emotional_graph")
        self.assertNotEqual(handler_start, -1, "Handler method not found")
        handler_section = source[handler_start:handler_start + 16000]
        for key in ["nodes", "edges", "counts"]:
            self.assertIn(f'"{key}"', handler_section,
                f"_handle_emotional_graph response should include '{key}'")

    def test_emotional_node_detail_handler_returns_related_sections(self):
        source = self._get_view_source()
        handler_start = source.find("def _handle_emotional_node_detail")
        self.assertNotEqual(handler_start, -1, "Detail handler not found")
        handler_section = source[handler_start:handler_start + 10000]
        for key in ["related_conversations", "related_signals", "related_nodes", "connected_signals"]:
            self.assertIn(key, handler_section)

    def test_api_has_fetch_emotional_graph(self):
        """api.js exports fetchEmotionalGraph."""
        source = self._read_frontend_file("api.js")
        self.assertIn("fetchEmotionalGraph", source)
        self.assertIn("emotional_graph", source)
        self.assertIn("fetchEmotionalNodeDetail", source)
        self.assertIn("emotional_node_detail", source)

    def test_graph_view_model_has_emotional_builder(self):
        """graphViewModel.js exports buildEmotionalGraphData and emotional config."""
        source = self._read_frontend_file("features/graphViewModel.js")
        self.assertIn("buildEmotionalGraphData", source)
        self.assertIn("EMOTIONAL_NODE_COLORS", source)
        self.assertIn("EMOTIONAL_NODE_SIZES", source)
        self.assertIn("export function normalizeEmotionalNodeLabel", source)
        self.assertIn("^[^:\\s]+:(.+)$", source)

    def test_emotional_node_colors_cover_all_categories(self):
        """EMOTIONAL_NODE_COLORS defines colors for all 5 emotional categories."""
        source = self._read_frontend_file("features/graphViewModel.js")
        for cat in ["Emotion", "Person", "ContextNode", "ActionNode", "TemporalNode"]:
            self.assertIn(cat, source,
                f"graphViewModel.js missing emotional node type: {cat}")

    def test_insights_page_uses_emotional_graph(self):
        """InsightsPage calls fetchEmotionalGraph not fetchWorkspaceGraph."""
        source = self._read_frontend_file("pages/InsightsPage.jsx")
        self.assertIn("fetchEmotionalGraph", source)
        self.assertIn("fetchEmotionalNodeDetail", source)
        self.assertIn("buildEmotionalGraphData", source)
        self.assertIn("selectedCategories", source)
        self.assertIn("DEFAULT_SELECTED_CATEGORIES", source)
        self.assertIn("loadNodeDetail", source)
        self.assertIn("selectedNodeDetail", source)
        for category in ["Emotion", "Person", "ContextNode", "ActionNode", "TemporalNode"]:
            self.assertIn(category, source)

    def test_sidebar_no_time_range_filter(self):
        """InsightsSidebar no longer has time range filter."""
        source = self._read_frontend_file("components/InsightsSidebar.jsx")
        self.assertNotIn("insights-range-selector", source)
        self.assertNotIn("TEMPORAL_OPTIONS", source)

    def test_sidebar_no_confidence_slider(self):
        """InsightsSidebar no longer has confidence slider."""
        source = self._read_frontend_file("components/InsightsSidebar.jsx")
        self.assertNotIn("insights-confidence-slider", source)
        self.assertNotIn("minConfidence", source)

    def test_sidebar_has_category_groups(self):
        """InsightsSidebar defines category groups with educational content."""
        source = self._read_frontend_file("components/InsightsSidebar.jsx")
        self.assertIn("CATEGORY_GROUPS", source)
        self.assertIn("What you feel", source)
        self.assertIn("What shapes your feelings", source)

    def test_sidebar_has_educational_descriptions(self):
        """InsightsSidebar categories include descriptions and expanded text."""
        source = self._read_frontend_file("components/InsightsSidebar.jsx")
        self.assertIn("description", source)
        self.assertIn("expandedText", source)
        self.assertIn("emotional-category-description", source)
        self.assertIn("emotional-category-explainer", source)

    def test_sidebar_has_item_chips(self):
        """InsightsSidebar renders item chips from nodesByCategory."""
        source = self._read_frontend_file("components/InsightsSidebar.jsx")
        self.assertIn("emotional-item-chip", source)
        self.assertIn("nodesByCategory", source)

    def test_css_has_emotional_category_styles(self):
        """styles.css includes new emotional category CSS classes."""
        source = self._read_frontend_file("styles.css")
        expected = [
            ".emotional-category-group",
            ".emotional-category-row",
            ".emotional-category-expanded",
            ".emotional-category-description",
            ".emotional-item-chips",
            ".emotional-item-chip",
            ".emotional-category-empty",
        ]
        for cls in expected:
            self.assertIn(cls, source, f"styles.css missing emotional class: {cls}")

    def test_css_removed_old_filter_styles(self):
        """styles.css no longer has old filter CSS classes."""
        source = self._read_frontend_file("styles.css")
        self.assertNotIn(".insights-range-selector", source)
        self.assertNotIn(".insights-confidence-slider", source)
        self.assertNotIn(".insights-type-toggle", source)


if __name__ == "__main__":
    unittest.main()
