#!/usr/bin/env python3
"""
Reflect Model Benchmark — reflect_benchmark.py

Automated capability scoring across 7 dimensions.
Each dimension has test cases that produce a 0–5 score.
Results are written to reflect_benchmark_results.json.

Usage:
    cd services/reflect-api
    python reflect_benchmark.py

Zone 1: Internal evaluation tool — never share externally.
"""

import importlib
import inspect
import json
import os
import sys
import time
from datetime import datetime, timezone

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "reflect_api.settings")

import django
django.setup()

# ── Test Infrastructure ─────────────────────────────────────────────────

class BenchmarkResult:
    def __init__(self, test_name, dimension, score, max_score, detail=""):
        self.test_name = test_name
        self.dimension = dimension
        self.score = score
        self.max_score = max_score
        self.detail = detail

    def to_dict(self):
        return {
            "test": self.test_name,
            "dimension": self.dimension,
            "score": self.score,
            "max_score": self.max_score,
            "passed": self.score == self.max_score,
            "detail": self.detail,
        }


results: list[BenchmarkResult] = []


def record(test_name, dimension, score, max_score=1.0, detail=""):
    results.append(BenchmarkResult(test_name, dimension, score, max_score, detail))
    status = "PASS" if score == max_score else ("PARTIAL" if score > 0 else "FAIL")
    print(f"  [{status}] {test_name}: {score}/{max_score}" + (f" — {detail}" if detail else ""))


# ── Dimension: SIG (Signal Detection) ───────────────────────────────────

def bench_signal_detection():
    print("\n═══ SIG: Signal Detection ═══")

    # Test 1: SA coordinate parsing
    try:
        from events_api.coordinate_system import parse_signal_address
        result = parse_signal_address("SA(work, manager, dismissal, yesterday)")
        assert result["context"] == "work"
        assert result["person"] == "manager"
        assert result["action"] == "dismissal"
        assert result["temporal"] == "yesterday"
        record("sa_parsing_basic", "SIG", 1.0, detail="SA(c,p,a,t) parses correctly")
    except Exception as e:
        record("sa_parsing_basic", "SIG", 0.0, detail=str(e))

    # Test 2: Wildcard detection
    try:
        from events_api.coordinate_system import parse_signal_address, get_wildcards
        parsed = parse_signal_address("SA(work, *, dismissal, *)")
        wildcards = get_wildcards(parsed)
        assert "person" in wildcards
        assert "temporal" in wildcards
        assert "context" not in wildcards
        record("wildcard_detection", "SIG", 1.0, detail="Wildcards correctly identified")
    except Exception as e:
        record("wildcard_detection", "SIG", 0.0, detail=str(e))

    # Test 3: Keyword fallback emotion detection
    try:
        from events_api.signal_engine import SignalGenerator
        gen = SignalGenerator()
        # Test the keyword fallback path
        signals = gen._generate_signals_keyword(
            "I'm furious at my boss for dismissing my idea in front of everyone",
            conversation_id="test-conv-1",
        )
        emotion_names = set()
        for s in signals:
            for e in s.get("emotions", []):
                emotion_names.add(e.get("name", "").lower())
        has_anger = any(e in emotion_names for e in ("anger", "frustration", "furious"))
        record("keyword_emotion_detection", "SIG", 1.0 if has_anger else 0.0,
               detail=f"Detected emotions: {emotion_names}")
    except Exception as e:
        record("keyword_emotion_detection", "SIG", 0.0, detail=str(e))

    # Test 4: Multi-emotion support in signal structure
    try:
        from events_api.signal_engine import SignalGenerator
        gen = SignalGenerator()
        signals = gen._generate_signals_keyword(
            "I feel both angry and deeply hurt by what happened",
            conversation_id="test-conv-2",
        )
        total_emotions = sum(len(s.get("emotions", [])) for s in signals)
        score = 1.0 if total_emotions >= 2 else 0.5 if total_emotions >= 1 else 0.0
        record("multi_emotion_detection", "SIG", score,
               detail=f"Total emotions across signals: {total_emotions}")
    except Exception as e:
        record("multi_emotion_detection", "SIG", 0.0, detail=str(e))

    # Test 5: Observation bias flag structure
    try:
        from events_api.signal_engine import SignalGenerator
        gen = SignalGenerator()
        # Check that the generator has bias detection capability
        has_assess = hasattr(gen, "assess_confidence") or hasattr(gen, "_assess_confidence")
        # Check that bias types are defined
        known_biases = {"projection", "rumination_amplification", "confirmation_bias", "narrative_construction"}
        source = inspect.getsource(type(gen))
        found_biases = sum(1 for b in known_biases if b in source)
        score = 0.0
        if found_biases >= 4:
            score = 1.0
        elif found_biases >= 2:
            score = 0.5
        record("observation_bias_structure", "SIG", score,
               detail=f"Found {found_biases}/4 bias types in source")
    except Exception as e:
        record("observation_bias_structure", "SIG", 0.0, detail=str(e))

    # Test 6: Embedding quality (placeholder vs real)
    try:
        from events_api.signal_engine import SignalGenerator
        gen = SignalGenerator()
        emb = gen.compute_embedding("I feel anxious about work")
        if emb is None:
            record("embedding_quality", "SIG", 0.0, detail="No embedding function")
        elif len(emb) == 32:
            # SHA-256 hash produces 32 floats — placeholder
            record("embedding_quality", "SIG", 0.0,
                   detail="SHA-256 placeholder (32 floats), not semantic")
        elif len(emb) >= 256:
            record("embedding_quality", "SIG", 1.0,
                   detail=f"Real embedding ({len(emb)} dims)")
        else:
            record("embedding_quality", "SIG", 0.5,
                   detail=f"Unknown embedding type ({len(emb)} dims)")
    except Exception as e:
        record("embedding_quality", "SIG", 0.0, detail=str(e))

    # Test 7: Known emotion vocabulary size
    try:
        from events_api.signal_engine import KNOWN_EMOTIONS
        count = len(KNOWN_EMOTIONS)
        score = 1.0 if count >= 25 else 0.5 if count >= 15 else 0.0
        record("emotion_vocabulary", "SIG", score,
               detail=f"{count} emotions in vocabulary")
    except ImportError:
        try:
            source = inspect.getsource(importlib.import_module("events_api.signal_engine"))
            if "KNOWN_EMOTIONS" in source or "EMOTION_KEYWORDS" in source:
                record("emotion_vocabulary", "SIG", 0.5, detail="Emotion set exists but not importable as constant")
            else:
                record("emotion_vocabulary", "SIG", 0.0, detail="No emotion vocabulary found")
        except Exception as e:
            record("emotion_vocabulary", "SIG", 0.0, detail=str(e))


# ── Dimension: CLU (Cluster Intelligence) ───────────────────────────────

def bench_cluster_intelligence():
    print("\n═══ CLU: Cluster Intelligence ═══")

    # Test 1: All 6 cluster types defined
    try:
        source = inspect.getsource(importlib.import_module("events_api.cluster_engine"))
        types = ["same_time_diff_emotion", "same_person_diff_time", "same_person_diff_context",
                 "same_context_diff_person", "same_action", "emotional_convergence",
                 "emotional_divergence", "cross_dimensional", "general"]
        found = sum(1 for t in types if t in source)
        score = 1.0 if found >= 6 else 0.5 if found >= 3 else 0.0
        record("cluster_types_defined", "CLU", score,
               detail=f"Found {found} cluster type references")
    except Exception as e:
        record("cluster_types_defined", "CLU", 0.0, detail=str(e))

    # Test 2: Strength computation exists
    try:
        source = inspect.getsource(importlib.import_module("events_api.cluster_engine"))
        has_strength = "compute_strength" in source or "strength" in source
        has_density = "density_factor" in source or "density" in source
        has_recency = "recency_factor" in source or "recency" in source
        score = 1.0 if (has_strength and has_density and has_recency) else 0.5 if has_strength else 0.0
        record("strength_computation", "CLU", score,
               detail=f"strength={has_strength}, density={has_density}, recency={has_recency}")
    except Exception as e:
        record("strength_computation", "CLU", 0.0, detail=str(e))

    # Test 3: Trajectory tracking
    try:
        source = inspect.getsource(importlib.import_module("events_api.cluster_engine"))
        has_trajectory = "trajectory" in source
        has_history = "trajectory_history" in source
        has_snapshot = "snapshot" in source or "timestamp" in source
        score = 1.0 if (has_trajectory and has_history) else 0.5 if has_trajectory else 0.0
        record("trajectory_tracking", "CLU", score,
               detail=f"trajectory={has_trajectory}, history={has_history}")
    except Exception as e:
        record("trajectory_tracking", "CLU", 0.0, detail=str(e))

    # Test 4: Dissolution logic
    try:
        source = inspect.getsource(importlib.import_module("events_api.cluster_engine"))
        has_dissolve = "dissolve" in source or "dissolution" in source
        has_threshold = "dissolution_threshold" in source or "check_dissolution" in source
        has_status = '"dissolved"' in source or "'dissolved'" in source
        score = 1.0 if (has_dissolve and has_status) else 0.5 if has_dissolve else 0.0
        record("dissolution_logic", "CLU", score,
               detail=f"dissolve={has_dissolve}, threshold={has_threshold}, status={has_status}")
    except Exception as e:
        record("dissolution_logic", "CLU", 0.0, detail=str(e))

    # Test 5: Coordinate overlap scoring
    try:
        from events_api.coordinate_system import compare_coordinates
        result = compare_coordinates(
            {"context": "work", "person": "manager", "action": "dismissal", "temporal": "yesterday"},
            {"context": "work", "person": "manager", "action": "criticism", "temporal": "last_week"},
        )
        has_overlap = "overlap_score" in result or "shared" in result
        record("coordinate_overlap", "CLU", 1.0 if has_overlap else 0.0,
               detail=f"Comparison result keys: {list(result.keys())}")
    except Exception as e:
        record("coordinate_overlap", "CLU", 0.0, detail=str(e))


# ── Dimension: CTX (Context Assembly) ───────────────────────────────────

def bench_context_assembly():
    print("\n═══ CTX: Context Assembly ═══")

    # Test 1: Entity extraction works
    try:
        from events_api.context_assembly import extract_entities
        entities = extract_entities("My manager dismissed my idea at work yesterday")
        has_persons = bool(entities.get("persons") or entities.get("people"))
        has_contexts = bool(entities.get("contexts"))
        has_temporal = bool(entities.get("temporal") or entities.get("temporal_references"))
        score = (0.33 * has_persons) + (0.33 * has_contexts) + (0.34 * has_temporal)
        record("entity_extraction", "CTX", round(score, 2),
               detail=f"persons={has_persons}, contexts={has_contexts}, temporal={has_temporal}")
    except Exception as e:
        record("entity_extraction", "CTX", 0.0, detail=str(e))

    # Test 2: Graph enrichment (the critical gap)
    try:
        from events_api.context_assembly import enrich_from_graph
        # Call with dummy params to see if it returns real data or empty stubs
        result = enrich_from_graph(
            entities={"persons": ["manager"], "contexts": ["work"]},
            owner_user_id=1,
            workspace_id=1,
            persona_config=None,
        )
        # Check if it returns actual graph data or empty placeholders
        signals = result.get("signals", result.get("related_signals", []))
        clusters = result.get("clusters", result.get("active_clusters", []))
        if signals or clusters:
            record("graph_enrichment", "CTX", 1.0,
                   detail=f"Returns real data: {len(signals)} signals, {len(clusters)} clusters")
        else:
            record("graph_enrichment", "CTX", 0.0,
                   detail="Returns empty/stubbed data — graph context never reaches LLM")
    except TypeError:
        # Function signature might not match
        record("graph_enrichment", "CTX", 0.0,
               detail="enrich_from_graph() exists but call failed — likely stubbed")
    except Exception as e:
        record("graph_enrichment", "CTX", 0.0, detail=f"Not functional: {e}")

    # Test 3: Token budget management
    try:
        source = inspect.getsource(importlib.import_module("events_api.context_assembly"))
        has_budget = "token_budget" in source or "max_context_tokens" in source or "CHARS_PER_TOKEN" in source
        has_compress = "compress" in source or "truncat" in source
        score = 1.0 if (has_budget and has_compress) else 0.5 if has_budget else 0.0
        record("token_budget", "CTX", score,
               detail=f"budget={has_budget}, compress={has_compress}")
    except Exception as e:
        record("token_budget", "CTX", 0.0, detail=str(e))

    # Test 4: Context packet assembly
    try:
        from events_api.context_assembly import assemble_context
        packet = assemble_context(
            message="I'm stressed about work again",
            conversation_history=[],
            graph_context=None,
        )
        is_string = isinstance(packet, str)
        has_content = len(packet) > 10 if is_string else False
        record("context_packet_assembly", "CTX", 1.0 if has_content else 0.5 if is_string else 0.0,
               detail=f"Returns {'string' if is_string else type(packet).__name__}, len={len(packet) if is_string else 'N/A'}")
    except Exception as e:
        # May need different args
        record("context_packet_assembly", "CTX", 0.0, detail=str(e))


# ── Dimension: CONV (Conversation Quality) ──────────────────────────────

def bench_conversation_quality():
    print("\n═══ CONV: Conversation Quality ═══")

    # Test 1: Pipeline exists end-to-end
    try:
        from events_api.live_conversation_service import LiveConversationService
        has_run = hasattr(LiveConversationService, "run")
        has_build = hasattr(LiveConversationService, "build_prompt_context") or \
                    hasattr(LiveConversationService, "_build_prompt_context")
        score = 1.0 if (has_run and has_build) else 0.5 if has_run else 0.0
        record("pipeline_exists", "CONV", score,
               detail=f"run={has_run}, build_prompt_context={has_build}")
    except Exception as e:
        record("pipeline_exists", "CONV", 0.0, detail=str(e))

    # Test 2: Graph context reaches LLM (the critical test)
    try:
        source = inspect.getsource(importlib.import_module("events_api.live_conversation_service"))
        calls_enrich = "enrich_from_graph" in source
        # Check if the enrichment result is actually used in prompt building
        uses_enrichment = "context_packet" in source or "graph_context" in source
        if calls_enrich and uses_enrichment:
            # Now check if enrich_from_graph is stubbed
            from events_api.context_assembly import enrich_from_graph
            esource = inspect.getsource(enrich_from_graph)
            is_stubbed = "return {" in esource and ("[]" in esource or "{}" in esource)
            if is_stubbed:
                record("graph_informs_llm", "CONV", 0.0,
                       detail="Pipeline calls enrich_from_graph but it returns empty stubs")
            else:
                record("graph_informs_llm", "CONV", 1.0,
                       detail="Graph context flows into LLM prompt")
        else:
            record("graph_informs_llm", "CONV", 0.0,
                   detail=f"calls_enrich={calls_enrich}, uses_enrichment={uses_enrichment}")
    except Exception as e:
        record("graph_informs_llm", "CONV", 0.0, detail=str(e))

    # Test 3: Persona modifier injected
    try:
        source = inspect.getsource(importlib.import_module("events_api.live_conversation_service"))
        has_persona = "persona" in source.lower()
        has_modifier = "system_prompt_modifier" in source or "persona_modifier" in source or "build_system_prompt" in source
        score = 1.0 if has_modifier else 0.5 if has_persona else 0.0
        record("persona_modifier_injected", "CONV", score,
               detail=f"persona_ref={has_persona}, modifier={has_modifier}")
    except Exception as e:
        record("persona_modifier_injected", "CONV", 0.0, detail=str(e))

    # Test 4: Signal generation post-response
    try:
        source = inspect.getsource(importlib.import_module("events_api.live_conversation_service"))
        has_signal_gen = "SignalGenerator" in source or "generate_signal" in source
        has_post_response = "signal" in source.lower()
        score = 1.0 if has_signal_gen else 0.5 if has_post_response else 0.0
        record("post_response_signals", "CONV", score,
               detail=f"SignalGenerator={has_signal_gen}")
    except Exception as e:
        record("post_response_signals", "CONV", 0.0, detail=str(e))


# ── Dimension: GRAPH (Graph Richness) ───────────────────────────────────

def bench_graph_richness():
    print("\n═══ GRAPH: Graph Richness ═══")

    # Test 1: Core node types defined
    try:
        source = inspect.getsource(importlib.import_module("events_api.graph_models"))
        core_nodes = ["Signal", "Cluster", "Insight", "Conversation", "UserProfile",
                      "Person", "Context", "Action", "Temporal"]
        found = sum(1 for n in core_nodes if f"class {n}" in source or f"'{n}'" in source or f'"{n}"' in source)
        score = 1.0 if found >= 7 else 0.5 if found >= 4 else 0.0
        record("core_node_types", "GRAPH", score,
               detail=f"Found {found}/{len(core_nodes)} core node types")
    except Exception as e:
        record("core_node_types", "GRAPH", 0.0, detail=str(e))

    # Test 2: Advanced node types (Theme, Belief, Outcome, Behavior)
    try:
        source = inspect.getsource(importlib.import_module("events_api.graph_models"))
        advanced = ["Theme", "Belief", "Outcome", "Behavior"]
        found = sum(1 for n in advanced if f"class {n}" in source)
        score = 1.0 if found >= 3 else 0.5 if found >= 1 else 0.0
        record("advanced_node_types", "GRAPH", score,
               detail=f"Found {found}/{len(advanced)} advanced node types")
    except Exception as e:
        record("advanced_node_types", "GRAPH", 0.0, detail=str(e))

    # Test 3: Relationship types
    try:
        source = inspect.getsource(importlib.import_module("events_api.graph_models"))
        rels = ["MEMBER_OF", "PARTICIPANT_IN", "IN_CONTEXT", "INVOLVES_ACTION",
                "AT_TIME", "DERIVED_FROM", "INFORMED_BY", "RESPONDS_TO",
                "OWNS_CONVERSATION", "CONTAINS_SIGNAL"]
        found = sum(1 for r in rels if r in source)
        score = 1.0 if found >= 7 else 0.5 if found >= 4 else 0.0
        record("relationship_types", "GRAPH", score,
               detail=f"Found {found}/{len(rels)} relationship types")
    except Exception as e:
        record("relationship_types", "GRAPH", 0.0, detail=str(e))

    # Test 4: Graph write pipeline
    try:
        from events_api.live_graph import write_live_conversation_to_graph
        record("graph_write_pipeline", "GRAPH", 1.0,
               detail="write_live_conversation_to_graph importable")
    except ImportError:
        record("graph_write_pipeline", "GRAPH", 0.0, detail="Graph write function not found")

    # Test 5: Pipeline trace recording
    try:
        source = inspect.getsource(importlib.import_module("events_api.live_graph"))
        has_trace = "PipelineTrace" in source or "pipeline_trace" in source
        record("pipeline_trace", "GRAPH", 1.0 if has_trace else 0.0,
               detail=f"PipelineTrace recording: {has_trace}")
    except Exception as e:
        record("pipeline_trace", "GRAPH", 0.0, detail=str(e))


# ── Dimension: AUTO (Autonomy) ──────────────────────────────────────────

def bench_autonomy():
    print("\n═══ AUTO: Autonomy ═══")

    # Test 1: Graph agent detection cycle
    try:
        from events_api.graph_agent import GraphAgent
        agent = GraphAgent.__new__(GraphAgent)
        has_cycle = hasattr(agent, "run_detection_cycle")
        has_continuous = hasattr(agent, "run_continuous")
        score = 1.0 if (has_cycle and has_continuous) else 0.5 if has_cycle else 0.0
        record("detection_cycle", "AUTO", score,
               detail=f"run_detection_cycle={has_cycle}, run_continuous={has_continuous}")
    except Exception as e:
        record("detection_cycle", "AUTO", 0.0, detail=str(e))

    # Test 2: PendingInsight generation
    try:
        source = inspect.getsource(importlib.import_module("events_api.graph_agent"))
        has_pending = "PendingInsight" in source or "pending_insight" in source
        has_create = "create" in source.lower() and has_pending
        score = 1.0 if has_create else 0.5 if has_pending else 0.0
        record("pending_insight_generation", "AUTO", score,
               detail=f"PendingInsight={has_pending}, creation={has_create}")
    except Exception as e:
        record("pending_insight_generation", "AUTO", 0.0, detail=str(e))

    # Test 3: Stale insight pruning
    try:
        source = inspect.getsource(importlib.import_module("events_api.graph_agent"))
        has_prune = "prune" in source.lower() or "expir" in source.lower()
        has_days = "EXPIRY_DAYS" in source or "expiry" in source
        score = 1.0 if (has_prune and has_days) else 0.5 if has_prune else 0.0
        record("stale_pruning", "AUTO", score,
               detail=f"pruning={has_prune}, expiry_config={has_days}")
    except Exception as e:
        record("stale_pruning", "AUTO", 0.0, detail=str(e))

    # Test 4: No LLM calls in agent (architecture advantage)
    try:
        source = inspect.getsource(importlib.import_module("events_api.graph_agent"))
        has_llm = "llm_client" in source or "generate_conversation" in source or "anthropic" in source.lower()
        record("no_llm_in_agent", "AUTO", 1.0 if not has_llm else 0.0,
               detail=f"LLM-free agent: {not has_llm}")
    except Exception as e:
        record("no_llm_in_agent", "AUTO", 0.0, detail=str(e))

    # Test 5: Embedding backfill
    try:
        source = inspect.getsource(importlib.import_module("events_api.graph_agent"))
        has_embed = "embedding" in source.lower() or "backfill" in source.lower()
        record("embedding_backfill", "AUTO", 1.0 if has_embed else 0.0,
               detail=f"embedding_backfill={has_embed}")
    except Exception as e:
        record("embedding_backfill", "AUTO", 0.0, detail=str(e))


# ── Dimension: PERS (Persona Adaptation) ────────────────────────────────

def bench_persona_adaptation():
    print("\n═══ PERS: Persona Adaptation ═══")

    # Test 1: All 3 personas defined
    try:
        from events_api.persona_config import PERSONAS, get_persona
        persona_names = set()
        if isinstance(PERSONAS, dict):
            persona_names = set(PERSONAS.keys())
        elif isinstance(PERSONAS, list):
            persona_names = {p.get("id", p.get("name", "")) for p in PERSONAS}
        expected = {"direct_challenger", "gentle_explorer", "neutral_observer"}
        found = len(expected.intersection({n.lower().replace(" ", "_") for n in persona_names}))
        score = 1.0 if found >= 3 else 0.5 if found >= 1 else 0.0
        record("personas_defined", "PERS", score,
               detail=f"Found personas: {persona_names}")
    except Exception as e:
        record("personas_defined", "PERS", 0.0, detail=str(e))

    # Test 2: Threshold values present
    try:
        source = inspect.getsource(importlib.import_module("events_api.persona_config"))
        thresholds = ["context_depth", "confidence_threshold", "cluster_surfacing_threshold",
                      "observation_bias_aggressiveness", "max_context_tokens"]
        found = sum(1 for t in thresholds if t in source)
        score = 1.0 if found >= 4 else 0.5 if found >= 2 else 0.0
        record("threshold_values", "PERS", score,
               detail=f"Found {found}/{len(thresholds)} threshold parameters")
    except Exception as e:
        record("threshold_values", "PERS", 0.0, detail=str(e))

    # Test 3: Thresholds actually filter graph context
    try:
        # Check if context_assembly uses persona thresholds to filter
        ctx_source = inspect.getsource(importlib.import_module("events_api.context_assembly"))
        uses_depth = "context_depth" in ctx_source
        uses_confidence = "confidence_threshold" in ctx_source
        uses_cluster = "cluster_surfacing" in ctx_source or "cluster_threshold" in ctx_source
        # If enrich_from_graph is stubbed, thresholds have no effect
        from events_api.context_assembly import enrich_from_graph
        enrich_source = inspect.getsource(enrich_from_graph)
        is_stubbed = "return {" in enrich_source and "[]" in enrich_source

        if is_stubbed:
            record("thresholds_filter_context", "PERS", 0.0,
                   detail="Thresholds exist in code but enrich_from_graph is stubbed — no effect")
        elif uses_depth or uses_confidence:
            record("thresholds_filter_context", "PERS", 1.0,
                   detail=f"depth={uses_depth}, confidence={uses_confidence}, cluster={uses_cluster}")
        else:
            record("thresholds_filter_context", "PERS", 0.0,
                   detail="Thresholds not used in context assembly")
    except Exception as e:
        record("thresholds_filter_context", "PERS", 0.0, detail=str(e))

    # Test 4: System prompt modifier injection
    try:
        source = inspect.getsource(importlib.import_module("events_api.llm_prompts"))
        has_builder = "build_system_prompt" in source
        has_modifier = "system_prompt_modifier" in source or "persona_modifier" in source
        score = 1.0 if (has_builder and has_modifier) else 0.5 if has_builder else 0.0
        record("prompt_modifier_injection", "PERS", score,
               detail=f"build_system_prompt={has_builder}, modifier={has_modifier}")
    except Exception as e:
        record("prompt_modifier_injection", "PERS", 0.0, detail=str(e))


# ── Scoring & Output ────────────────────────────────────────────────────

def compute_scores():
    """Aggregate per-test results into dimension scores (0–5 scale)."""
    dimensions = {}
    for r in results:
        if r.dimension not in dimensions:
            dimensions[r.dimension] = {"total_score": 0, "total_max": 0, "tests": []}
        dimensions[r.dimension]["total_score"] += r.score
        dimensions[r.dimension]["total_max"] += r.max_score
        dimensions[r.dimension]["tests"].append(r.to_dict())

    scored = {}
    for dim_id, data in dimensions.items():
        raw = data["total_score"] / data["total_max"] if data["total_max"] > 0 else 0
        # Scale to 0–5, round to nearest 0.5
        scaled = round(raw * 5 * 2) / 2
        scored[dim_id] = {
            "score": scaled,
            "raw_ratio": round(raw, 3),
            "tests_passed": sum(1 for t in data["tests"] if t["passed"]),
            "tests_total": len(data["tests"]),
            "tests": data["tests"],
        }
    return scored


def main():
    print("╔══════════════════════════════════════════════════╗")
    print("║       REFLECT MODEL BENCHMARK                   ║")
    print("║       Automated Capability Scoring               ║")
    print("╚══════════════════════════════════════════════════╝")

    start = time.time()

    bench_signal_detection()
    bench_cluster_intelligence()
    bench_context_assembly()
    bench_conversation_quality()
    bench_graph_richness()
    bench_autonomy()
    bench_persona_adaptation()

    duration = time.time() - start
    scored = compute_scores()

    # Print summary
    dim_names = {
        "SIG": "Signal Detection",
        "CLU": "Cluster Intelligence",
        "CTX": "Context Assembly",
        "CONV": "Conversation Quality",
        "GRAPH": "Graph Richness",
        "AUTO": "Autonomy",
        "PERS": "Persona Adaptation",
    }

    print("\n" + "═" * 55)
    print("  SCORECARD")
    print("═" * 55)
    total = 0
    for dim_id in ["SIG", "CLU", "CTX", "CONV", "GRAPH", "AUTO", "PERS"]:
        if dim_id in scored:
            s = scored[dim_id]["score"]
            total += s
            bar = "█" * int(s * 2) + "░" * (10 - int(s * 2))
            print(f"  {dim_id:5s} {dim_names.get(dim_id, dim_id):25s} {bar} {s}/5.0")
    print("─" * 55)
    print(f"  {'TOTAL':5s} {'':25s}            {total}/35.0")
    print("═" * 55)

    # R-0 comparison
    r0_tier1 = {"SIG": 3.5, "CLU": 4.0, "CTX": 1.5, "CONV": 1.0, "GRAPH": 3.5, "AUTO": 4.0, "PERS": 2.0}
    r0_tier2 = {"SIG": 4.5, "CLU": 4.5, "CTX": 4.0, "CONV": 3.5, "GRAPH": 4.5, "AUTO": 4.5, "PERS": 4.0}

    dims_above_t2 = 0
    print("\n  vs R-0 Tier 2:")
    for dim_id in ["SIG", "CLU", "CTX", "CONV", "GRAPH", "AUTO", "PERS"]:
        if dim_id in scored:
            current = scored[dim_id]["score"]
            t2 = r0_tier2[dim_id]
            diff = current - t2
            if diff >= 1.0:
                dims_above_t2 += 1
            marker = "✓ +1.0" if diff >= 1.0 else f"  {diff:+.1f}"
            print(f"    {dim_id}: {current}/5 vs Tier2 {t2}/5  {marker}")

    print(f"\n  Dimensions exceeding R-0 Tier 2 by ≥1.0: {dims_above_t2}/7")
    if dims_above_t2 >= 5:
        print("  ★ QUALIFIES FOR R-1 DESIGNATION ★")
    else:
        print(f"  Need {5 - dims_above_t2} more dimensions to qualify for R-1")

    # Write results
    output = {
        "benchmark_version": "1.0",
        "model": "current",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": round(duration, 2),
        "total_score": total,
        "max_score": 35.0,
        "r0_tier1_total": sum(r0_tier1.values()),
        "r0_tier2_total": sum(r0_tier2.values()),
        "dims_above_tier2": dims_above_t2,
        "qualifies_r1": dims_above_t2 >= 5,
        "dimensions": scored,
    }

    out_path = os.path.join(os.path.dirname(__file__), "reflect_benchmark_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results written to: {out_path}")


if __name__ == "__main__":
    main()
