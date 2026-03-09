"""
ThriveSight Signal Pipeline Tests — V3 SA Architecture.

Tests the core signal address system components that don't require a Neo4j
connection (coordinate parsing, wildcard detection, entity extraction, persona
configuration, prompt formatting).

For tests requiring Neo4j, use test_signal_pipeline_e2e.py with a live database.
"""

import unittest

from events_api.coordinate_system import (
    COORDINATE_NAMES,
    WILDCARD,
    build_signal_address,
    coordinate_overlap,
    detect_wildcards,
    is_fully_resolved,
    parse_signal_address,
    resolution_completeness,
)
from events_api.llm_prompts import ALL_PROMPTS, V2_PROMPTS, V3_PROMPTS, get_prompt
from events_api.persona_config import (
    DEFAULT_PERSONA,
    PERSONAS,
    get_persona,
    list_personas,
)


class TestSignalAddressParsing(unittest.TestCase):
    """Test SA string parsing and construction."""

    def test_parse_full_address(self):
        result = parse_signal_address("SA(work, manager, dismissal, monday)")
        self.assertEqual(result["context"], "work")
        self.assertEqual(result["person"], "manager")
        self.assertEqual(result["action"], "dismissal")
        self.assertEqual(result["temporal"], "monday")
        self.assertIsNone(result["emotion"])
        self.assertEqual(result["wildcards"], [])

    def test_parse_address_with_emotion(self):
        result = parse_signal_address("SA(work, manager, dismissal, monday)|anger")
        self.assertEqual(result["emotion"], "anger")
        self.assertEqual(result["context"], "work")

    def test_parse_address_with_wildcards(self):
        result = parse_signal_address("SA(work, *, dismissal, *)")
        self.assertEqual(result["person"], "*")
        self.assertEqual(result["temporal"], "*")
        self.assertEqual(result["wildcards"], ["person", "temporal"])

    def test_parse_all_wildcards(self):
        result = parse_signal_address("SA(*, *, *, *)")
        self.assertEqual(len(result["wildcards"]), 4)
        self.assertEqual(result["wildcards"], COORDINATE_NAMES)

    def test_parse_strips_whitespace(self):
        result = parse_signal_address("  SA( work , manager , dismissal , monday )  ")
        self.assertEqual(result["context"], "work")
        self.assertEqual(result["person"], "manager")

    def test_parse_invalid_format_raises(self):
        with self.assertRaises(ValueError):
            parse_signal_address("invalid format")

    def test_parse_empty_raises(self):
        with self.assertRaises(ValueError):
            parse_signal_address("")

    def test_parse_missing_coordinates_raises(self):
        with self.assertRaises(ValueError):
            parse_signal_address("SA(work, manager)")

    def test_build_signal_address_default(self):
        sa = build_signal_address()
        self.assertEqual(sa, "SA(*, *, *, *)")

    def test_build_signal_address_with_values(self):
        sa = build_signal_address(
            context="work",
            person="manager",
            action="dismissal",
            temporal="monday",
        )
        self.assertEqual(sa, "SA(work, manager, dismissal, monday)")

    def test_build_signal_address_with_emotion(self):
        sa = build_signal_address(
            context="work",
            person="manager",
            action="dismissal",
            temporal="monday",
            emotion="anger",
        )
        self.assertEqual(sa, "SA(work, manager, dismissal, monday)|anger")

    def test_roundtrip_parse_build(self):
        original = "SA(work, manager, dismissal, monday)|anger"
        parsed = parse_signal_address(original)
        rebuilt = build_signal_address(
            context=parsed["context"],
            person=parsed["person"],
            action=parsed["action"],
            temporal=parsed["temporal"],
            emotion=parsed["emotion"],
        )
        self.assertEqual(rebuilt, original)


class TestWildcardDetection(unittest.TestCase):
    """Test wildcard detection and resolution completeness."""

    def test_detect_no_wildcards(self):
        parsed = parse_signal_address("SA(work, manager, dismissal, monday)")
        self.assertEqual(detect_wildcards(parsed), [])

    def test_detect_single_wildcard(self):
        parsed = parse_signal_address("SA(work, *, dismissal, monday)")
        self.assertEqual(detect_wildcards(parsed), ["person"])

    def test_detect_multiple_wildcards(self):
        parsed = parse_signal_address("SA(*, *, dismissal, *)")
        wildcards = detect_wildcards(parsed)
        self.assertIn("context", wildcards)
        self.assertIn("person", wildcards)
        self.assertIn("temporal", wildcards)
        self.assertNotIn("action", wildcards)

    def test_is_fully_resolved_true(self):
        parsed = parse_signal_address("SA(work, manager, dismissal, monday)")
        self.assertTrue(is_fully_resolved(parsed))

    def test_is_fully_resolved_false(self):
        parsed = parse_signal_address("SA(work, *, dismissal, monday)")
        self.assertFalse(is_fully_resolved(parsed))

    def test_resolution_completeness_full(self):
        parsed = parse_signal_address("SA(work, manager, dismissal, monday)")
        self.assertAlmostEqual(resolution_completeness(parsed), 1.0)

    def test_resolution_completeness_half(self):
        parsed = parse_signal_address("SA(work, *, dismissal, *)")
        self.assertAlmostEqual(resolution_completeness(parsed), 0.5)

    def test_resolution_completeness_none(self):
        parsed = parse_signal_address("SA(*, *, *, *)")
        self.assertAlmostEqual(resolution_completeness(parsed), 0.0)

    def test_resolution_completeness_quarter(self):
        parsed = parse_signal_address("SA(work, *, *, *)")
        self.assertAlmostEqual(resolution_completeness(parsed), 0.25)


class TestCoordinateOverlap(unittest.TestCase):
    """Test signal address comparison."""

    def test_identical_addresses(self):
        sa1 = parse_signal_address("SA(work, manager, dismissal, monday)")
        sa2 = parse_signal_address("SA(work, manager, dismissal, monday)")
        result = coordinate_overlap(sa1, sa2)
        self.assertEqual(len(result["shared"]), 4)
        self.assertEqual(len(result["divergent"]), 0)
        self.assertAlmostEqual(result["overlap_score"], 1.0)

    def test_completely_different(self):
        sa1 = parse_signal_address("SA(work, manager, dismissal, monday)")
        sa2 = parse_signal_address("SA(home, partner, validation, weekend)")
        result = coordinate_overlap(sa1, sa2)
        self.assertEqual(len(result["shared"]), 0)
        self.assertEqual(len(result["divergent"]), 4)
        self.assertAlmostEqual(result["overlap_score"], 0.0)

    def test_partial_overlap(self):
        sa1 = parse_signal_address("SA(work, manager, dismissal, monday)")
        sa2 = parse_signal_address("SA(work, manager, accusation, friday)")
        result = coordinate_overlap(sa1, sa2)
        self.assertEqual(len(result["shared"]), 2)  # context + person
        self.assertEqual(len(result["divergent"]), 2)  # action + temporal
        self.assertAlmostEqual(result["overlap_score"], 0.5)

    def test_wildcards_excluded_from_overlap(self):
        sa1 = parse_signal_address("SA(work, *, dismissal, monday)")
        sa2 = parse_signal_address("SA(work, manager, dismissal, monday)")
        result = coordinate_overlap(sa1, sa2)
        self.assertIn("person", result["wildcards"])
        self.assertNotIn("person", result["shared"])

    def test_case_insensitive_comparison(self):
        sa1 = parse_signal_address("SA(Work, Manager, Dismissal, Monday)")
        sa2 = parse_signal_address("SA(work, manager, dismissal, monday)")
        result = coordinate_overlap(sa1, sa2)
        self.assertAlmostEqual(result["overlap_score"], 1.0)


class TestPersonaConfig(unittest.TestCase):
    """Test persona configuration system."""

    def test_list_personas(self):
        personas = list_personas()
        self.assertGreaterEqual(len(personas), 3)
        names = [p["id"] for p in personas]
        self.assertIn("direct_challenger", names)
        self.assertIn("gentle_explorer", names)
        self.assertIn("neutral_observer", names)

    def test_get_default_persona(self):
        persona = get_persona(DEFAULT_PERSONA)
        self.assertIsNotNone(persona.name)
        self.assertIsNotNone(persona.system_prompt_modifier)

    def test_get_direct_challenger(self):
        persona = get_persona("direct_challenger")
        self.assertEqual(persona.name, "Direct Challenger")
        self.assertEqual(persona.observation_bias_aggressiveness, "high")
        self.assertGreater(persona.context_depth, 15)

    def test_get_gentle_explorer(self):
        persona = get_persona("gentle_explorer")
        self.assertEqual(persona.observation_bias_aggressiveness, "low")
        self.assertGreater(persona.confidence_threshold, 0.6)
        self.assertFalse(persona.include_pending_insights)

    def test_persona_overrides(self):
        persona = get_persona("neutral_observer", overrides={
            "confidence_threshold": 0.9,
            "context_depth": 5,
        })
        self.assertAlmostEqual(persona.confidence_threshold, 0.9)
        self.assertEqual(persona.context_depth, 5)
        # Non-overridden values remain default
        self.assertEqual(persona.observation_bias_aggressiveness, "medium")

    def test_unknown_persona_raises(self):
        with self.assertRaises(KeyError):
            get_persona("nonexistent_persona")

    def test_unknown_override_ignored(self):
        persona = get_persona("neutral_observer", overrides={
            "nonexistent_key": "value",
        })
        self.assertFalse(hasattr(persona, "nonexistent_key"))


class TestLLMPrompts(unittest.TestCase):
    """Test LLM prompt registry and formatting."""

    def test_v2_prompts_present(self):
        self.assertIn("parser", V2_PROMPTS)
        self.assertIn("signal", V2_PROMPTS)
        self.assertIn("pattern", V2_PROMPTS)
        self.assertIn("reflection", V2_PROMPTS)
        self.assertIn("reframe", V2_PROMPTS)

    def test_v3_prompts_present(self):
        self.assertIn("signal_generation", V3_PROMPTS)
        self.assertIn("wildcard_exploration", V3_PROMPTS)
        self.assertIn("emotion_attribution", V3_PROMPTS)
        self.assertIn("confidence_assessment", V3_PROMPTS)
        self.assertIn("entity_extraction", V3_PROMPTS)
        self.assertIn("cluster_reasoning", V3_PROMPTS)

    def test_all_prompts_combined(self):
        self.assertEqual(len(ALL_PROMPTS), len(V2_PROMPTS) + len(V3_PROMPTS))

    def test_get_prompt_returns_string(self):
        prompt = get_prompt("signal_generation")
        self.assertIsInstance(prompt, str)
        self.assertGreater(len(prompt), 100)

    def test_get_prompt_with_formatting(self):
        prompt = get_prompt(
            "wildcard_exploration",
            signal_address="SA(work, *, dismissal, *)",
            wildcards="person, temporal",
            context="User is frustrated with work",
        )
        self.assertIn("SA(work, *, dismissal, *)", prompt)
        self.assertIn("person, temporal", prompt)

    def test_get_prompt_unknown_raises(self):
        with self.assertRaises(KeyError):
            get_prompt("nonexistent_prompt")

    def test_signal_generation_prompt_contains_key_elements(self):
        prompt = get_prompt("signal_generation")
        self.assertIn("Signal Address", prompt)
        self.assertIn("emotions", prompt)
        self.assertIn("observation_bias_flags", prompt)
        self.assertIn("participants", prompt)
        self.assertIn("wildcards", prompt)
        self.assertIn("confidence", prompt)


class TestEntityExtraction(unittest.TestCase):
    """Test the context assembly layer's entity extraction (no graph required)."""

    def test_extract_persons_from_roles(self):
        from events_api.context_assembly import ContextAssembler

        assembler = ContextAssembler()
        entities = assembler.extract_entities("My manager dismissed my work today")
        person_names = [p["normalized"] for p in entities["persons"]]
        self.assertIn("manager", person_names)

    def test_extract_contexts(self):
        from events_api.context_assembly import ContextAssembler

        assembler = ContextAssembler()
        entities = assembler.extract_entities("At work today, the meeting was tense")
        context_names = [c["context"] for c in entities["contexts"]]
        self.assertIn("work", context_names)

    def test_extract_temporal(self):
        from events_api.context_assembly import ContextAssembler

        assembler = ContextAssembler()
        entities = assembler.extract_entities("Yesterday my boss called me into a meeting")
        temporal_mentions = [t["mention"] for t in entities["temporal"]]
        self.assertIn("yesterday", temporal_mentions)

    def test_extract_actions(self):
        from events_api.context_assembly import ContextAssembler

        assembler = ContextAssembler()
        entities = assembler.extract_entities("She dismissed what I said and blamed me")
        action_categories = [a["category"] for a in entities["actions"]]
        self.assertIn("dismissal", action_categories)
        self.assertIn("accusation", action_categories)

    def test_extract_multiple_entities(self):
        from events_api.context_assembly import ContextAssembler

        assembler = ContextAssembler()
        entities = assembler.extract_entities(
            "Yesterday at work, my manager and colleague had a meeting where "
            "they dismissed my proposal and blamed me for the delay"
        )
        self.assertGreater(len(entities["persons"]), 0)
        self.assertGreater(len(entities["contexts"]), 0)
        self.assertGreater(len(entities["temporal"]), 0)
        self.assertGreater(len(entities["actions"]), 0)

    def test_extract_empty_message(self):
        from events_api.context_assembly import ContextAssembler

        assembler = ContextAssembler()
        entities = assembler.extract_entities("")
        self.assertEqual(entities["persons"], [])
        self.assertEqual(entities["contexts"], [])
        self.assertEqual(entities["temporal"], [])
        self.assertEqual(entities["actions"], [])


class TestSignalGeneratorFallback(unittest.TestCase):
    """Test signal generator keyword fallback (no LLM or graph required)."""

    def test_fallback_detects_emotions(self):
        from events_api.signal_engine import SignalGenerator

        generator = SignalGenerator(use_llm=False)
        result = generator._generate_signals_fallback(
            "I'm so frustrated and angry about this",
            participants=["manager"],
        )

        signals = result.get("signals", [])
        self.assertEqual(len(signals), 1)

        emotions = signals[0].get("emotions", [])
        emotion_names = [e["emotion"] for e in emotions]
        self.assertIn("frustration", emotion_names)
        self.assertIn("anger", emotion_names)

    def test_fallback_detects_context(self):
        from events_api.signal_engine import SignalGenerator

        generator = SignalGenerator(use_llm=False)
        result = generator._generate_signals_fallback(
            "The meeting at work was terrible",
            participants=[],
        )

        signals = result.get("signals", [])
        self.assertEqual(len(signals), 1)
        self.assertIn("work", signals[0]["signal_address"])

    def test_fallback_includes_wildcards(self):
        from events_api.signal_engine import SignalGenerator

        generator = SignalGenerator(use_llm=False)
        result = generator._generate_signals_fallback(
            "I feel really sad",
            participants=[],
        )

        signals = result.get("signals", [])
        self.assertEqual(len(signals), 1)
        self.assertGreater(len(signals[0].get("wildcards", [])), 0)

    def test_fallback_no_emotion_returns_neutral(self):
        from events_api.signal_engine import SignalGenerator

        generator = SignalGenerator(use_llm=False)
        result = generator._generate_signals_fallback(
            "The sky is blue",
            participants=[],
        )

        signals = result.get("signals", [])
        emotions = signals[0].get("emotions", [])
        self.assertEqual(emotions[0]["emotion"], "neutral")


if __name__ == "__main__":
    unittest.main()
