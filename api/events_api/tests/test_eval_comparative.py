"""
Tests for the comparative evaluation framework.
Validates scenario generation, condition runners, and evaluation logic
WITHOUT making actual LLM API calls.

Zone 1: Internal evaluation infrastructure.
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "reflect_api.settings")
import django
django.setup()

# Import after Django setup
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


class TestScenarioGeneration(unittest.TestCase):
    """Verify the scenario matrix generates correctly."""

    def test_full_matrix_size(self):
        from eval_comparative import generate_scenarios, MESSAGE_TYPES, CONVERSATION_DEPTHS, TOPIC_DOMAINS, PROFILES
        scenarios = generate_scenarios()
        expected = len(MESSAGE_TYPES) * len(CONVERSATION_DEPTHS) * len(TOPIC_DOMAINS) * len(PROFILES)
        self.assertEqual(len(scenarios), expected)
        # 5 × 3 × 4 × 3 = 180
        self.assertEqual(expected, 180)

    def test_max_scenarios_limits_output(self):
        from eval_comparative import generate_scenarios
        scenarios = generate_scenarios(max_scenarios=10)
        self.assertEqual(len(scenarios), 10)

    def test_scenario_ids_unique(self):
        from eval_comparative import generate_scenarios
        scenarios = generate_scenarios()
        ids = [s.scenario_id for s in scenarios]
        self.assertEqual(len(ids), len(set(ids)))

    def test_scenario_has_required_fields(self):
        from eval_comparative import generate_scenarios
        scenarios = generate_scenarios(max_scenarios=1)
        s = scenarios[0]
        self.assertIsNotNone(s.scenario_id)
        self.assertIsNotNone(s.message_type)
        self.assertIsNotNone(s.message)
        self.assertIsNotNone(s.depth)
        self.assertIsNotNone(s.domain)
        self.assertIsNotNone(s.profile_id)
        self.assertIsInstance(s.history, list)

    def test_all_message_types_represented(self):
        from eval_comparative import generate_scenarios, MESSAGE_TYPES
        scenarios = generate_scenarios()
        types_seen = set(s.message_type for s in scenarios)
        self.assertEqual(types_seen, set(MESSAGE_TYPES.keys()))

    def test_all_depths_represented(self):
        from eval_comparative import generate_scenarios, CONVERSATION_DEPTHS
        scenarios = generate_scenarios()
        depths_seen = set(s.depth for s in scenarios)
        self.assertEqual(depths_seen, set(CONVERSATION_DEPTHS.keys()))

    def test_all_domains_represented(self):
        from eval_comparative import generate_scenarios, TOPIC_DOMAINS
        scenarios = generate_scenarios()
        domains_seen = set(s.domain for s in scenarios)
        self.assertEqual(domains_seen, set(TOPIC_DOMAINS.keys()))

    def test_all_profiles_represented(self):
        from eval_comparative import generate_scenarios, PROFILES
        scenarios = generate_scenarios()
        profiles_seen = set(s.profile_id for s in scenarios)
        self.assertEqual(profiles_seen, set(PROFILES.keys()))


class TestSyntheticProfiles(unittest.TestCase):
    """Verify synthetic profiles are well-formed."""

    def test_three_profiles_exist(self):
        from eval_comparative import PROFILES
        self.assertEqual(len(PROFILES), 3)
        self.assertIn("A", PROFILES)
        self.assertIn("B", PROFILES)
        self.assertIn("C", PROFILES)

    def test_profiles_have_required_keys(self):
        from eval_comparative import PROFILES
        for pid, profile in PROFILES.items():
            self.assertIn("label", profile, f"Profile {pid} missing label")
            self.assertIn("journey_state", profile, f"Profile {pid} missing journey_state")
            self.assertIn("value_profile", profile, f"Profile {pid} missing value_profile")
            self.assertIn("soul_profile", profile, f"Profile {pid} missing soul_profile")

    def test_value_profiles_have_12_values(self):
        from eval_comparative import PROFILES
        for pid, profile in PROFILES.items():
            hierarchy = profile["value_profile"]["hierarchy"]
            self.assertEqual(len(hierarchy), 12, f"Profile {pid} has {len(hierarchy)} values, expected 12")

    def test_soul_profiles_have_regime(self):
        from eval_comparative import PROFILES
        valid_regimes = {"aristocratic", "timocratic", "oligarchic", "democratic"}
        for pid, profile in PROFILES.items():
            regime = profile["soul_profile"]["regime"]
            self.assertIn(regime, valid_regimes, f"Profile {pid} has invalid regime: {regime}")

    def test_profiles_have_distinct_top_values(self):
        from eval_comparative import PROFILES
        top_values = {}
        for pid, profile in PROFILES.items():
            top = profile["value_profile"]["hierarchy"][0]["valueId"]
            top_values[pid] = top
        # All three profiles should have different top values
        self.assertEqual(len(set(top_values.values())), 3)

    def test_profiles_have_distinct_regimes(self):
        from eval_comparative import PROFILES
        regimes = set(p["soul_profile"]["regime"] for p in PROFILES.values())
        self.assertEqual(len(regimes), 3)  # aristocratic, timocratic, democratic


class TestConditionRunners(unittest.TestCase):
    """Verify condition runners are properly defined."""

    def test_all_five_runners_exist(self):
        from eval_comparative import CONDITION_RUNNERS
        self.assertEqual(len(CONDITION_RUNNERS), 5)
        for cond in ["C0", "C1", "C2", "C3", "C4"]:
            self.assertIn(cond, CONDITION_RUNNERS)

    def test_runners_are_callable(self):
        from eval_comparative import CONDITION_RUNNERS
        for cond, runner in CONDITION_RUNNERS.items():
            self.assertTrue(callable(runner), f"{cond} runner is not callable")

    @patch("eval_comparative.generate_conversation_reply")
    def test_c0_vanilla_no_context(self, mock_reply):
        mock_reply.return_value = "I hear you. That sounds difficult."
        from eval_comparative import run_c0_vanilla, generate_scenarios
        scenario = generate_scenarios(max_scenarios=1)[0]
        result = run_c0_vanilla(scenario)
        self.assertEqual(result.condition, "C0")
        self.assertEqual(result.context_tokens, 0)
        self.assertIsNone(result.error)
        self.assertGreater(len(result.response), 0)

    @patch("eval_comparative.generate_conversation_reply")
    def test_c1_history_includes_prior_messages(self, mock_reply):
        mock_reply.return_value = "I notice this has been building for a while."
        from eval_comparative import run_c1_history, generate_scenarios
        # Use mid_session depth which has history
        scenarios = generate_scenarios(max_scenarios=180)
        mid_session = [s for s in scenarios if s.depth == "mid_session"][0]
        result = run_c1_history(mid_session)
        self.assertEqual(result.condition, "C1")
        self.assertGreater(result.context_tokens, 0)  # Has history tokens
        self.assertIsNone(result.error)

    @patch("eval_comparative.generate_conversation_reply")
    def test_c4_full_includes_journey_context(self, mock_reply):
        mock_reply.return_value = "Given your commitment to justice..."
        from eval_comparative import run_c4_polity_full, generate_scenarios
        scenario = generate_scenarios(max_scenarios=1)[0]
        result = run_c4_polity_full(scenario)
        self.assertEqual(result.condition, "C4")
        self.assertGreater(result.context_tokens, 0)  # Journey context adds tokens
        self.assertIsNone(result.error)


class TestSignalContextBuilder(unittest.TestCase):
    """Verify the SA signal context packet builder."""

    def test_empty_signals_returns_empty(self):
        from eval_comparative import _build_signal_context, Scenario
        scenario = Scenario(
            scenario_id="test", message_type="test", message="test",
            depth="first_contact", domain="interpersonal", profile_id="A",
        )
        result = _build_signal_context(scenario)
        self.assertEqual(result, "")

    def test_signals_produce_context(self):
        from eval_comparative import _build_signal_context, Scenario
        scenario = Scenario(
            scenario_id="test", message_type="test", message="test",
            depth="test", domain="test", profile_id="A",
            signals=[{
                "address": "SA(work, manager, dismissal, yesterday)",
                "emotions": [{"emotion": "anger", "intensity": 8}],
                "bias_flags": ["rumination_amplification"],
            }],
        )
        result = _build_signal_context(scenario)
        self.assertIn("SA(work, manager, dismissal, yesterday)", result)
        self.assertIn("anger", result)
        self.assertIn("rumination_amplification", result)

    def test_clusters_included(self):
        from eval_comparative import _build_signal_context, Scenario
        scenario = Scenario(
            scenario_id="test", message_type="test", message="test",
            depth="test", domain="test", profile_id="A",
            signals=[{"address": "SA(work, *, *, *)", "emotions": [], "bias_flags": []}],
            clusters=[{
                "cluster_type": "same_person_diff_time",
                "shared": {"person": ["manager"]},
                "divergent": {"temporal": ["last_week", "yesterday"]},
                "strength": 0.82,
            }],
        )
        result = _build_signal_context(scenario)
        self.assertIn("same_person_diff_time", result)
        self.assertIn("0.82", result)


class TestCogneeSimulation(unittest.TestCase):
    """Verify Cognee fallback simulation produces reasonable output."""

    def test_no_history_returns_empty(self):
        from eval_comparative import _get_cognee_context, Scenario
        scenario = Scenario(
            scenario_id="test", message_type="test", message="test",
            depth="first_contact", domain="test", profile_id="A",
        )
        result = _get_cognee_context(scenario)
        self.assertEqual(result, "")

    def test_history_extracts_entities(self):
        from eval_comparative import _get_cognee_context, Scenario
        scenario = Scenario(
            scenario_id="test", message_type="test", message="test",
            depth="test", domain="test", profile_id="A",
            history=[
                {"role": "user", "content": "My manager dismissed my idea at work."},
                {"role": "assistant", "content": "That sounds difficult."},
            ],
        )
        result = _get_cognee_context(scenario)
        self.assertIn("manager", result)
        self.assertIn("work", result)


class TestJourneyContextIntegration(unittest.TestCase):
    """Verify Journey context integrates correctly with profiles."""

    def test_format_journey_context_works_with_profiles(self):
        from eval_comparative import PROFILES
        from events_api.journey_context import format_journey_context

        for pid, profile in PROFILES.items():
            result = format_journey_context(
                journey_state=profile["journey_state"],
                value_profile=profile["value_profile"],
                soul_profile=profile["soul_profile"],
            )
            self.assertIsNotNone(result, f"Profile {pid} produced None context")
            self.assertGreater(len(result), 50, f"Profile {pid} context too short")

    def test_profile_a_mentions_justice(self):
        from eval_comparative import PROFILES
        from events_api.journey_context import format_journey_context
        p = PROFILES["A"]
        result = format_journey_context(p["journey_state"], p["value_profile"], p["soul_profile"])
        self.assertIn("justice", result.lower())

    def test_profile_b_mentions_solidarity(self):
        from eval_comparative import PROFILES
        from events_api.journey_context import format_journey_context
        p = PROFILES["B"]
        result = format_journey_context(p["journey_state"], p["value_profile"], p["soul_profile"])
        self.assertIn("solidarity", result.lower())

    def test_profile_c_mentions_democratic(self):
        from eval_comparative import PROFILES
        from events_api.journey_context import format_journey_context
        p = PROFILES["C"]
        result = format_journey_context(p["journey_state"], p["value_profile"], p["soul_profile"])
        self.assertIn("democratic", result.lower())


class TestEvaluatorPrompt(unittest.TestCase):
    """Verify the evaluator prompt is well-formed."""

    def test_prompt_has_all_placeholders(self):
        from eval_comparative import EVALUATOR_PROMPT
        self.assertIn("{user_message}", EVALUATOR_PROMPT)
        self.assertIn("{ai_response}", EVALUATOR_PROMPT)
        self.assertIn("{user_profile}", EVALUATOR_PROMPT)

    def test_prompt_mentions_all_dimensions(self):
        from eval_comparative import EVALUATOR_PROMPT
        for dim in ["Personalization", "Follow-Up", "Pattern", "Safety", "Identity Coherence"]:
            self.assertIn(dim, EVALUATOR_PROMPT)

    def test_format_profile_for_evaluator(self):
        from eval_comparative import _format_profile_for_evaluator
        result = _format_profile_for_evaluator("A")
        self.assertIn("philosopher", result)
        self.assertIn("aristocratic", result)
        self.assertIn("justice", result)


class TestSummaryStats(unittest.TestCase):
    """Verify statistical computation."""

    def test_compute_summary_with_empty_results(self):
        from eval_comparative import compute_summary_stats
        summary = compute_summary_stats([], ["C0", "C4"])
        self.assertIn("by_condition", summary)

    def test_token_count_approximation(self):
        from eval_comparative import _count_tokens_approx
        # ~4 chars per token
        self.assertEqual(_count_tokens_approx("hello world!"), 3)  # 12 chars / 4
        self.assertEqual(_count_tokens_approx(""), 0)


class TestReportGenerator(unittest.TestCase):
    """Verify report generation."""

    def test_generate_report_returns_markdown(self):
        from eval_comparative import generate_report
        report = generate_report([], {"by_condition": {}}, ["C0", "C4"])
        self.assertIn("# Comparative Evaluation Report", report)
        self.assertIn("Condition", report)

    def test_report_includes_falsification_check(self):
        from eval_comparative import generate_report
        summary = {
            "by_condition": {
                "C2": {"quality_composite": 3.0},
                "C3": {"quality_composite": 3.5},
                "C4": {"quality_composite": 4.0},
            }
        }
        report = generate_report([], summary, ["C2", "C3", "C4"])
        self.assertIn("Falsification", report)


if __name__ == "__main__":
    unittest.main()
