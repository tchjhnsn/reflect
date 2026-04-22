"""
Tests for journey context formatting and injection into AI pipeline.

Validates that format_journey_context produces correct natural-language
output and that the LiveConversationService injects it properly.
"""

import os
import sys
import unittest

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "reflect_api.settings")
sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)

import django
django.setup()

from events_api.journey_context import format_journey_context


class TestFormatJourneyContext(unittest.TestCase):
    """Test format_journey_context output."""

    def test_returns_none_for_empty_data(self):
        result = format_journey_context(None, None, None)
        self.assertIsNone(result)

    def test_returns_none_for_no_phase(self):
        result = format_journey_context(
            {"journey_phase": None},
            None,
            None,
        )
        self.assertIsNone(result)

    def test_includes_path_and_phase(self):
        journey_state = {
            "journey_phase": "provocation",
            "path_id": "philosopher",
            "philosopher_mode": "guided",
        }
        result = format_journey_context(journey_state, None, None)
        self.assertIsNotNone(result)
        self.assertIn("Philosopher", result)
        self.assertIn("guided", result)
        self.assertIn("provocation", result)

    def test_includes_sovereign_goal(self):
        journey_state = {
            "journey_phase": "provocation",
            "path_id": "sovereign",
            "sovereign_mode": "self-advised",
            "sovereign_end_statement": "Build a just city",
        }
        result = format_journey_context(journey_state, None, None)
        self.assertIn("Sovereign", result)
        self.assertIn("self-advised", result)
        self.assertIn("Build a just city", result)

    def test_includes_soul_ordering_one_rules(self):
        soul_profile = {
            "revealedOrdering": {
                "type": "one-rules",
                "ruler": "reason",
                "second": "spirit",
                "third": "appetite",
            },
            "statedOrdering": {
                "type": "one-rules",
                "ruler": "reason",
                "second": "spirit",
                "third": "appetite",
            },
            "regime": "aristocratic",
            "virtuesPresent": ["wisdom", "courage", "moderation", "justice"],
        }
        result = format_journey_context(None, None, soul_profile)
        self.assertIn("Reason rules", result)
        self.assertIn("Spirit second", result)
        self.assertIn("Aristocratic", result)
        self.assertIn("wisdom", result)

    def test_includes_soul_ordering_co_rulers(self):
        soul_profile = {
            "revealedOrdering": {
                "type": "co-rulers",
                "rulers": ["reason", "spirit"],
                "subordinate": "appetite",
            },
            "statedOrdering": {"type": "equal"},
            "regime": "aristocratic",
            "virtuesPresent": ["wisdom", "courage"],
        }
        result = format_journey_context(None, None, soul_profile)
        self.assertIn("co-rule", result)
        self.assertIn("Reason", result)
        self.assertIn("Spirit", result)

    def test_notes_stated_vs_revealed_difference(self):
        soul_profile = {
            "revealedOrdering": {"type": "one-rules", "ruler": "spirit", "second": "reason", "third": "appetite"},
            "statedOrdering": {"type": "equal"},
            "regime": "timocratic",
            "virtuesPresent": [],
        }
        result = format_journey_context(None, None, soul_profile)
        self.assertIn("differs from", result)

    def test_includes_value_hierarchy(self):
        value_profile = {
            "hierarchy": [
                {"valueId": "justice", "rank": 1, "confidence": 1.0},
                {"valueId": "dignity", "rank": 2, "confidence": 0.8},
                {"valueId": "liberty", "rank": 3, "confidence": 0.7},
                {"valueId": "order", "rank": 4, "confidence": 0.6},
                {"valueId": "authority", "rank": 5, "confidence": 0.5},
                {"valueId": "sovereignty", "rank": 6, "confidence": 0.4},
                {"valueId": "equality", "rank": 7, "confidence": 0.3},
                {"valueId": "prosperity", "rank": 8, "confidence": 0.2},
                {"valueId": "solidarity", "rank": 9, "confidence": 0.1},
                {"valueId": "pluralism", "rank": 10, "confidence": 0.1},
                {"valueId": "merit", "rank": 11, "confidence": 0.1},
                {"valueId": "stewardship", "rank": 12, "confidence": 0.1},
            ],
            "scores": {
                "justice": {"timesProtected": 3, "timesSacrificed": 0, "avgDeliberationMs": 8000},
                "dignity": {"timesProtected": 2, "timesSacrificed": 1, "avgDeliberationMs": 5000},
                "liberty": {"timesProtected": 1, "timesSacrificed": 0, "avgDeliberationMs": 3000},
                "order": {"timesProtected": 0, "timesSacrificed": 0, "avgDeliberationMs": 0},
                "authority": {"timesProtected": 0, "timesSacrificed": 0, "avgDeliberationMs": 0},
                "sovereignty": {"timesProtected": 0, "timesSacrificed": 0, "avgDeliberationMs": 0},
                "equality": {"timesProtected": 0, "timesSacrificed": 0, "avgDeliberationMs": 0},
                "prosperity": {"timesProtected": 0, "timesSacrificed": 0, "avgDeliberationMs": 0},
                "solidarity": {"timesProtected": 0, "timesSacrificed": 0, "avgDeliberationMs": 0},
                "pluralism": {"timesProtected": 0, "timesSacrificed": 0, "avgDeliberationMs": 0},
                "merit": {"timesProtected": 0, "timesSacrificed": 0, "avgDeliberationMs": 0},
                "stewardship": {"timesProtected": 0, "timesSacrificed": 0, "avgDeliberationMs": 0},
            },
            "computedAt": "2026-03-11T00:00:00Z",
            "scenarioCount": 4,
        }
        result = format_journey_context(None, value_profile, None)
        self.assertIn("justice", result)
        self.assertIn("dignity", result)
        self.assertIn("liberty", result)
        # Should include deliberation insight
        self.assertIn("Most conflicted", result)
        self.assertIn("justice", result)

    def test_full_context_all_sections(self):
        journey = {"journey_phase": "act-i-complete", "path_id": "philosopher", "philosopher_mode": "socratic"}
        soul = {
            "revealedOrdering": {"type": "one-rules", "ruler": "reason", "second": "spirit", "third": "appetite"},
            "statedOrdering": {"type": "one-rules", "ruler": "reason", "second": "spirit", "third": "appetite"},
            "regime": "aristocratic",
            "virtuesPresent": ["wisdom", "courage", "moderation", "justice"],
        }
        values = {
            "hierarchy": [{"valueId": f"v{i}", "rank": i, "confidence": 0.5} for i in range(1, 13)],
            "scores": {},
            "computedAt": "2026-03-11T00:00:00Z",
            "scenarioCount": 12,
        }
        result = format_journey_context(journey, values, soul)
        self.assertIn("civic identity", result)
        self.assertIn("Philosopher", result)
        self.assertIn("Aristocratic", result)
        self.assertIn("Top values", result)


class TestLiveConversationServiceIntegration(unittest.TestCase):
    """Verify the injection point exists in LiveConversationService."""

    def test_inject_method_exists(self):
        from events_api.live_conversation_service import LiveConversationService
        self.assertTrue(hasattr(LiveConversationService, "_inject_journey_context"))

    def test_imports_journey_modules(self):
        import events_api.live_conversation_service as lcs
        # Verify the module has the journey imports
        self.assertTrue(hasattr(lcs, "format_journey_context"))
        self.assertTrue(hasattr(lcs, "compute_value_profile"))
        self.assertTrue(hasattr(lcs, "compute_soul_profile"))
        self.assertTrue(hasattr(lcs, "get_journey_state_from_graph"))


if __name__ == "__main__":
    unittest.main()
