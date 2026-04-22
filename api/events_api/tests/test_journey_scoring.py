"""
Tests for backend journey scoring algorithms.

Validates that compute_value_profile and compute_soul_profile produce
correct results, matching the TypeScript scoring.ts implementations.
"""

import os
import sys
import unittest
from unittest.mock import patch

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "reflect_api.settings")
sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)

import django
django.setup()

from events_api.journey_scoring import (
    _classify_regime,
    _derive_ordering,
    _identify_virtues,
    compute_soul_profile,
    compute_value_profile,
)


class TestDeriveOrdering(unittest.TestCase):
    """Test the _derive_ordering helper."""

    def test_clear_winner(self):
        result = _derive_ordering({"reason": 2, "spirit": 1, "appetite": 0})
        self.assertEqual(result["type"], "one-rules")
        self.assertEqual(result["ruler"], "reason")
        self.assertEqual(result["second"], "spirit")
        self.assertEqual(result["third"], "appetite")

    def test_three_way_tie(self):
        result = _derive_ordering({"reason": 1, "spirit": 1, "appetite": 1})
        self.assertEqual(result["type"], "equal")

    def test_two_way_tie_top(self):
        result = _derive_ordering({"reason": 2, "spirit": 2, "appetite": 0})
        self.assertEqual(result["type"], "co-rulers")
        self.assertIn("reason", result["rulers"])
        self.assertIn("spirit", result["rulers"])
        self.assertEqual(result["subordinate"], "appetite")

    def test_spirit_dominates(self):
        result = _derive_ordering({"reason": 0, "spirit": 2, "appetite": 1})
        self.assertEqual(result["type"], "one-rules")
        self.assertEqual(result["ruler"], "spirit")

    def test_appetite_dominates(self):
        result = _derive_ordering({"reason": 0, "spirit": 0, "appetite": 2})
        self.assertEqual(result["type"], "one-rules")
        self.assertEqual(result["ruler"], "appetite")


class TestClassifyRegime(unittest.TestCase):
    """Test regime classification from ordering."""

    def test_reason_rules_aristocratic(self):
        self.assertEqual(
            _classify_regime({"type": "one-rules", "ruler": "reason", "second": "spirit", "third": "appetite"}),
            "aristocratic",
        )

    def test_spirit_rules_timocratic(self):
        self.assertEqual(
            _classify_regime({"type": "one-rules", "ruler": "spirit", "second": "reason", "third": "appetite"}),
            "timocratic",
        )

    def test_appetite_rules_oligarchic(self):
        self.assertEqual(
            _classify_regime({"type": "one-rules", "ruler": "appetite", "second": "spirit", "third": "reason"}),
            "oligarchic",
        )

    def test_equal_democratic(self):
        self.assertEqual(_classify_regime({"type": "equal"}), "democratic")

    def test_co_rulers_with_reason_aristocratic(self):
        self.assertEqual(
            _classify_regime({"type": "co-rulers", "rulers": ["reason", "spirit"], "subordinate": "appetite"}),
            "aristocratic",
        )

    def test_co_rulers_spirit_appetite_timocratic(self):
        self.assertEqual(
            _classify_regime({"type": "co-rulers", "rulers": ["spirit", "appetite"], "subordinate": "reason"}),
            "timocratic",
        )


class TestIdentifyVirtues(unittest.TestCase):
    """Test virtue identification from ordering."""

    def test_just_city_has_all_four(self):
        ordering = {"type": "one-rules", "ruler": "reason", "second": "spirit", "third": "appetite"}
        virtues = _identify_virtues(ordering)
        self.assertIn("wisdom", virtues)
        self.assertIn("courage", virtues)
        self.assertIn("moderation", virtues)
        self.assertIn("justice", virtues)

    def test_spirit_rules_no_wisdom(self):
        ordering = {"type": "one-rules", "ruler": "spirit", "second": "reason", "third": "appetite"}
        virtues = _identify_virtues(ordering)
        self.assertNotIn("wisdom", virtues)
        self.assertIn("moderation", virtues)  # appetite still governed
        self.assertNotIn("justice", virtues)   # reason doesn't rule

    def test_appetite_rules_no_moderation(self):
        ordering = {"type": "one-rules", "ruler": "appetite", "second": "spirit", "third": "reason"}
        virtues = _identify_virtues(ordering)
        self.assertNotIn("wisdom", virtues)
        self.assertNotIn("moderation", virtues)
        self.assertNotIn("justice", virtues)

    def test_equal_ordering_no_virtues(self):
        virtues = _identify_virtues({"type": "equal"})
        self.assertEqual(virtues, [])

    def test_co_rulers_reason_spirit_has_wisdom_courage_moderation(self):
        ordering = {"type": "co-rulers", "rulers": ["reason", "spirit"], "subordinate": "appetite"}
        virtues = _identify_virtues(ordering)
        self.assertIn("wisdom", virtues)
        self.assertIn("courage", virtues)
        self.assertIn("moderation", virtues)
        self.assertNotIn("justice", virtues)  # no clear hierarchy for justice


class TestComputeValueProfile(unittest.TestCase):
    """Test the graph-based value profile computation."""

    @patch("neomodel.db")
    def test_no_responses_returns_none(self, mock_db):
        mock_db.cypher_query.return_value = ([], [])
        result = compute_value_profile(workspace_id="ws1", owner_user_id=1)
        self.assertIsNone(result)

    @patch("neomodel.db")
    def test_single_response_scores_correctly(self, mock_db):
        mock_db.cypher_query.return_value = (
            [
                (["justice"], ["dignity"], 5000),
            ],
            ["protected", "sacrificed", "delib_ms"],
        )
        result = compute_value_profile(workspace_id="ws1", owner_user_id=1)
        self.assertIsNotNone(result)
        self.assertEqual(result["scenarioCount"], 1)

        # justice should rank higher than dignity
        hierarchy = result["hierarchy"]
        justice_rank = next(h["rank"] for h in hierarchy if h["valueId"] == "justice")
        dignity_rank = next(h["rank"] for h in hierarchy if h["valueId"] == "dignity")
        self.assertLess(justice_rank, dignity_rank)

        # Scores should be correct
        self.assertEqual(result["scores"]["justice"]["timesProtected"], 1)
        self.assertEqual(result["scores"]["justice"]["timesSacrificed"], 0)
        self.assertEqual(result["scores"]["dignity"]["timesProtected"], 0)
        self.assertEqual(result["scores"]["dignity"]["timesSacrificed"], 1)

    @patch("neomodel.db")
    def test_multiple_responses_aggregate(self, mock_db):
        mock_db.cypher_query.return_value = (
            [
                (["justice"], ["dignity"], 3000),
                (["dignity"], ["justice"], 7000),
                (["liberty"], ["order"], 2000),
            ],
            ["protected", "sacrificed", "delib_ms"],
        )
        result = compute_value_profile(workspace_id="ws1", owner_user_id=1)
        self.assertEqual(result["scenarioCount"], 3)

        # justice: 1 protected, 1 sacrificed → net 0
        self.assertEqual(result["scores"]["justice"]["timesProtected"], 1)
        self.assertEqual(result["scores"]["justice"]["timesSacrificed"], 1)

        # liberty: 1 protected, 0 sacrificed → net +1
        self.assertEqual(result["scores"]["liberty"]["timesProtected"], 1)

    @patch("neomodel.db")
    def test_deliberation_time_averaged(self, mock_db):
        mock_db.cypher_query.return_value = (
            [
                (["justice"], ["dignity"], 4000),
                (["justice"], ["order"], 6000),
            ],
            ["protected", "sacrificed", "delib_ms"],
        )
        result = compute_value_profile(workspace_id="ws1", owner_user_id=1)
        # justice was involved in both: avg = (4000+6000)/2 = 5000
        self.assertEqual(result["scores"]["justice"]["avgDeliberationMs"], 5000)


class TestComputeSoulProfile(unittest.TestCase):
    """Test the graph-based soul profile computation."""

    @patch("neomodel.db")
    def test_no_responses_returns_none(self, mock_db):
        mock_db.cypher_query.return_value = ([("equal", [])], ["stated", "served_parts"])
        result = compute_soul_profile(workspace_id="ws1", owner_user_id=1)
        self.assertIsNone(result)  # empty served_parts

    @patch("neomodel.db")
    def test_reason_dominant_profile(self, mock_db):
        stated = {"type": "one-rules", "ruler": "reason", "second": "spirit", "third": "appetite"}
        mock_db.cypher_query.return_value = (
            [(stated, ["reason", "reason", "reason", "spirit", "appetite", "reason"])],
            ["stated", "served_parts"],
        )
        result = compute_soul_profile(workspace_id="ws1", owner_user_id=1)
        self.assertIsNotNone(result)
        self.assertEqual(result["regime"], "aristocratic")
        self.assertEqual(result["revealedOrdering"]["type"], "one-rules")
        self.assertEqual(result["revealedOrdering"]["ruler"], "reason")
        self.assertIn("wisdom", result["virtuesPresent"])

    @patch("neomodel.db")
    def test_spirit_dominant_profile(self, mock_db):
        stated = {"type": "equal"}
        mock_db.cypher_query.return_value = (
            [(stated, ["spirit", "spirit", "spirit", "reason", "appetite", "spirit"])],
            ["stated", "served_parts"],
        )
        result = compute_soul_profile(workspace_id="ws1", owner_user_id=1)
        self.assertEqual(result["regime"], "timocratic")
        self.assertEqual(result["revealedOrdering"]["ruler"], "spirit")

    @patch("neomodel.db")
    def test_stated_ordering_preserved(self, mock_db):
        stated = {"type": "co-rulers", "rulers": ["reason", "appetite"], "subordinate": "spirit"}
        mock_db.cypher_query.return_value = (
            [(stated, ["reason", "spirit"])],
            ["stated", "served_parts"],
        )
        result = compute_soul_profile(workspace_id="ws1", owner_user_id=1)
        self.assertEqual(result["statedOrdering"]["type"], "co-rulers")

    @patch("neomodel.db")
    def test_frequencies_counted(self, mock_db):
        stated = {"type": "equal"}
        mock_db.cypher_query.return_value = (
            [(stated, ["reason", "spirit", "appetite"])],
            ["stated", "served_parts"],
        )
        result = compute_soul_profile(workspace_id="ws1", owner_user_id=1)
        freq = result["frequencies"]
        self.assertEqual(freq["reasonVsSpirit"]["reasonWins"], 1)
        self.assertEqual(freq["reasonVsSpirit"]["spiritWins"], 1)
        self.assertEqual(freq["spiritVsAppetite"]["spiritWins"], 1)
        self.assertEqual(freq["spiritVsAppetite"]["appetiteWins"], 1)


class TestProfileViewsExist(unittest.TestCase):
    """Verify the profile views and URL routes exist."""

    def test_journey_profile_view_importable(self):
        from events_api.journey_views import JourneyProfileView
        self.assertTrue(hasattr(JourneyProfileView, "get"))

    def test_value_profile_view_importable(self):
        from events_api.journey_views import ValueProfileView
        self.assertTrue(hasattr(ValueProfileView, "get"))

    def test_soul_profile_view_importable(self):
        from events_api.journey_views import SoulProfileView
        self.assertTrue(hasattr(SoulProfileView, "get"))

    def test_profile_urls_registered(self):
        from events_api.urls import urlpatterns
        names = [p.name for p in urlpatterns]
        self.assertIn("journey-profile", names)
        self.assertIn("journey-profile-values", names)
        self.assertIn("journey-profile-soul", names)


if __name__ == "__main__":
    unittest.main()
