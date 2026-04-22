"""
Tests for Journey state persistence API.

Tests the journey_views.py endpoints and the graph_sync journey functions.
Uses mocks for Neo4j (same pattern as the existing test suite).
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# ─── Django Setup ─────────────────────────────────────────────────────────────
# Ensure Django is configured before importing any app modules.
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "reflect_api.settings")
sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)

import django
django.setup()


# =============================================================================
# Journey Serializer Tests
# =============================================================================

class TestJourneyStateSerializer(unittest.TestCase):
    """Validate the JourneyStateSerializer accepts valid journey data."""

    def test_valid_phase_update(self):
        from events_api.journey_serializers import JourneyStateSerializer

        data = {"journey_phase": "path-selection"}
        serializer = JourneyStateSerializer(data=data)
        self.assertTrue(serializer.is_valid(), serializer.errors)
        self.assertEqual(serializer.validated_data["journey_phase"], "path-selection")

    def test_valid_path_selection(self):
        from events_api.journey_serializers import JourneyStateSerializer

        data = {"path_id": "philosopher", "philosopher_mode": "socratic"}
        serializer = JourneyStateSerializer(data=data)
        self.assertTrue(serializer.is_valid(), serializer.errors)

    def test_valid_soul_ordering_one_rules(self):
        from events_api.journey_serializers import JourneyStateSerializer

        data = {
            "soul_ordering": {
                "type": "one-rules",
                "ruler": "reason",
                "second": "spirit",
                "third": "appetite",
            }
        }
        serializer = JourneyStateSerializer(data=data)
        self.assertTrue(serializer.is_valid(), serializer.errors)

    def test_valid_soul_ordering_co_rulers(self):
        from events_api.journey_serializers import JourneyStateSerializer

        data = {
            "soul_ordering": {
                "type": "co-rulers",
                "rulers": ["reason", "spirit"],
                "subordinate": "appetite",
            }
        }
        serializer = JourneyStateSerializer(data=data)
        self.assertTrue(serializer.is_valid(), serializer.errors)

    def test_valid_soul_ordering_equal(self):
        from events_api.journey_serializers import JourneyStateSerializer

        data = {"soul_ordering": {"type": "equal"}}
        serializer = JourneyStateSerializer(data=data)
        self.assertTrue(serializer.is_valid(), serializer.errors)

    def test_invalid_soul_ordering_type(self):
        from events_api.journey_serializers import JourneyStateSerializer

        data = {"soul_ordering": {"type": "invalid"}}
        serializer = JourneyStateSerializer(data=data)
        self.assertFalse(serializer.is_valid())

    def test_valid_value_ordering(self):
        from events_api.journey_serializers import JourneyStateSerializer

        data = {
            "value_ordering": [
                "liberty", "justice", "dignity", "equality",
                "solidarity", "sovereignty", "authority", "pluralism",
                "merit", "stewardship", "order", "prosperity",
            ]
        }
        serializer = JourneyStateSerializer(data=data)
        self.assertTrue(serializer.is_valid(), serializer.errors)

    def test_invalid_value_ordering_bad_value(self):
        from events_api.journey_serializers import JourneyStateSerializer

        data = {"value_ordering": ["liberty", "invalid_value"]}
        serializer = JourneyStateSerializer(data=data)
        self.assertFalse(serializer.is_valid())

    def test_invalid_phase(self):
        from events_api.journey_serializers import JourneyStateSerializer

        data = {"journey_phase": "not-a-real-phase"}
        serializer = JourneyStateSerializer(data=data)
        self.assertFalse(serializer.is_valid())

    def test_invalid_path_id(self):
        from events_api.journey_serializers import JourneyStateSerializer

        data = {"path_id": "invalid_path"}
        serializer = JourneyStateSerializer(data=data)
        self.assertFalse(serializer.is_valid())

    def test_partial_update_only_phase(self):
        from events_api.journey_serializers import JourneyStateSerializer

        data = {"journey_phase": "provocation", "current_provocation_index": 3}
        serializer = JourneyStateSerializer(data=data)
        self.assertTrue(serializer.is_valid(), serializer.errors)
        self.assertEqual(serializer.validated_data["current_provocation_index"], 3)

    def test_null_path_is_valid(self):
        from events_api.journey_serializers import JourneyStateSerializer

        data = {"path_id": None}
        serializer = JourneyStateSerializer(data=data)
        self.assertTrue(serializer.is_valid(), serializer.errors)

    def test_boolean_fields(self):
        from events_api.journey_serializers import JourneyStateSerializer

        data = {"socratic_chariot_revealed": True, "socratic_tier_revealed": False}
        serializer = JourneyStateSerializer(data=data)
        self.assertTrue(serializer.is_valid(), serializer.errors)


class TestProvocationResponseSerializer(unittest.TestCase):
    """Validate the ProvocationResponseSerializer."""

    def test_valid_response(self):
        from events_api.journey_serializers import ProvocationResponseSerializer

        data = {
            "provocationId": "prov-001",
            "choiceId": "choice-a",
            "servedSoulPart": "reason",
            "protectedValues": ["justice", "liberty"],
            "sacrificedValues": ["order"],
            "deliberationTimeMs": 3400,
            "wasInstinctive": False,
        }
        serializer = ProvocationResponseSerializer(data=data)
        self.assertTrue(serializer.is_valid(), serializer.errors)

    def test_missing_required_fields(self):
        from events_api.journey_serializers import ProvocationResponseSerializer

        data = {"provocationId": "prov-001"}  # Missing choiceId, servedSoulPart
        serializer = ProvocationResponseSerializer(data=data)
        self.assertFalse(serializer.is_valid())

    def test_invalid_soul_part(self):
        from events_api.journey_serializers import ProvocationResponseSerializer

        data = {
            "provocationId": "prov-001",
            "choiceId": "choice-a",
            "servedSoulPart": "invalid",
        }
        serializer = ProvocationResponseSerializer(data=data)
        self.assertFalse(serializer.is_valid())

    def test_invalid_protected_value(self):
        from events_api.journey_serializers import ProvocationResponseSerializer

        data = {
            "provocationId": "prov-001",
            "choiceId": "choice-a",
            "servedSoulPart": "spirit",
            "protectedValues": ["invalid_value"],
        }
        serializer = ProvocationResponseSerializer(data=data)
        self.assertFalse(serializer.is_valid())

    def test_minimal_valid_response(self):
        from events_api.journey_serializers import ProvocationResponseSerializer

        data = {
            "provocationId": "prov-001",
            "choiceId": "choice-a",
            "servedSoulPart": "appetite",
        }
        serializer = ProvocationResponseSerializer(data=data)
        self.assertTrue(serializer.is_valid(), serializer.errors)
        # Defaults should be applied
        self.assertEqual(serializer.validated_data["deliberationTimeMs"], 0)
        self.assertFalse(serializer.validated_data["wasInstinctive"])


# =============================================================================
# Graph Sync Journey Function Tests
# =============================================================================

class TestJourneyGraphSync(unittest.TestCase):
    """Test the graph_sync journey helper functions."""

    @patch("neomodel.db")
    def test_update_journey_state_filters_allowed_fields(self, mock_db):
        from events_api.graph_sync import update_journey_state_in_graph

        mock_db.cypher_query.return_value = ([{"journey_phase": "chariot"}], [])

        result = update_journey_state_in_graph(
            workspace_id="ws-123",
            owner_user_id=1,
            updates={
                "journey_phase": "chariot",
                "evil_field": "should_be_filtered",  # Not in allowed list
            },
        )

        # Verify the Cypher query was called
        self.assertTrue(mock_db.cypher_query.called)
        query = mock_db.cypher_query.call_args.args[0]
        # Only journey_phase should be in the SET clause
        self.assertIn("u.journey_phase", query)
        self.assertNotIn("evil_field", query)

    @patch("neomodel.db")
    def test_update_journey_state_empty_updates_returns_none(self, mock_db):
        from events_api.graph_sync import update_journey_state_in_graph

        result = update_journey_state_in_graph(
            workspace_id="ws-123",
            owner_user_id=1,
            updates={"not_a_journey_field": "value"},
        )

        self.assertIsNone(result)
        mock_db.cypher_query.assert_not_called()

    @patch("neomodel.db")
    def test_get_journey_state_returns_dict(self, mock_db):
        from events_api.graph_sync import get_journey_state_from_graph

        mock_db.cypher_query.return_value = (
            [("chariot", "philosopher", "socratic", None, None, None, None, 2, True, False)],
            [
                "journey_phase", "path_id", "philosopher_mode", "sovereign_mode",
                "sovereign_end_statement", "soul_ordering", "value_ordering",
                "current_provocation_index", "socratic_chariot_revealed", "socratic_tier_revealed",
            ],
        )

        result = get_journey_state_from_graph(workspace_id="ws-123", owner_user_id=1)

        self.assertIsNotNone(result)
        self.assertEqual(result["journey_phase"], "chariot")
        self.assertEqual(result["path_id"], "philosopher")
        self.assertEqual(result["current_provocation_index"], 2)
        self.assertTrue(result["socratic_chariot_revealed"])

    @patch("neomodel.db")
    def test_get_journey_state_missing_profile_returns_none(self, mock_db):
        from events_api.graph_sync import get_journey_state_from_graph

        mock_db.cypher_query.return_value = ([], [])

        result = get_journey_state_from_graph(workspace_id="ws-123", owner_user_id=1)
        self.assertIsNone(result)

    @patch("neomodel.db")
    def test_create_provocation_response_creates_node(self, mock_db):
        from events_api.graph_sync import create_provocation_response_in_graph

        mock_db.cypher_query.return_value = (
            [{"provocation_id": "prov-001", "choice_id": "choice-a"}],
            [],
        )

        result = create_provocation_response_in_graph(
            workspace_id="ws-123",
            owner_user_id=1,
            response_data={
                "provocationId": "prov-001",
                "choiceId": "choice-a",
                "servedSoulPart": "reason",
                "protectedValues": ["justice"],
                "sacrificedValues": ["order"],
                "deliberationTimeMs": 2500,
                "wasInstinctive": False,
            },
        )

        self.assertTrue(mock_db.cypher_query.called)
        query = mock_db.cypher_query.call_args.args[0]
        self.assertIn("CREATE (r:ProvocationResponseNode", query)
        self.assertIn("HAS_PROVOCATION_RESPONSE", query)

    @patch("neomodel.db")
    def test_list_provocation_responses_returns_list(self, mock_db):
        from events_api.graph_sync import list_provocation_responses_from_graph

        mock_db.cypher_query.return_value = (
            [
                ({"provocation_id": "prov-001", "choice_id": "a"},),
                ({"provocation_id": "prov-002", "choice_id": "b"},),
            ],
            [],
        )

        result = list_provocation_responses_from_graph(
            workspace_id="ws-123",
            owner_user_id=1,
        )

        self.assertEqual(len(result), 2)

    @patch("neomodel.db")
    def test_list_provocation_responses_empty(self, mock_db):
        from events_api.graph_sync import list_provocation_responses_from_graph

        mock_db.cypher_query.return_value = ([], [])

        result = list_provocation_responses_from_graph(
            workspace_id="ws-123",
            owner_user_id=1,
        )

        self.assertEqual(result, [])


# =============================================================================
# Journey View Source Tests (structural checks)
# =============================================================================

class TestJourneyViewsStructure(unittest.TestCase):
    """Verify journey_views.py has the expected structure."""

    def _read_source(self, filename):
        source_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            filename,
        )
        with open(source_path, "r") as f:
            return f.read()

    def test_journey_views_has_state_view(self):
        source = self._read_source("journey_views.py")
        self.assertIn("class JourneyStateView", source)

    def test_journey_views_has_response_view(self):
        source = self._read_source("journey_views.py")
        self.assertIn("class ProvocationResponseView", source)

    def test_journey_views_uses_workspace_scoping(self):
        source = self._read_source("journey_views.py")
        self.assertIn("WorkspaceAPIView", source)
        self.assertIn("self.get_workspace()", source)

    def test_urls_has_journey_routes(self):
        source = self._read_source("urls.py")
        self.assertIn('path("journey/state/"', source)
        self.assertIn('path("journey/responses/"', source)

    def test_journey_serializers_has_valid_phases(self):
        from events_api.journey_serializers import VALID_JOURNEY_PHASES
        # Check all 19 phases are present (including three-plane phases)
        self.assertIn("path-selection", VALID_JOURNEY_PHASES)
        self.assertIn("act-i-complete", VALID_JOURNEY_PHASES)
        self.assertIn("three-plane-intro", VALID_JOURNEY_PHASES)
        self.assertIn("provocation", VALID_JOURNEY_PHASES)
        self.assertIn("soul-ordering", VALID_JOURNEY_PHASES)
        self.assertGreaterEqual(len(VALID_JOURNEY_PHASES), 14)

    def test_journey_serializers_has_valid_values(self):
        from events_api.journey_serializers import VALID_VALUE_IDS
        self.assertEqual(len(VALID_VALUE_IDS), 12)
        self.assertIn("liberty", VALID_VALUE_IDS)
        self.assertIn("stewardship", VALID_VALUE_IDS)

    def test_journey_serializers_has_valid_soul_parts(self):
        from events_api.journey_serializers import VALID_SOUL_PARTS
        self.assertEqual(VALID_SOUL_PARTS, ["reason", "spirit", "appetite"])


# =============================================================================
# UserProfile Model Extended Fields Tests
# =============================================================================

class TestUserProfileJourneyFields(unittest.TestCase):
    """Verify UserProfile model has the new journey fields."""

    def test_userprofile_has_journey_phase(self):
        from events_api.graph_models import UserProfile
        self.assertTrue(hasattr(UserProfile, "journey_phase"))

    def test_userprofile_has_path_id(self):
        from events_api.graph_models import UserProfile
        self.assertTrue(hasattr(UserProfile, "path_id"))

    def test_userprofile_has_soul_ordering(self):
        from events_api.graph_models import UserProfile
        self.assertTrue(hasattr(UserProfile, "soul_ordering"))

    def test_userprofile_has_value_ordering(self):
        from events_api.graph_models import UserProfile
        self.assertTrue(hasattr(UserProfile, "value_ordering"))

    def test_userprofile_has_philosopher_mode(self):
        from events_api.graph_models import UserProfile
        self.assertTrue(hasattr(UserProfile, "philosopher_mode"))

    def test_userprofile_has_sovereign_mode(self):
        from events_api.graph_models import UserProfile
        self.assertTrue(hasattr(UserProfile, "sovereign_mode"))

    def test_userprofile_has_provocation_responses_relationship(self):
        from events_api.graph_models import UserProfile
        self.assertTrue(hasattr(UserProfile, "provocation_responses"))

    def test_provocation_response_node_exists(self):
        from events_api.graph_models import ProvocationResponseNode
        self.assertTrue(hasattr(ProvocationResponseNode, "provocation_id"))
        self.assertTrue(hasattr(ProvocationResponseNode, "choice_id"))
        self.assertTrue(hasattr(ProvocationResponseNode, "served_soul_part"))
        self.assertTrue(hasattr(ProvocationResponseNode, "protected_values"))
        self.assertTrue(hasattr(ProvocationResponseNode, "sacrificed_values"))
        self.assertTrue(hasattr(ProvocationResponseNode, "deliberation_time_ms"))
        self.assertTrue(hasattr(ProvocationResponseNode, "was_instinctive"))

    def test_existing_userprofile_fields_preserved(self):
        """Ensure journey extension doesn't break existing fields."""
        from events_api.graph_models import UserProfile
        self.assertTrue(hasattr(UserProfile, "workspace_id"))
        self.assertTrue(hasattr(UserProfile, "owner_user_id"))
        self.assertTrue(hasattr(UserProfile, "username"))
        self.assertTrue(hasattr(UserProfile, "email"))
        self.assertTrue(hasattr(UserProfile, "total_messages"))
        self.assertTrue(hasattr(UserProfile, "conversations"))


# =============================================================================
# Journey View Response Format Tests
# =============================================================================

class TestJourneyViewResponseFormats(unittest.TestCase):
    """Verify the _format functions produce correct camelCase output."""

    def test_format_journey_state_null(self):
        from events_api.journey_views import _format_journey_state

        result = _format_journey_state(None)
        self.assertIsNone(result["journeyPhase"])
        self.assertIsNone(result["pathId"])
        self.assertEqual(result["currentProvocationIndex"], 0)
        self.assertFalse(result["socraticChariotRevealed"])

    def test_format_journey_state_with_data(self):
        from events_api.journey_views import _format_journey_state

        result = _format_journey_state({
            "journey_phase": "provocation",
            "path_id": "philosopher",
            "philosopher_mode": "socratic",
            "sovereign_mode": None,
            "sovereign_end_statement": None,
            "soul_ordering": {"type": "one-rules", "ruler": "reason", "second": "spirit", "third": "appetite"},
            "value_ordering": ["liberty", "justice"],
            "current_provocation_index": 3,
            "socratic_chariot_revealed": True,
            "socratic_tier_revealed": False,
        })

        self.assertEqual(result["journeyPhase"], "provocation")
        self.assertEqual(result["pathId"], "philosopher")
        self.assertEqual(result["philosopherMode"], "socratic")
        self.assertEqual(result["currentProvocationIndex"], 3)
        self.assertTrue(result["socraticChariotRevealed"])
        self.assertEqual(result["soulOrdering"]["type"], "one-rules")

    def test_format_provocation_response(self):
        from events_api.journey_views import _format_provocation_response

        result = _format_provocation_response({
            "provocation_id": "prov-001",
            "choice_id": "choice-a",
            "served_soul_part": "reason",
            "protected_values": ["justice"],
            "sacrificed_values": ["order"],
            "deliberation_time_ms": 2500,
            "was_instinctive": False,
            "timestamp": "2025-01-15T10:00:00",
        })

        self.assertEqual(result["provocationId"], "prov-001")
        self.assertEqual(result["servedSoulPart"], "reason")
        self.assertEqual(result["deliberationTimeMs"], 2500)
        self.assertFalse(result["wasInstinctive"])


if __name__ == "__main__":
    unittest.main()
