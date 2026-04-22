"""
Tests for the civic ontology data and seed command.

Validates that ontology_data.py has the correct structure and that the
seed_ontology management command would create the right graph topology.
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


class TestOntologyData(unittest.TestCase):
    """Validate the ontology_data.py reference data."""

    def test_has_12_values(self):
        from events_api.ontology_data import VALUES
        self.assertEqual(len(VALUES), 12)

    def test_has_3_soul_parts(self):
        from events_api.ontology_data import SOUL_PARTS
        self.assertEqual(len(SOUL_PARTS), 3)
        self.assertIn("reason", SOUL_PARTS)
        self.assertIn("spirit", SOUL_PARTS)
        self.assertIn("appetite", SOUL_PARTS)

    def test_has_12_provocations(self):
        from events_api.ontology_data import PROVOCATIONS
        self.assertEqual(len(PROVOCATIONS), 12)

    def test_each_provocation_has_2_choices(self):
        from events_api.ontology_data import PROVOCATIONS
        for prov in PROVOCATIONS:
            self.assertEqual(
                len(prov["choices"]), 2,
                f"Provocation {prov['id']} should have exactly 2 choices",
            )

    def test_total_24_choices(self):
        from events_api.ontology_data import PROVOCATIONS
        total = sum(len(p["choices"]) for p in PROVOCATIONS)
        self.assertEqual(total, 24)

    def test_value_has_required_fields(self):
        from events_api.ontology_data import VALUES
        required = {"id", "name", "definition", "tradeoff", "tensions",
                     "soul_part_affinity", "tier", "corrupt_name", "corrupt_form"}
        for v_id, val in VALUES.items():
            for field in required:
                self.assertIn(field, val, f"Value {v_id} missing field {field}")

    def test_value_tensions_reference_valid_values(self):
        from events_api.ontology_data import VALUES
        valid_ids = set(VALUES.keys())
        for v_id, val in VALUES.items():
            for tension_id in val["tensions"]:
                self.assertIn(
                    tension_id, valid_ids,
                    f"Value {v_id} has tension to unknown value {tension_id}",
                )

    def test_value_soul_affinity_valid(self):
        from events_api.ontology_data import VALUES, SOUL_PARTS
        valid_parts = set(SOUL_PARTS.keys())
        for v_id, val in VALUES.items():
            self.assertIn(
                val["soul_part_affinity"], valid_parts,
                f"Value {v_id} has invalid soul_part_affinity",
            )

    def test_value_tiers_are_valid(self):
        from events_api.ontology_data import VALUES
        valid_tiers = {"foundational", "structural", "aspirational"}
        for v_id, val in VALUES.items():
            self.assertIn(val["tier"], valid_tiers, f"Value {v_id} has invalid tier")

    def test_foundational_values_have_reason_affinity(self):
        from events_api.ontology_data import VALUES
        for v_id, val in VALUES.items():
            if val["tier"] == "foundational":
                self.assertEqual(
                    val["soul_part_affinity"], "reason",
                    f"Foundational value {v_id} should have reason affinity",
                )

    def test_structural_values_have_spirit_affinity(self):
        from events_api.ontology_data import VALUES
        for v_id, val in VALUES.items():
            if val["tier"] == "structural":
                self.assertEqual(
                    val["soul_part_affinity"], "spirit",
                    f"Structural value {v_id} should have spirit affinity",
                )

    def test_aspirational_values_have_appetite_affinity(self):
        from events_api.ontology_data import VALUES
        for v_id, val in VALUES.items():
            if val["tier"] == "aspirational":
                self.assertEqual(
                    val["soul_part_affinity"], "appetite",
                    f"Aspirational value {v_id} should have appetite affinity",
                )

    def test_choice_serves_valid_soul_part(self):
        from events_api.ontology_data import PROVOCATIONS, SOUL_PARTS
        valid_parts = set(SOUL_PARTS.keys())
        for prov in PROVOCATIONS:
            for choice in prov["choices"]:
                self.assertIn(
                    choice["serves_soul_part"], valid_parts,
                    f"Choice {choice['id']} serves invalid soul part",
                )

    def test_choice_protects_valid_values(self):
        from events_api.ontology_data import PROVOCATIONS, VALUES
        valid_ids = set(VALUES.keys())
        for prov in PROVOCATIONS:
            for choice in prov["choices"]:
                for v_id in choice["protects_values"]:
                    self.assertIn(v_id, valid_ids, f"Choice {choice['id']} protects unknown value {v_id}")

    def test_choice_sacrifices_valid_values(self):
        from events_api.ontology_data import PROVOCATIONS, VALUES
        valid_ids = set(VALUES.keys())
        for prov in PROVOCATIONS:
            for choice in prov["choices"]:
                for v_id in choice["sacrifices_values"]:
                    self.assertIn(v_id, valid_ids, f"Choice {choice['id']} sacrifices unknown value {v_id}")

    def test_provocation_ids_unique(self):
        from events_api.ontology_data import PROVOCATIONS
        ids = [p["id"] for p in PROVOCATIONS]
        self.assertEqual(len(ids), len(set(ids)), "Provocation IDs must be unique")

    def test_choice_ids_unique(self):
        from events_api.ontology_data import PROVOCATIONS
        ids = []
        for prov in PROVOCATIONS:
            for choice in prov["choices"]:
                ids.append(choice["id"])
        self.assertEqual(len(ids), len(set(ids)), "Choice IDs must be unique")

    def test_sequence_orders_are_1_to_12(self):
        from events_api.ontology_data import PROVOCATIONS
        orders = sorted(p["sequence_order"] for p in PROVOCATIONS)
        self.assertEqual(orders, list(range(1, 13)))

    def test_provocation_tension_valid(self):
        from events_api.ontology_data import PROVOCATIONS
        valid = {"reason-spirit", "reason-appetite", "spirit-appetite"}
        for prov in PROVOCATIONS:
            self.assertIn(
                prov["tension"], valid,
                f"Provocation {prov['id']} has invalid tension: {prov['tension']}",
            )


class TestOntologyGraphModels(unittest.TestCase):
    """Verify the neomodel classes for ontology nodes exist."""

    def test_value_node_exists(self):
        from events_api.graph_models import ValueNode
        self.assertTrue(hasattr(ValueNode, "value_id"))
        self.assertTrue(hasattr(ValueNode, "tensions"))
        self.assertTrue(hasattr(ValueNode, "soul_affinity"))

    def test_soul_part_node_exists(self):
        from events_api.graph_models import SoulPartNode
        self.assertTrue(hasattr(SoulPartNode, "part_id"))

    def test_provocation_ontology_node_exists(self):
        from events_api.graph_models import ProvocationOntologyNode
        self.assertTrue(hasattr(ProvocationOntologyNode, "provocation_id"))
        self.assertTrue(hasattr(ProvocationOntologyNode, "choices"))

    def test_provocation_choice_node_exists(self):
        from events_api.graph_models import ProvocationChoiceNode
        self.assertTrue(hasattr(ProvocationChoiceNode, "choice_id"))
        self.assertTrue(hasattr(ProvocationChoiceNode, "serves"))
        self.assertTrue(hasattr(ProvocationChoiceNode, "protects"))
        self.assertTrue(hasattr(ProvocationChoiceNode, "sacrifices"))

    def test_provocation_response_has_ontology_links(self):
        from events_api.graph_models import ProvocationResponseNode
        self.assertTrue(hasattr(ProvocationResponseNode, "to_provocation"))
        self.assertTrue(hasattr(ProvocationResponseNode, "chose"))

    def test_seed_command_importable(self):
        from events_api.management.commands.seed_ontology import Command
        self.assertTrue(hasattr(Command, "handle"))


if __name__ == "__main__":
    unittest.main()
