"""
Seed the civic ontology into Neo4j.

Creates ValueNode, SoulPartNode, ProvocationOntologyNode, and
ProvocationChoiceNode nodes with all their relationships (TENSIONS,
SOUL_AFFINITY, HAS_CHOICE, SERVES, PROTECTS, SACRIFICES).

Idempotent: uses MERGE on unique IDs so re-running is safe.

Usage:
    python manage.py seed_ontology
    python manage.py seed_ontology --clear   # wipe and re-seed
"""

import logging

from django.core.management.base import BaseCommand

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = "Seed the civic ontology (values, soul parts, provocations) into Neo4j."

    def add_arguments(self, parser):
        parser.add_argument(
            "--clear",
            action="store_true",
            help="Delete existing ontology nodes before seeding.",
        )

    def handle(self, *args, **options):
        from neomodel import db

        from events_api.ontology_data import PROVOCATIONS, SOUL_PARTS, VALUES

        if options["clear"]:
            self._clear_ontology(db)

        counts = {"values": 0, "soul_parts": 0, "provocations": 0, "choices": 0}

        # --- Soul Parts ---
        self.stdout.write("Seeding soul parts...")
        for sp in SOUL_PARTS.values():
            db.cypher_query(
                """
                MERGE (s:SoulPartNode {part_id: $part_id})
                ON CREATE SET
                    s.name = $name,
                    s.description = $description,
                    s.tier_affiliation = $tier_affiliation
                ON MATCH SET
                    s.name = $name,
                    s.description = $description,
                    s.tier_affiliation = $tier_affiliation
                """,
                {
                    "part_id": sp["id"],
                    "name": sp["name"],
                    "description": sp["description"],
                    "tier_affiliation": sp["tier_affiliation"],
                },
            )
            counts["soul_parts"] += 1

        # --- Values ---
        self.stdout.write("Seeding values...")
        for val in VALUES.values():
            db.cypher_query(
                """
                MERGE (v:ValueNode {value_id: $value_id})
                ON CREATE SET
                    v.name = $name,
                    v.definition = $definition,
                    v.tradeoff = $tradeoff,
                    v.tier = $tier,
                    v.corrupt_name = $corrupt_name,
                    v.corrupt_form = $corrupt_form
                ON MATCH SET
                    v.name = $name,
                    v.definition = $definition,
                    v.tradeoff = $tradeoff,
                    v.tier = $tier,
                    v.corrupt_name = $corrupt_name,
                    v.corrupt_form = $corrupt_form
                """,
                {
                    "value_id": val["id"],
                    "name": val["name"],
                    "definition": val["definition"],
                    "tradeoff": val["tradeoff"],
                    "tier": val["tier"],
                    "corrupt_name": val["corrupt_name"],
                    "corrupt_form": val["corrupt_form"],
                },
            )
            counts["values"] += 1

        # --- Value → SoulPart SOUL_AFFINITY relationships ---
        self.stdout.write("Creating SOUL_AFFINITY relationships...")
        for val in VALUES.values():
            db.cypher_query(
                """
                MATCH (v:ValueNode {value_id: $value_id})
                MATCH (s:SoulPartNode {part_id: $part_id})
                MERGE (v)-[:SOUL_AFFINITY]->(s)
                """,
                {"value_id": val["id"], "part_id": val["soul_part_affinity"]},
            )

        # --- Value → Value TENSIONS relationships ---
        self.stdout.write("Creating TENSIONS relationships...")
        for val in VALUES.values():
            for tension_id in val["tensions"]:
                db.cypher_query(
                    """
                    MATCH (a:ValueNode {value_id: $from_id})
                    MATCH (b:ValueNode {value_id: $to_id})
                    MERGE (a)-[:TENSIONS]->(b)
                    """,
                    {"from_id": val["id"], "to_id": tension_id},
                )

        # --- Provocations ---
        self.stdout.write("Seeding provocations...")
        for prov in PROVOCATIONS:
            db.cypher_query(
                """
                MERGE (p:ProvocationOntologyNode {provocation_id: $provocation_id})
                ON CREATE SET
                    p.form = $form,
                    p.title = $title,
                    p.text = $text,
                    p.tension = $tension,
                    p.sequence_order = $sequence_order
                ON MATCH SET
                    p.form = $form,
                    p.title = $title,
                    p.text = $text,
                    p.tension = $tension,
                    p.sequence_order = $sequence_order
                """,
                {
                    "provocation_id": prov["id"],
                    "form": prov["form"],
                    "title": prov["title"],
                    "text": prov["text"],
                    "tension": prov["tension"],
                    "sequence_order": prov["sequence_order"],
                },
            )
            counts["provocations"] += 1

            # --- Choices within each provocation ---
            for choice in prov["choices"]:
                db.cypher_query(
                    """
                    MERGE (c:ProvocationChoiceNode {choice_id: $choice_id})
                    ON CREATE SET
                        c.text = $text,
                        c.parent_provocation_id = $parent_id
                    ON MATCH SET
                        c.text = $text,
                        c.parent_provocation_id = $parent_id
                    """,
                    {
                        "choice_id": choice["id"],
                        "text": choice["text"],
                        "parent_id": prov["id"],
                    },
                )
                counts["choices"] += 1

                # Provocation → Choice
                db.cypher_query(
                    """
                    MATCH (p:ProvocationOntologyNode {provocation_id: $prov_id})
                    MATCH (c:ProvocationChoiceNode {choice_id: $choice_id})
                    MERGE (p)-[:HAS_CHOICE]->(c)
                    """,
                    {"prov_id": prov["id"], "choice_id": choice["id"]},
                )

                # Choice → SoulPart SERVES
                db.cypher_query(
                    """
                    MATCH (c:ProvocationChoiceNode {choice_id: $choice_id})
                    MATCH (s:SoulPartNode {part_id: $part_id})
                    MERGE (c)-[:SERVES]->(s)
                    """,
                    {"choice_id": choice["id"], "part_id": choice["serves_soul_part"]},
                )

                # Choice → Value PROTECTS
                for v_id in choice["protects_values"]:
                    db.cypher_query(
                        """
                        MATCH (c:ProvocationChoiceNode {choice_id: $choice_id})
                        MATCH (v:ValueNode {value_id: $value_id})
                        MERGE (c)-[:PROTECTS]->(v)
                        """,
                        {"choice_id": choice["id"], "value_id": v_id},
                    )

                # Choice → Value SACRIFICES
                for v_id in choice["sacrifices_values"]:
                    db.cypher_query(
                        """
                        MATCH (c:ProvocationChoiceNode {choice_id: $choice_id})
                        MATCH (v:ValueNode {value_id: $value_id})
                        MERGE (c)-[:SACRIFICES]->(v)
                        """,
                        {"choice_id": choice["id"], "value_id": v_id},
                    )

        summary = (
            f"Ontology seeded: {counts['soul_parts']} soul parts, "
            f"{counts['values']} values, {counts['provocations']} provocations, "
            f"{counts['choices']} choices."
        )
        self.stdout.write(self.style.SUCCESS(summary))


    def _clear_ontology(self, db):
        """Remove all ontology nodes and their relationships."""
        self.stdout.write(self.style.WARNING("Clearing existing ontology nodes..."))
        for label in [
            "ProvocationChoiceNode",
            "ProvocationOntologyNode",
            "ValueNode",
            "SoulPartNode",
        ]:
            db.cypher_query(f"MATCH (n:{label}) DETACH DELETE n")
        self.stdout.write("Cleared.")
