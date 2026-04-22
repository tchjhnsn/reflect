"""
Management command to backfill existing Neo4j data for the emotional graph.

Fixes three issues with historical data:
  1. Signal nodes missing workspace_id — infers from linked Conversation
  2. Emotion nodes never materialized — parses Signal.emotions JSON
  3. Coordinate nodes (ContextNode, Person, ActionNode, TemporalNode)
     missing uid property

Usage:
    python manage.py backfill_emotional_graph
    python manage.py backfill_emotional_graph --dry-run
    python manage.py backfill_emotional_graph --workspace ws_abc123

Zone 1: Proprietary — internal tool.
"""

import json

from django.core.management.base import BaseCommand

from events_api.neo4j_client import cypher_query


class Command(BaseCommand):
    help = "Backfill workspace_id on Signals, materialize Emotion nodes, and add uid to coordinate nodes."

    def add_arguments(self, parser):
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Preview changes without writing to Neo4j",
        )
        parser.add_argument(
            "--workspace",
            type=str,
            default=None,
            help="Limit backfill to a specific workspace_id",
        )

    def handle(self, *args, **options):
        dry_run = options["dry_run"]
        workspace_filter = options.get("workspace")
        prefix = "[DRY RUN] " if dry_run else ""

        self.stdout.write(self.style.MIGRATE_HEADING(
            f"{prefix}Starting emotional graph backfill..."
        ))

        # ── Step 1: Add workspace_id to Signals that are missing it ────
        self.stdout.write(f"\n{prefix}Step 1: Backfill workspace_id on Signal nodes...")
        self._backfill_signal_workspace_ids(dry_run, workspace_filter)

        # ── Step 2: Add uid to coordinate nodes missing it ─────────────
        self.stdout.write(f"\n{prefix}Step 2: Add uid to coordinate nodes...")
        self._backfill_coordinate_uids(dry_run)

        # ── Step 3: Materialize Emotion nodes from Signal.emotions ─────
        self.stdout.write(f"\n{prefix}Step 3: Materialize Emotion nodes...")
        self._materialize_emotion_nodes(dry_run, workspace_filter)

        self.stdout.write(self.style.SUCCESS(
            f"\n{prefix}Backfill complete."
        ))

    def _backfill_signal_workspace_ids(self, dry_run, workspace_filter):
        """
        Find Signal nodes without workspace_id and infer it from
        the Conversation they're linked to (via CONTAINS_SIGNAL).
        """
        # First, count how many are missing
        count_q = """
            MATCH (s:Signal)
            WHERE s.workspace_id IS NULL OR s.workspace_id = ''
            RETURN count(s) AS cnt
        """
        rows, _ = cypher_query(count_q)
        missing_count = rows[0][0] if rows else 0
        self.stdout.write(f"  Found {missing_count} Signal nodes missing workspace_id")

        if missing_count == 0 or dry_run:
            return

        # Try to infer workspace_id from linked Conversation
        infer_q = """
            MATCH (c:Conversation)-[:CONTAINS_SIGNAL]->(s:Signal)
            WHERE (s.workspace_id IS NULL OR s.workspace_id = '')
                  AND c.workspace_id IS NOT NULL
            SET s.workspace_id = c.workspace_id
            RETURN count(s) AS updated
        """
        rows, _ = cypher_query(infer_q)
        updated = rows[0][0] if rows else 0
        self.stdout.write(self.style.SUCCESS(f"  Updated {updated} Signals from linked Conversations"))

        # For any remaining orphan Signals, try inferring from UserTurn in same conversation
        orphan_q = """
            MATCH (s:Signal)
            WHERE s.workspace_id IS NULL OR s.workspace_id = ''
            RETURN count(s) AS cnt
        """
        rows, _ = cypher_query(orphan_q)
        still_missing = rows[0][0] if rows else 0

        if still_missing > 0 and workspace_filter:
            # If user specified a workspace, assign it to remaining orphans
            fix_q = """
                MATCH (s:Signal)
                WHERE s.workspace_id IS NULL OR s.workspace_id = ''
                SET s.workspace_id = $ws
                RETURN count(s) AS updated
            """
            rows, _ = cypher_query(fix_q, {"ws": workspace_filter})
            fixed = rows[0][0] if rows else 0
            self.stdout.write(f"  Assigned workspace '{workspace_filter}' to {fixed} orphan Signals")
        elif still_missing > 0:
            self.stdout.write(self.style.WARNING(
                f"  {still_missing} Signals still missing workspace_id (no linked Conversation). "
                f"Use --workspace to assign them."
            ))

    def _backfill_coordinate_uids(self, dry_run):
        """Add uid = randomUUID() to coordinate nodes that lack one."""
        node_types = ["Emotion", "ContextNode", "Person", "ActionNode", "TemporalNode"]

        for label in node_types:
            count_q = f"""
                MATCH (n:{label})
                WHERE n.uid IS NULL
                RETURN count(n) AS cnt
            """
            rows, _ = cypher_query(count_q)
            missing = rows[0][0] if rows else 0

            if missing == 0:
                self.stdout.write(f"  {label}: all nodes have uid")
                continue

            self.stdout.write(f"  {label}: {missing} nodes missing uid")

            if dry_run:
                continue

            fix_q = f"""
                MATCH (n:{label})
                WHERE n.uid IS NULL
                SET n.uid = randomUUID()
                RETURN count(n) AS updated
            """
            rows, _ = cypher_query(fix_q)
            updated = rows[0][0] if rows else 0
            self.stdout.write(self.style.SUCCESS(f"  {label}: added uid to {updated} nodes"))

    def _materialize_emotion_nodes(self, dry_run, workspace_filter):
        """
        Parse the emotions JSON array stored on each Signal node and
        create (MERGE) separate Emotion nodes with EXPRESSES_EMOTION edges.
        """
        # Find Signals that have emotions data but no EXPRESSES_EMOTION edge yet
        ws_clause = "AND s.workspace_id = $ws" if workspace_filter else ""
        find_q = f"""
            MATCH (s:Signal)
            WHERE s.emotions IS NOT NULL
                  AND s.emotions <> '[]'
                  AND s.emotions <> ''
                  AND NOT (s)-[:EXPRESSES_EMOTION]->(:Emotion)
                  {ws_clause}
            RETURN s.uid AS uid, s.emotions AS emotions
        """
        params = {"ws": workspace_filter} if workspace_filter else {}
        rows, cols = cypher_query(find_q, params)
        columns = cols if isinstance(cols, list) else []

        signals_to_process = []
        for row in rows:
            d = dict(zip(columns, row)) if columns else {}
            sig_uid = d.get("uid")
            raw = d.get("emotions")
            if not sig_uid or not raw:
                continue
            signals_to_process.append((sig_uid, raw))

        self.stdout.write(f"  Found {len(signals_to_process)} Signals with emotions to materialize")

        if dry_run:
            # Show sample
            for uid, raw in signals_to_process[:5]:
                try:
                    parsed = json.loads(raw) if isinstance(raw, str) else raw
                    names = [
                        (e.get("emotion") if isinstance(e, dict) else str(e))
                        for e in (parsed if isinstance(parsed, list) else [])
                    ]
                    self.stdout.write(f"    Signal {uid[:12]}...: {names}")
                except Exception:
                    self.stdout.write(f"    Signal {uid[:12]}...: [unparseable]")
            return

        created_count = 0
        linked_count = 0

        for sig_uid, raw in signals_to_process:
            try:
                parsed = json.loads(raw) if isinstance(raw, str) else raw
                if not isinstance(parsed, list):
                    continue
            except (json.JSONDecodeError, TypeError):
                continue

            for emo in parsed:
                if isinstance(emo, dict):
                    emo_name = (emo.get("emotion") or "").strip()
                    valence = emo.get("intensity")
                    description = emo.get("source_description")
                elif isinstance(emo, str):
                    emo_name = emo.strip()
                    valence = None
                    description = None
                else:
                    continue

                if not emo_name:
                    continue

                merge_q = """
                    MATCH (s:Signal {uid: $sig_uid})
                    MERGE (e:Emotion {name: $name})
                    ON CREATE SET e.uid = randomUUID(),
                                  e.created_at = datetime()
                    SET e.valence = coalesce($valence, e.valence),
                        e.description = coalesce($description, e.description)
                    MERGE (s)-[:EXPRESSES_EMOTION]->(e)
                    RETURN e.uid AS euid
                """
                try:
                    result_rows, _ = cypher_query(merge_q, {
                        "sig_uid": sig_uid,
                        "name": emo_name,
                        "valence": valence,
                        "description": description,
                    })
                    if result_rows:
                        linked_count += 1
                except Exception as e:
                    self.stderr.write(f"  Error linking emotion '{emo_name}' to Signal {sig_uid[:12]}: {e}")

        self.stdout.write(self.style.SUCCESS(
            f"  Created/linked {linked_count} Emotion↔Signal relationships"
        ))
