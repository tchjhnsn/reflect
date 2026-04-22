import json
from collections import Counter
from pathlib import Path

from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction
from django.utils import timezone
from django.utils.dateparse import parse_datetime

from events_api.coordinate_system import WILDCARD, build_signal_address
from events_api.identity import build_self_alias_names, normalize_graph_person_name
from events_api.live_graph import extract_live_topics
from events_api.models import Event, PatternRun, ensure_personal_workspace
from events_api.neo4j_client import cypher_query

GRAPH_PLATFORM = "signal-import"


def _normalize_string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []

    cleaned: list[str] = []
    for item in value:
        if isinstance(item, str):
            normalized = " ".join(item.strip().split())
            if normalized:
                cleaned.append(normalized)
    return cleaned


def _normalize_topic_label(value: str) -> str:
    return " ".join((value or "").strip().lower().split())


def _build_event_text(item: dict) -> str:
    title = " ".join(str(item.get("title", "")).split()).strip()
    body = str(item.get("text", "")).strip()
    if title and body:
        return f"{title}\n\n{body}"
    return title or body


def _build_context_tags(item: dict) -> list[str]:
    marker = f"import_id:{item['id']}"
    tags = [marker]
    tags.extend(_normalize_string_list(item.get("contexts")))
    tags.extend(f"action:{value}" for value in _normalize_string_list(item.get("actions")))
    tags.extend(f"emotion_tag:{value}" for value in _normalize_string_list(item.get("emotions")))
    return tags


def _extract_signal_topics(item: dict, *, max_topics: int = 12) -> list[dict[str, int | str]]:
    text = _build_event_text(item)
    counts: Counter[str] = Counter()

    for topic in extract_live_topics(text, max_topics=max_topics):
        word = _normalize_topic_label(str(topic["word"]))
        if word:
            counts[word] += int(topic["count"])

    for value in _normalize_string_list(item.get("contexts")):
        counts[_normalize_topic_label(value)] += 3
    for value in _normalize_string_list(item.get("actions")):
        counts[_normalize_topic_label(value)] += 2
    for value in _normalize_string_list(item.get("emotions")):
        counts[_normalize_topic_label(value)] += 2

    return [
        {"word": word, "count": count}
        for word, count in counts.most_common(max_topics)
        if word
    ]


def _find_existing_event(*, workspace, marker: str) -> Event | None:
    queryset = Event.objects.filter(workspace=workspace, source=Event.SOURCE_IMPORT).only("id", "context_tags")
    for event in queryset:
        if marker in (event.context_tags or []):
            return event
    return None


def _parse_signal_file(file_path: Path) -> list[dict]:
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CommandError(f"Invalid JSON in {file_path}: {exc}") from exc

    if not isinstance(payload, list):
        raise CommandError("Signal import file must contain a JSON array.")

    validated: list[dict] = []
    for index, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            raise CommandError(f"Record {index} must be an object.")
        if not isinstance(item.get("id"), str) or not item["id"].strip():
            raise CommandError(f"Record {index} is missing a valid 'id'.")
        if not isinstance(item.get("date"), str) or parse_datetime(item["date"]) is None:
            raise CommandError(f"Record {index} has an invalid 'date': {item.get('date')!r}")
        if not isinstance(item.get("title"), str) or not item["title"].strip():
            raise CommandError(f"Record {index} is missing a valid 'title'.")
        if not isinstance(item.get("text"), str) or not item["text"].strip():
            raise CommandError(f"Record {index} is missing a valid 'text'.")
        validated.append(item)
    return validated


def _signal_people(item: dict, *, username: str) -> list[str]:
    people = []
    for raw_person in _normalize_string_list(item.get("people")):
        person_name = normalize_graph_person_name(raw_person, current_username=username)
        if person_name:
            people.append(person_name)
    return list(dict.fromkeys(people))


def _signal_contexts(item: dict) -> list[str]:
    return list(dict.fromkeys(_normalize_string_list(item.get("contexts"))))


def _signal_actions(item: dict) -> list[str]:
    return list(dict.fromkeys(_normalize_string_list(item.get("actions"))))


def _signal_emotions(item: dict) -> list[str]:
    return list(dict.fromkeys(_normalize_string_list(item.get("emotions"))))


def _temporal_coordinate_label(occurred_at) -> str:
    return occurred_at.date().isoformat()


def _upsert_import_signal_graph(
    *,
    workspace_id: str,
    owner_user_id: int,
    username: str,
    conversation_id: str,
    item: dict,
    text: str,
    preview: str,
    occurred_at,
) -> None:
    contexts = _signal_contexts(item)
    people = _signal_people(item, username=username)
    actions = _signal_actions(item)
    emotions = _signal_emotions(item)
    temporal_label = _temporal_coordinate_label(occurred_at)
    signal_address = build_signal_address(
        context=contexts[0] if contexts else WILDCARD,
        person=people[0] if people else WILDCARD,
        action=actions[0] if actions else WILDCARD,
        temporal=temporal_label or WILDCARD,
        emotion=emotions[0] if emotions else None,
    )
    signal_payload = [
        {"emotion": emotion, "confidence": 1.0, "source_description": text[:200]}
        for emotion in emotions
    ]

    cypher_query(
        """
        MERGE (s:Signal {source_import_id: $source_import_id, workspace_id: $workspace_id})
        ON CREATE SET s.uid = randomUUID(),
                      s.owner_user_id = $owner_user_id,
                      s.created_at = datetime($occurred_at)
        SET s.signal_address = $signal_address,
            s.emotions = $emotions_json,
            s.confidence_score = 1.0,
            s.provenance = 'import',
            s.observation_bias_flags = '[]',
            s.embedding = '[]',
            s.is_resolved = $is_resolved,
            s.content_preview = $preview
        WITH s
        MATCH (c:Conversation {conversation_id: $conversation_id, workspace_id: $workspace_id})
        MERGE (c)-[:CONTAINS_SIGNAL]->(s)
        """,
        {
            "source_import_id": item["id"],
            "workspace_id": workspace_id,
            "owner_user_id": owner_user_id,
            "occurred_at": occurred_at.isoformat(),
            "signal_address": signal_address,
            "emotions_json": json.dumps(signal_payload),
            "is_resolved": bool(contexts or people or actions),
            "preview": preview[:200],
            "conversation_id": conversation_id,
        },
    )

    cypher_query(
        """
        MATCH (s:Signal {source_import_id: $source_import_id, workspace_id: $workspace_id})
        OPTIONAL MATCH (s)-[coord_rel:IN_CONTEXT|INVOLVES_ACTION|AT_TIME|EXPRESSES_EMOTION]->()
        DELETE coord_rel
        WITH s
        OPTIONAL MATCH (:Person {workspace_id: $workspace_id})-[person_rel:PARTICIPANT_IN]->(s)
        DELETE person_rel
        """,
        {
            "source_import_id": item["id"],
            "workspace_id": workspace_id,
        },
    )

    for person_name in people:
        cypher_query(
            """
            MATCH (s:Signal {source_import_id: $source_import_id, workspace_id: $workspace_id})
            MERGE (p:Person {name: $name, workspace_id: $workspace_id})
            ON CREATE SET p.uid = randomUUID(),
                          p.owner_user_id = $owner_user_id,
                          p.role = 'mentioned',
                          p.role_type = 'specific_person',
                          p.created_at = datetime()
            ON MATCH SET p.owner_user_id = $owner_user_id,
                         p.role = coalesce(p.role, 'mentioned'),
                         p.role_type = coalesce(p.role_type, 'specific_person')
            MERGE (p)-[:PARTICIPANT_IN {role: 'primary_actor'}]->(s)
            """,
            {
                "source_import_id": item["id"],
                "workspace_id": workspace_id,
                "owner_user_id": owner_user_id,
                "name": person_name,
            },
        )

    for context_name in contexts:
        cypher_query(
            """
            MATCH (s:Signal {source_import_id: $source_import_id, workspace_id: $workspace_id})
            MERGE (c:ContextNode {name: $name, workspace_id: $workspace_id})
            ON CREATE SET c.uid = randomUUID(),
                          c.owner_user_id = $owner_user_id,
                          c.created_at = datetime()
            ON MATCH SET c.owner_user_id = coalesce(c.owner_user_id, $owner_user_id)
            MERGE (s)-[:IN_CONTEXT]->(c)
            """,
            {
                "source_import_id": item["id"],
                "workspace_id": workspace_id,
                "owner_user_id": owner_user_id,
                "name": context_name,
            },
        )

    for action_name in actions:
        cypher_query(
            """
            MATCH (s:Signal {source_import_id: $source_import_id, workspace_id: $workspace_id})
            MERGE (a:ActionNode {name: $name, workspace_id: $workspace_id})
            ON CREATE SET a.uid = randomUUID(),
                          a.owner_user_id = $owner_user_id,
                          a.created_at = datetime()
            ON MATCH SET a.owner_user_id = coalesce(a.owner_user_id, $owner_user_id)
            MERGE (s)-[:INVOLVES_ACTION]->(a)
            """,
            {
                "source_import_id": item["id"],
                "workspace_id": workspace_id,
                "owner_user_id": owner_user_id,
                "name": action_name,
            },
        )

    cypher_query(
        """
        MATCH (s:Signal {source_import_id: $source_import_id, workspace_id: $workspace_id})
        MERGE (t:TemporalNode {name: $name, workspace_id: $workspace_id})
        ON CREATE SET t.uid = randomUUID(),
                      t.owner_user_id = $owner_user_id,
                      t.created_at = datetime()
        ON MATCH SET t.owner_user_id = coalesce(t.owner_user_id, $owner_user_id)
        MERGE (s)-[:AT_TIME]->(t)
        """,
        {
            "source_import_id": item["id"],
            "workspace_id": workspace_id,
            "owner_user_id": owner_user_id,
            "name": temporal_label,
        },
    )

    for emotion_name in emotions:
        cypher_query(
            """
            MATCH (s:Signal {source_import_id: $source_import_id, workspace_id: $workspace_id})
            MERGE (e:Emotion {name: $name, workspace_id: $workspace_id})
            ON CREATE SET e.uid = randomUUID(),
                          e.owner_user_id = $owner_user_id,
                          e.created_at = datetime()
            ON MATCH SET e.owner_user_id = coalesce(e.owner_user_id, $owner_user_id)
            MERGE (s)-[:EXPRESSES_EMOTION]->(e)
            """,
            {
                "source_import_id": item["id"],
                "workspace_id": workspace_id,
                "owner_user_id": owner_user_id,
                "name": emotion_name,
            },
        )


def _sync_graph_reflection(*, workspace_id: str, owner_user_id: int, username: str, item: dict) -> None:
    conversation_id = f"signal-import:{item['id']}"
    occurred_at = parse_datetime(item["date"])
    if occurred_at is None:
        raise CommandError(f"Unparseable date: {item['date']}")
    if timezone.is_naive(occurred_at):
        occurred_at = timezone.make_aware(occurred_at)

    text = _build_event_text(item)
    preview = text[:500] if len(text) > 500 else text
    topics = _extract_signal_topics(item)
    topic_counts_json = json.dumps({topic["word"]: topic["count"] for topic in topics}, sort_keys=True)
    create_time = occurred_at.timestamp()

    existing_rows, _ = cypher_query(
        """
        MATCH (c:Conversation {conversation_id: $conversation_id, workspace_id: $workspace_id})
        RETURN c.import_topic_counts_json
        """,
        {"conversation_id": conversation_id, "workspace_id": workspace_id},
    )
    previous_topic_counts: dict[str, int] = {}
    if existing_rows and existing_rows[0][0]:
        try:
            previous_topic_counts = json.loads(existing_rows[0][0])
        except (TypeError, json.JSONDecodeError):
            previous_topic_counts = {}

    for word, count in previous_topic_counts.items():
        cypher_query(
            """
            MATCH (topic:Topic {word: $word, workspace_id: $workspace_id})
            OPTIONAL MATCH (topic)-[rel:DISCUSSED_IN]->(:Conversation {conversation_id: $conversation_id, workspace_id: $workspace_id})
            WITH topic, collect(rel) AS rels
            FOREACH (r IN rels | DELETE r)
            SET topic.total_count = CASE
                    WHEN coalesce(topic.total_count, 0) - $count < 0 THEN 0
                    ELSE coalesce(topic.total_count, 0) - $count
                END,
                topic.conversation_count = CASE
                    WHEN size(rels) = 0 THEN coalesce(topic.conversation_count, 0)
                    WHEN coalesce(topic.conversation_count, 0) - 1 < 0 THEN 0
                    ELSE coalesce(topic.conversation_count, 0) - 1
                END
            """,
            {
                "word": word,
                "count": int(count),
                "workspace_id": workspace_id,
                "conversation_id": conversation_id,
            },
        )

    cypher_query(
        """
        MERGE (src:DataSource {platform: $platform, workspace_id: $workspace_id})
        ON CREATE SET src.uid = randomUUID(),
                      src.owner_user_id = $owner_user_id,
                      src.imported_at = datetime()
        ON MATCH SET src.owner_user_id = $owner_user_id,
                     src.last_updated = datetime()

        MERGE (c:Conversation {conversation_id: $conversation_id, workspace_id: $workspace_id})
        ON CREATE SET c.uid = randomUUID()
        SET c.owner_user_id = $owner_user_id,
            c.title = $title,
            c.platform = $platform,
            c.source_type = 'import',
            c.create_time = $create_time,
            c.last_active = $create_time,
            c.turn_count = 1,
            c.import_source_id = $import_source_id,
            c.import_topic_counts_json = $topic_counts_json

        MERGE (c)-[:IMPORTED_FROM]->(src)
        """,
        {
            "platform": GRAPH_PLATFORM,
            "workspace_id": workspace_id,
            "owner_user_id": owner_user_id,
            "conversation_id": conversation_id,
            "title": item["title"].strip(),
            "create_time": create_time,
            "import_source_id": item["id"],
            "topic_counts_json": topic_counts_json,
        },
    )

    cypher_query(
        """
        MATCH (c:Conversation {conversation_id: $conversation_id, workspace_id: $workspace_id})-[rel:CONTAINS]->(turn)
        DELETE rel, turn
        """,
        {"conversation_id": conversation_id, "workspace_id": workspace_id},
    )
    cypher_query(
        """
        MATCH (c:Conversation {conversation_id: $conversation_id, workspace_id: $workspace_id})-[rel:INVOLVES]->()
        DELETE rel
        """,
        {"conversation_id": conversation_id, "workspace_id": workspace_id},
    )

    cypher_query(
        """
        MATCH (c:Conversation {conversation_id: $conversation_id, workspace_id: $workspace_id})
        CREATE (turn:UserTurn {
            uid: randomUUID(),
            workspace_id: $workspace_id,
            owner_user_id: $owner_user_id,
            turn_number: 1,
            content_preview: $preview,
            word_count: $word_count,
            has_question: $has_question,
            create_time: $create_time
        })
        MERGE (c)-[:CONTAINS]->(turn)
        """,
        {
            "conversation_id": conversation_id,
            "workspace_id": workspace_id,
            "owner_user_id": owner_user_id,
            "preview": preview,
            "word_count": len(text.split()),
            "has_question": "?" in text,
            "create_time": create_time,
        },
    )

    people = _signal_people(item, username=username)

    for person_name in dict.fromkeys(people):
        cypher_query(
            """
            MATCH (c:Conversation {conversation_id: $conversation_id, workspace_id: $workspace_id})
            MERGE (p:Person {name: $name, workspace_id: $workspace_id})
            ON CREATE SET p.uid = randomUUID(),
                          p.owner_user_id = $owner_user_id,
                          p.role = 'mentioned',
                          p.role_type = 'specific_person',
                          p.created_at = datetime()
            ON MATCH SET p.owner_user_id = $owner_user_id,
                         p.role = coalesce(p.role, 'mentioned'),
                         p.role_type = coalesce(p.role_type, 'specific_person')
            MERGE (c)-[:INVOLVES]->(p)
            """,
            {
                "conversation_id": conversation_id,
                "workspace_id": workspace_id,
                "name": person_name,
                "owner_user_id": owner_user_id,
            },
        )

    _upsert_import_signal_graph(
        workspace_id=workspace_id,
        owner_user_id=owner_user_id,
        username=username,
        conversation_id=conversation_id,
        item=item,
        text=text,
        preview=preview,
        occurred_at=occurred_at,
    )

    cypher_query(
        """
        MERGE (u:UserProfile {workspace_id: $workspace_id, owner_user_id: $owner_user_id})
        ON CREATE SET u.uid = randomUUID(),
                      u.username = $username,
                      u.created_at = datetime()
        SET u.username = coalesce($username, u.username),
            u.last_active = datetime(),
            u.last_import = datetime()
        WITH u
        MATCH (c:Conversation {conversation_id: $conversation_id, workspace_id: $workspace_id})
        MERGE (u)-[:OWNS_CONVERSATION]->(c)
        """,
        {
            "conversation_id": conversation_id,
            "workspace_id": workspace_id,
            "owner_user_id": owner_user_id,
            "username": username,
        },
    )

    if topics:
        cypher_query(
            """
            UNWIND $topics AS topic_data
            MERGE (topic:Topic {word: topic_data.word, workspace_id: $workspace_id})
            ON CREATE SET topic.uid = randomUUID(),
                          topic.owner_user_id = $owner_user_id,
                          topic.name = topic_data.word,
                          topic.total_count = 0,
                          topic.conversation_count = 0
            SET topic.owner_user_id = $owner_user_id,
                topic.name = coalesce(topic.name, topic_data.word),
                topic.total_count = coalesce(topic.total_count, 0) + topic_data.count
            WITH topic, topic_data
            MATCH (c:Conversation {conversation_id: $conversation_id, workspace_id: $workspace_id})
            MERGE (topic)-[rel:DISCUSSED_IN]->(c)
            ON CREATE SET topic.conversation_count = coalesce(topic.conversation_count, 0) + 1
            """,
            {
                "topics": topics,
                "workspace_id": workspace_id,
                "owner_user_id": owner_user_id,
                "conversation_id": conversation_id,
            },
        )


def _refresh_graph_aggregates(*, workspace_id: str, owner_user_id: int, username: str) -> None:
    cypher_query(
        """
        MERGE (u:UserProfile {workspace_id: $workspace_id, owner_user_id: $owner_user_id})
        ON CREATE SET u.uid = randomUUID(),
                      u.username = $username,
                      u.created_at = datetime()
        SET u.owner_user_id = $owner_user_id,
            u.username = coalesce($username, u.username),
            u.last_import = datetime(),
            u.last_active = datetime()
        WITH u
        OPTIONAL MATCH (turn:UserTurn {workspace_id: $workspace_id})
        WITH u,
             count(turn) AS total_messages,
             avg(turn.word_count) AS avg_length,
             avg(CASE WHEN coalesce(turn.has_question, false) THEN 1.0 ELSE 0.0 END) AS question_ratio
        SET u.total_messages = total_messages,
            u.avg_message_length = coalesce(avg_length, 0.0),
            u.question_ratio = coalesce(question_ratio, 0.0)
        """,
        {
            "workspace_id": workspace_id,
            "owner_user_id": owner_user_id,
            "username": username,
        },
    )

    cypher_query(
        """
        MATCH (p:Person {workspace_id: $workspace_id})
        WHERE toLower(trim(p.name)) IN $self_aliases
        DETACH DELETE p
        """,
        {
            "workspace_id": workspace_id,
            "self_aliases": build_self_alias_names(current_username=username),
        },
    )

    cypher_query(
        """
        MATCH (src:DataSource {platform: $platform, workspace_id: $workspace_id})
        OPTIONAL MATCH (c:Conversation {platform: $platform, workspace_id: $workspace_id})-[:CONTAINS]->(turn:UserTurn)
        WITH src, count(DISTINCT c) AS total_conversations, count(turn) AS total_turns
        SET src.owner_user_id = $owner_user_id,
            src.total_conversations = total_conversations,
            src.total_turns = total_turns,
            src.last_updated = datetime()
        """,
        {
            "platform": GRAPH_PLATFORM,
            "workspace_id": workspace_id,
            "owner_user_id": owner_user_id,
        },
    )


class Command(BaseCommand):
    help = "Import structured reflection signals into a user's Postgres events and Neo4j insights graph."

    def add_arguments(self, parser):
        parser.add_argument("--username", required=True, help="Username to attach imported records to.")
        parser.add_argument("--file", required=True, help="Path to a JSON array file of signal records.")
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Validate the file and show what would be imported without writing anything.",
        )
        parser.add_argument(
            "--skip-graph",
            action="store_true",
            help="Only import Postgres events and skip the Neo4j graph sync.",
        )

    def handle(self, *args, **options):
        username = options["username"].strip()
        file_path = Path(options["file"]).expanduser().resolve()
        dry_run = options["dry_run"]
        skip_graph = options["skip_graph"]

        if not username:
            raise CommandError("--username is required.")
        if not file_path.exists():
            raise CommandError(f"File not found: {file_path}")

        records = _parse_signal_file(file_path)

        User = get_user_model()
        try:
            user = User.objects.get(username=username)
        except User.DoesNotExist as exc:
            raise CommandError(f"User not found: {username}") from exc

        workspace = ensure_personal_workspace(user)

        if dry_run:
            self.stdout.write(
                self.style.WARNING(
                    f"Dry run: validated {len(records)} records for {user.username} in workspace {workspace.id}"
                )
            )
            return

        created = 0
        updated = 0

        with transaction.atomic():
            for item in records:
                marker = f"import_id:{item['id']}"
                event_text = _build_event_text(item)
                emotions = _normalize_string_list(item.get("emotions"))
                actions = _normalize_string_list(item.get("actions"))
                payload = {
                    "workspace": workspace,
                    "created_by": user,
                    "occurred_at": parse_datetime(item["date"]),
                    "source": Event.SOURCE_IMPORT,
                    "text": event_text,
                    "context_tags": _build_context_tags(item),
                    "people": _normalize_string_list(item.get("people")),
                    "emotion": emotions[0] if emotions else None,
                    "reaction": ", ".join(actions) or None,
                    "outcome": str(item.get("outcome", "")).strip() or None,
                }

                existing = _find_existing_event(workspace=workspace, marker=marker)
                if existing is None:
                    Event.objects.create(**payload)
                    created += 1
                else:
                    for key, value in payload.items():
                        setattr(existing, key, value)
                    existing.save()
                    updated += 1

            PatternRun.objects.filter(workspace=workspace).delete()

        graph_synced = 0
        if not skip_graph:
            for item in records:
                _sync_graph_reflection(
                    workspace_id=str(workspace.id),
                    owner_user_id=user.id,
                    username=user.username,
                    item=item,
                )
                graph_synced += 1
            _refresh_graph_aggregates(
                workspace_id=str(workspace.id),
                owner_user_id=user.id,
                username=user.username,
            )

        self.stdout.write(
            self.style.SUCCESS(
                f"Imported {len(records)} records for {user.username} "
                f"(created={created}, updated={updated}, graph_synced={graph_synced})"
            )
        )
        self.stdout.write(f"workspace_id={workspace.id}")
