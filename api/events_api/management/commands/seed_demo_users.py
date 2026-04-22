"""
Seed demo users with pre-populated Neo4j graph data.

Creates 3 demo users (demo_alex, demo_jordan, demo_sam) with realistic
conversation histories, signals, clusters, insights, and bias flags so
visitors can explore the full power of the graph without conversing first.

Usage:
    cd apps/api
    python manage.py seed_demo_users [--reset]

Zone 1: Internal seed tooling — never share externally.
"""

import logging
import time
import uuid

from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand

from events_api.graph_sync import full_user_graph_sync
from events_api.models import ensure_personal_workspace
from events_api.neo4j_client import cypher_query

User = get_user_model()
logger = logging.getLogger(__name__)


def _ts():
    """Return current epoch timestamp as float."""
    return time.time()


# ──────────────────────────────────────────────────────────────────────
# Demo User Definitions
# ──────────────────────────────────────────────────────────────────────

DEMO_USERS = [
    {
        "username": "demo_alex",
        "display_name": "Alex",
        "theme": "Workplace Conflict",
    },
    {
        "username": "demo_jordan",
        "display_name": "Jordan",
        "theme": "Relationship Decision",
    },
    {
        "username": "demo_sam",
        "display_name": "Sam",
        "theme": "Exploratory Journaling",
    },
]


# ──────────────────────────────────────────────────────────────────────
# Graph Data Builders
# ──────────────────────────────────────────────────────────────────────


def _seed_alex(workspace_id, owner_user_id):
    """
    Alex — Recurring conflict with manager.
    8 conversation turns, 5 signals, 1 cluster, 1 insight, bias flags.
    """
    conv_id = str(uuid.uuid4())
    now = _ts()

    # Conversation node
    cypher_query(
        """
        MERGE (c:Conversation {conversation_id: $conv_id, workspace_id: $ws})
        ON CREATE SET
            c.uid = randomUUID(),
            c.title = 'Workplace frustration',
            c.turn_count = 8,
            c.owner_user_id = $uid,
            c.last_active = $now,
            c.create_time = $now - 86400
        """,
        {"conv_id": conv_id, "ws": workspace_id, "uid": owner_user_id, "now": now},
    )

    # User turns
    turns = [
        ("user", "My manager dismissed my idea in front of the whole team yesterday. I felt humiliated and angry, but I just sat there and said nothing."),
        ("assistant", "That sounds like it hit hard — being publicly dismissed can activate both shame and anger simultaneously. The silence afterward might be telling too. What kept you from speaking up in that moment?"),
        ("user", "I guess I was afraid of making it worse. He's done this before, about three weeks ago with a different proposal. Same thing — just shut me down without even reading it."),
        ("assistant", "So there's a pattern here — this isn't the first time you've felt silenced by him. I notice you described both incidents with the same dynamic: you bring something forward, it gets dismissed without real consideration, and you absorb the impact silently. What happens to that anger afterward?"),
        ("user", "It just sits there. I go home and vent to my partner about it, but honestly she's getting tired of hearing about it. Last night she just went quiet when I started talking about work."),
        ("assistant", "That's worth noticing — the frustration from work is rippling into your relationship. Your partner's withdrawal might be creating a second layer of feeling unheard. You're experiencing dismissal at work, then a version of it at home too."),
        ("user", "I hadn't thought of it that way. Yeah, it does feel like nobody is really listening to me right now."),
        ("assistant", "That feeling of not being heard — it's showing up across contexts, with your manager and with your partner. The situations are different but the emotional core is similar. What would it look like to address the pattern with your manager directly, rather than just absorbing it?"),
    ]

    for i, (role, content) in enumerate(turns):
        label = "UserTurn" if role == "user" else "AssistantTurn"
        turn_uid = str(uuid.uuid4())
        cypher_query(
            f"""
            MATCH (c:Conversation {{conversation_id: $conv_id, workspace_id: $ws}})
            MERGE (t:{label} {{uid: $turn_uid, workspace_id: $ws}})
            ON CREATE SET
                t.content = $content,
                t.content_preview = $preview,
                t.role = $role,
                t.create_time = $ts,
                t.owner_user_id = $uid,
                t.conversation_id = $conv_id
            MERGE (c)-[:CONTAINS]->(t)
            """,
            {
                "conv_id": conv_id,
                "ws": workspace_id,
                "turn_uid": turn_uid,
                "content": content,
                "preview": content[:120],
                "role": role,
                "ts": now - 86400 + (i * 60),
                "uid": owner_user_id,
            },
        )

    # Signals
    signals = [
        {
            "address": "SA(work, manager, dismissal, yesterday)",
            "emotions": [
                {"emotion": "frustration", "intensity": 7, "confidence": 0.85},
                {"emotion": "shame", "intensity": 5, "confidence": 0.7},
            ],
            "confidence": 0.82,
            "bias_flags": [],
            "preview": "Manager dismissed my idea in front of the whole team",
        },
        {
            "address": "SA(work, manager, dismissal, 3_weeks_ago)",
            "emotions": [
                {"emotion": "anger", "intensity": 6, "confidence": 0.8},
            ],
            "confidence": 0.78,
            "bias_flags": ["rumination_amplification"],
            "preview": "Same thing happened three weeks ago with a different proposal",
        },
        {
            "address": "SA(work, manager, silence, recurring)",
            "emotions": [
                {"emotion": "helplessness", "intensity": 6, "confidence": 0.75},
            ],
            "confidence": 0.7,
            "bias_flags": [],
            "preview": "I just sat there and said nothing — afraid of making it worse",
        },
        {
            "address": "SA(home, partner, withdrawal, last_night)",
            "emotions": [
                {"emotion": "sadness", "intensity": 5, "confidence": 0.72},
                {"emotion": "guilt", "intensity": 4, "confidence": 0.6},
            ],
            "confidence": 0.68,
            "bias_flags": ["projection"],
            "preview": "Partner went quiet when I started talking about work again",
        },
        {
            "address": "SA(self, *, unheard, recurring)",
            "emotions": [
                {"emotion": "loneliness", "intensity": 7, "confidence": 0.8},
            ],
            "confidence": 0.75,
            "bias_flags": ["rumination_amplification"],
            "preview": "Nobody is really listening to me right now",
        },
    ]

    for sig in signals:
        sig_uid = str(uuid.uuid4())
        cypher_query(
            """
            MATCH (c:Conversation {conversation_id: $conv_id, workspace_id: $ws})
            MERGE (s:Signal {uid: $sig_uid, workspace_id: $ws})
            ON CREATE SET
                s.signal_address = $address,
                s.confidence = $confidence,
                s.content_preview = $preview,
                s.observation_bias_flags = $bias_flags,
                s.create_time = $ts,
                s.owner_user_id = $uid,
                s.conversation_id = $conv_id
            MERGE (c)-[:PRODUCED]->(s)
            """,
            {
                "conv_id": conv_id,
                "ws": workspace_id,
                "sig_uid": sig_uid,
                "address": sig["address"],
                "confidence": sig["confidence"],
                "preview": sig["preview"],
                "bias_flags": [f for f in sig["bias_flags"]],
                "ts": now - 86000,
                "uid": owner_user_id,
            },
        )

        # Emotion nodes linked to signals
        for emo in sig["emotions"]:
            emo_uid = str(uuid.uuid4())
            cypher_query(
                """
                MATCH (s:Signal {uid: $sig_uid, workspace_id: $ws})
                MERGE (e:Emotion {uid: $emo_uid, workspace_id: $ws})
                ON CREATE SET
                    e.emotion = $emotion,
                    e.intensity = $intensity,
                    e.confidence = $confidence,
                    e.create_time = $ts
                MERGE (s)-[:HAS_EMOTION]->(e)
                """,
                {
                    "sig_uid": sig_uid,
                    "ws": workspace_id,
                    "emo_uid": emo_uid,
                    "emotion": emo["emotion"],
                    "intensity": emo["intensity"],
                    "confidence": emo["confidence"],
                    "ts": now - 86000,
                },
            )

    # Person nodes
    for person_name in ["manager", "partner"]:
        cypher_query(
            """
            MERGE (p:Person {name: $name, workspace_id: $ws})
            ON CREATE SET
                p.uid = randomUUID(),
                p.normalized = $name,
                p.create_time = $ts
            """,
            {"name": person_name, "ws": workspace_id, "ts": now - 86400},
        )

    # Cluster
    cluster_uid = str(uuid.uuid4())
    cypher_query(
        """
        MERGE (cl:Cluster {uid: $cl_uid, workspace_id: $ws})
        ON CREATE SET
            cl.cluster_type = 'same_person_diff_time',
            cl.shared_coordinates = '{"person": ["manager"]}',
            cl.divergent_dimensions = '{"temporal": ["yesterday", "3_weeks_ago"]}',
            cl.strength = 0.82,
            cl.member_count = 3,
            cl.description = 'Recurring dismissal by manager across multiple time points',
            cl.create_time = $ts,
            cl.owner_user_id = $uid
        """,
        {"cl_uid": cluster_uid, "ws": workspace_id, "ts": now - 43200, "uid": owner_user_id},
    )

    # Insight
    insight_uid = str(uuid.uuid4())
    cypher_query(
        """
        MERGE (ins:Insight {uid: $ins_uid, workspace_id: $ws})
        ON CREATE SET
            ins.detection_type = 'recurring_pattern',
            ins.description = 'Dismissal from authority figures consistently triggers shame followed by withdrawal — both at work and at home',
            ins.confidence = 0.78,
            ins.create_time = $ts,
            ins.owner_user_id = $uid
        """,
        {"ins_uid": insight_uid, "ws": workspace_id, "ts": now - 21600, "uid": owner_user_id},
    )

    return {"conversations": 1, "turns": len(turns), "signals": len(signals)}


def _seed_jordan(workspace_id, owner_user_id):
    """
    Jordan — Relationship decision.
    6 conversation turns, 4 signals across 2 contexts, 1 cluster, bias flags.
    """
    conv_id = str(uuid.uuid4())
    now = _ts()

    cypher_query(
        """
        MERGE (c:Conversation {conversation_id: $conv_id, workspace_id: $ws})
        ON CREATE SET
            c.uid = randomUUID(),
            c.title = 'Relationship crossroads',
            c.turn_count = 6,
            c.owner_user_id = $uid,
            c.last_active = $now,
            c.create_time = $now - 172800
        """,
        {"conv_id": conv_id, "ws": workspace_id, "uid": owner_user_id, "now": now},
    )

    turns = [
        ("user", "My partner and I have been together for four years, but lately I feel like we're just going through the motions. I keep thinking about whether this is really what I want."),
        ("assistant", "Four years is significant — that's enough time for the relationship to have settled into patterns that might feel automatic. The fact that you're questioning it suggests something has shifted. When did you first notice this feeling?"),
        ("user", "Maybe six months ago? We moved in together and I thought it would bring us closer, but it feels like we're roommates now. My friend keeps telling me I should just leave, but I don't know if she's projecting her own stuff."),
        ("assistant", "You're noticing two things at once — your own uncertainty about the relationship, and skepticism about your friend's advice. That's actually a sophisticated emotional awareness. The roommate feeling after moving in is common, but your hesitation about leaving suggests there's still something pulling you to stay. What does that pull feel like?"),
        ("user", "Comfortable, maybe? Or maybe it's just fear of being alone. I honestly can't tell the difference anymore."),
        ("assistant", "That distinction — comfort versus fear of being alone — is one of the hardest to untangle. They can look identical from the inside. One question that sometimes helps: when you imagine your partner genuinely happy with someone else, what comes up? Not what you think you should feel, but the first unfiltered reaction."),
    ]

    for i, (role, content) in enumerate(turns):
        label = "UserTurn" if role == "user" else "AssistantTurn"
        turn_uid = str(uuid.uuid4())
        cypher_query(
            f"""
            MATCH (c:Conversation {{conversation_id: $conv_id, workspace_id: $ws}})
            MERGE (t:{label} {{uid: $turn_uid, workspace_id: $ws}})
            ON CREATE SET
                t.content = $content,
                t.content_preview = $preview,
                t.role = $role,
                t.create_time = $ts,
                t.owner_user_id = $uid,
                t.conversation_id = $conv_id
            MERGE (c)-[:CONTAINS]->(t)
            """,
            {
                "conv_id": conv_id,
                "ws": workspace_id,
                "turn_uid": turn_uid,
                "content": content,
                "preview": content[:120],
                "role": role,
                "ts": now - 172800 + (i * 60),
                "uid": owner_user_id,
            },
        )

    signals = [
        {
            "address": "SA(relationship, partner, stagnation, 6_months)",
            "emotions": [
                {"emotion": "uncertainty", "intensity": 6, "confidence": 0.8},
                {"emotion": "sadness", "intensity": 4, "confidence": 0.65},
            ],
            "confidence": 0.76,
            "bias_flags": [],
            "preview": "We're just going through the motions",
        },
        {
            "address": "SA(home, partner, disconnection, since_moving_in)",
            "emotions": [
                {"emotion": "disappointment", "intensity": 5, "confidence": 0.75},
            ],
            "confidence": 0.72,
            "bias_flags": ["narrative_construction"],
            "preview": "Feels like we're roommates now since moving in together",
        },
        {
            "address": "SA(social, friend, advice, recently)",
            "emotions": [
                {"emotion": "skepticism", "intensity": 4, "confidence": 0.7},
            ],
            "confidence": 0.65,
            "bias_flags": ["projection"],
            "preview": "Friend keeps telling me I should leave but might be projecting",
        },
        {
            "address": "SA(self, *, fear_vs_comfort, present)",
            "emotions": [
                {"emotion": "confusion", "intensity": 7, "confidence": 0.82},
                {"emotion": "fear", "intensity": 5, "confidence": 0.7},
            ],
            "confidence": 0.74,
            "bias_flags": [],
            "preview": "Can't tell if it's comfort or fear of being alone",
        },
    ]

    for sig in signals:
        sig_uid = str(uuid.uuid4())
        cypher_query(
            """
            MATCH (c:Conversation {conversation_id: $conv_id, workspace_id: $ws})
            MERGE (s:Signal {uid: $sig_uid, workspace_id: $ws})
            ON CREATE SET
                s.signal_address = $address,
                s.confidence = $confidence,
                s.content_preview = $preview,
                s.observation_bias_flags = $bias_flags,
                s.create_time = $ts,
                s.owner_user_id = $uid,
                s.conversation_id = $conv_id
            MERGE (c)-[:PRODUCED]->(s)
            """,
            {
                "conv_id": conv_id,
                "ws": workspace_id,
                "sig_uid": sig_uid,
                "address": sig["address"],
                "confidence": sig["confidence"],
                "preview": sig["preview"],
                "bias_flags": sig["bias_flags"],
                "ts": now - 172000,
                "uid": owner_user_id,
            },
        )

        for emo in sig["emotions"]:
            emo_uid = str(uuid.uuid4())
            cypher_query(
                """
                MATCH (s:Signal {uid: $sig_uid, workspace_id: $ws})
                MERGE (e:Emotion {uid: $emo_uid, workspace_id: $ws})
                ON CREATE SET
                    e.emotion = $emotion,
                    e.intensity = $intensity,
                    e.confidence = $confidence,
                    e.create_time = $ts
                MERGE (s)-[:HAS_EMOTION]->(e)
                """,
                {
                    "sig_uid": sig_uid,
                    "ws": workspace_id,
                    "emo_uid": emo_uid,
                    "emotion": emo["emotion"],
                    "intensity": emo["intensity"],
                    "confidence": emo["confidence"],
                    "ts": now - 172000,
                },
            )

    for person_name in ["partner", "friend"]:
        cypher_query(
            """
            MERGE (p:Person {name: $name, workspace_id: $ws})
            ON CREATE SET
                p.uid = randomUUID(),
                p.normalized = $name,
                p.create_time = $ts
            """,
            {"name": person_name, "ws": workspace_id, "ts": now - 172800},
        )

    # Cluster: same emotional tension across relationship and self contexts
    cluster_uid = str(uuid.uuid4())
    cypher_query(
        """
        MERGE (cl:Cluster {uid: $cl_uid, workspace_id: $ws})
        ON CREATE SET
            cl.cluster_type = 'cross_context_echo',
            cl.shared_coordinates = '{"emotion": ["uncertainty", "fear"]}',
            cl.divergent_dimensions = '{"context": ["relationship", "self"]}',
            cl.strength = 0.71,
            cl.member_count = 2,
            cl.description = 'Fear and uncertainty appearing in both relationship assessment and self-reflection',
            cl.create_time = $ts,
            cl.owner_user_id = $uid
        """,
        {"cl_uid": cluster_uid, "ws": workspace_id, "ts": now - 86400, "uid": owner_user_id},
    )

    return {"conversations": 1, "turns": len(turns), "signals": len(signals)}


def _seed_sam(workspace_id, owner_user_id):
    """
    Sam — Vague journaling.
    5 conversation turns, 3 signals with wildcard coordinates, sparse graph.
    """
    conv_id = str(uuid.uuid4())
    now = _ts()

    cypher_query(
        """
        MERGE (c:Conversation {conversation_id: $conv_id, workspace_id: $ws})
        ON CREATE SET
            c.uid = randomUUID(),
            c.title = 'Something feels off',
            c.turn_count = 5,
            c.owner_user_id = $uid,
            c.last_active = $now,
            c.create_time = $now - 259200
        """,
        {"conv_id": conv_id, "ws": workspace_id, "uid": owner_user_id, "now": now},
    )

    turns = [
        ("user", "I don't know, something just feels off lately. Like things aren't right but I can't put my finger on it."),
        ("assistant", "That vague sense of something being off — it's actually a really common starting point. Sometimes the feeling arrives before the clarity does. If you had to point to one area of your life where the off-ness feels strongest, what comes to mind first?"),
        ("user", "Maybe everything? Work is fine, relationship is fine, health is fine. But fine isn't good, you know?"),
        ("assistant", "Fine but not good — that's a real distinction. When everything is technically okay but nothing feels alive or meaningful, that can be its own kind of distress. It's harder to name because there's no obvious problem to point to. Has anything changed recently, even something small?"),
        ("user", "Not really. That's what makes it confusing. Nothing bad happened. I just woke up one day and realized I couldn't remember the last time I felt genuinely excited about something."),
    ]

    for i, (role, content) in enumerate(turns):
        label = "UserTurn" if role == "user" else "AssistantTurn"
        turn_uid = str(uuid.uuid4())
        cypher_query(
            f"""
            MATCH (c:Conversation {{conversation_id: $conv_id, workspace_id: $ws}})
            MERGE (t:{label} {{uid: $turn_uid, workspace_id: $ws}})
            ON CREATE SET
                t.content = $content,
                t.content_preview = $preview,
                t.role = $role,
                t.create_time = $ts,
                t.owner_user_id = $uid,
                t.conversation_id = $conv_id
            MERGE (c)-[:CONTAINS]->(t)
            """,
            {
                "conv_id": conv_id,
                "ws": workspace_id,
                "turn_uid": turn_uid,
                "content": content,
                "preview": content[:120],
                "role": role,
                "ts": now - 259200 + (i * 60),
                "uid": owner_user_id,
            },
        )

    signals = [
        {
            "address": "SA(*, *, unease, lately)",
            "emotions": [
                {"emotion": "unease", "intensity": 4, "confidence": 0.6},
            ],
            "confidence": 0.55,
            "bias_flags": [],
            "preview": "Something just feels off lately",
        },
        {
            "address": "SA(*, *, flatness, present)",
            "emotions": [
                {"emotion": "numbness", "intensity": 5, "confidence": 0.65},
            ],
            "confidence": 0.6,
            "bias_flags": [],
            "preview": "Fine isn't good — everything is technically okay but nothing feels alive",
        },
        {
            "address": "SA(self, *, loss_of_excitement, gradual)",
            "emotions": [
                {"emotion": "emptiness", "intensity": 5, "confidence": 0.7},
                {"emotion": "confusion", "intensity": 4, "confidence": 0.6},
            ],
            "confidence": 0.62,
            "bias_flags": [],
            "preview": "Can't remember the last time I felt genuinely excited about something",
        },
    ]

    for sig in signals:
        sig_uid = str(uuid.uuid4())
        cypher_query(
            """
            MATCH (c:Conversation {conversation_id: $conv_id, workspace_id: $ws})
            MERGE (s:Signal {uid: $sig_uid, workspace_id: $ws})
            ON CREATE SET
                s.signal_address = $address,
                s.confidence = $confidence,
                s.content_preview = $preview,
                s.observation_bias_flags = $bias_flags,
                s.create_time = $ts,
                s.owner_user_id = $uid,
                s.conversation_id = $conv_id
            MERGE (c)-[:PRODUCED]->(s)
            """,
            {
                "conv_id": conv_id,
                "ws": workspace_id,
                "sig_uid": sig_uid,
                "address": sig["address"],
                "confidence": sig["confidence"],
                "preview": sig["preview"],
                "bias_flags": sig["bias_flags"],
                "ts": now - 259000,
                "uid": owner_user_id,
            },
        )

        for emo in sig["emotions"]:
            emo_uid = str(uuid.uuid4())
            cypher_query(
                """
                MATCH (s:Signal {uid: $sig_uid, workspace_id: $ws})
                MERGE (e:Emotion {uid: $emo_uid, workspace_id: $ws})
                ON CREATE SET
                    e.emotion = $emotion,
                    e.intensity = $intensity,
                    e.confidence = $confidence,
                    e.create_time = $ts
                MERGE (s)-[:HAS_EMOTION]->(e)
                """,
                {
                    "sig_uid": sig_uid,
                    "ws": workspace_id,
                    "emo_uid": emo_uid,
                    "emotion": emo["emotion"],
                    "intensity": emo["intensity"],
                    "confidence": emo["confidence"],
                    "ts": now - 259000,
                },
            )

    return {"conversations": 1, "turns": len(turns), "signals": len(signals)}


# ──────────────────────────────────────────────────────────────────────
# Management Command
# ──────────────────────────────────────────────────────────────────────


SEED_FUNCTIONS = {
    "demo_alex": _seed_alex,
    "demo_jordan": _seed_jordan,
    "demo_sam": _seed_sam,
}


class Command(BaseCommand):
    help = "Seed demo users with pre-populated Neo4j graph data for visitor exploration."

    def add_arguments(self, parser):
        parser.add_argument(
            "--reset",
            action="store_true",
            help="Delete existing demo user graph data before re-seeding.",
        )

    def handle(self, *args, **options):
        reset = options["reset"]

        for profile in DEMO_USERS:
            username = profile["username"]
            seed_fn = SEED_FUNCTIONS[username]

            self.stdout.write(f"\n{'─' * 50}")
            self.stdout.write(f"Processing: {username} ({profile['theme']})")

            # Get or create Django user
            user, created = User.objects.get_or_create(
                username=username,
                defaults={"email": ""},
            )
            if created:
                user.set_unusable_password()
                user.save(update_fields=["password"])
                self.stdout.write(self.style.SUCCESS(f"  Created user: {username}"))
            else:
                self.stdout.write(f"  User exists: {username}")

            # Ensure workspace
            workspace = ensure_personal_workspace(user)
            workspace_id = str(workspace.id)
            self.stdout.write(f"  Workspace: {workspace.name} ({workspace_id})")

            # Reset graph data if requested
            if reset:
                from events_api.graph_sync import delete_workspace_graph_data
                result = delete_workspace_graph_data(workspace_id=workspace_id)
                self.stdout.write(
                    self.style.WARNING(f"  Reset: deleted {result.get('deleted_nodes', 0)} graph nodes")
                )

            # Sync UserProfile to graph
            full_user_graph_sync(user, workspace)
            self.stdout.write(f"  UserProfile synced to Neo4j")

            # Seed graph data
            stats = seed_fn(workspace_id, user.id)
            self.stdout.write(
                self.style.SUCCESS(
                    f"  Seeded: {stats['conversations']} conversation(s), "
                    f"{stats['turns']} turns, {stats['signals']} signals"
                )
            )

        self.stdout.write(f"\n{'─' * 50}")
        self.stdout.write(self.style.SUCCESS("Demo user seeding complete."))
