from collections import Counter

from .import_parsers import STOP_WORDS, _tokenize
from .identity import build_self_alias_names
from .neo4j_client import cypher_query, get_driver, _get_connection_settings


def build_live_conversation_title(message: str) -> str:
    normalized = " ".join((message or "").split()).strip()
    if not normalized:
        return "Live reflection"
    preview = normalized[:50]
    suffix = "…" if len(normalized) > 50 else ""
    return f"Live: {preview}{suffix}"


def extract_live_topics(message: str, *, max_topics: int = 8) -> list[dict[str, int | str]]:
    meaningful = [
        word for word in _tokenize(message)
        if word not in STOP_WORDS and len(word) > 2
    ]
    counts = Counter(meaningful)
    return [
        {"word": word, "count": count}
        for word, count in counts.most_common(max_topics)
    ]


def _run_write_queries(queries_with_params):
    """Execute a set of Neo4j writes in one transaction for causal consistency."""
    _, _, database = _get_connection_settings()
    session_kwargs = {"database": database} if database else {}
    results = []

    with get_driver().session(**session_kwargs) as session:
        with session.begin_transaction() as tx:
            for query, params in queries_with_params:
                result = tx.run(query, parameters=params or {})
                columns = list(result.keys())
                rows = [list(record.values()) for record in result]
                results.append((rows, columns))
            tx.commit()

    return results


def write_live_conversation_to_graph(
    *,
    conversation_id: str,
    message: str,
    ai_response: str,
    workspace_id: str,
    owner_user_id: int,
    conversation_title: str | None = None,
    username: str = "",
    email: str = "",
) -> dict:
    title = conversation_title or build_live_conversation_title(message)
    user_preview = message[:500] if len(message) > 500 else message
    assistant_preview = ai_response[:500] if len(ai_response) > 500 else ai_response
    topics = extract_live_topics(message)
    common_params = {
        "conv_id": conversation_id,
        "workspace_id": workspace_id,
        "owner_user_id": owner_user_id,
        "self_aliases": build_self_alias_names(current_username=username),
    }

    queries = [
        (
            """
            MERGE (c:Conversation {conversation_id: $conv_id, workspace_id: $workspace_id})
            ON CREATE SET c.uid = randomUUID(),
                          c.owner_user_id = $owner_user_id,
                          c.title = $title,
                          c.platform = 'thrivesight',
                          c.create_time = timestamp() / 1000.0,
                          c.turn_count = 0
            SET c.last_active = timestamp() / 1000.0
            """,
            {
                **common_params,
                "title": title,
            },
        ),
        (
            """
            MATCH (c:Conversation {conversation_id: $conv_id, workspace_id: $workspace_id})
            CREATE (turn:UserTurn {
                uid: randomUUID(),
                workspace_id: $workspace_id,
                owner_user_id: $owner_user_id,
                content: $content,
                content_preview: $preview,
                word_count: $word_count,
                has_question: $has_q,
                create_time: timestamp() / 1000.0
            })
            MERGE (c)-[:CONTAINS]->(turn)
            SET c.turn_count = coalesce(c.turn_count, 0) + 1
            """,
            {
                **common_params,
                "content": message,
                "preview": user_preview,
                "word_count": len(message.split()),
                "has_q": "?" in message,
            },
        ),
        (
            """
            MATCH (c:Conversation {conversation_id: $conv_id, workspace_id: $workspace_id})
            CREATE (turn:AssistantTurn {
                uid: randomUUID(),
                workspace_id: $workspace_id,
                owner_user_id: $owner_user_id,
                content: $content,
                content_preview: $preview,
                create_time: timestamp() / 1000.0
            })
            MERGE (c)-[:CONTAINS]->(turn)
            SET c.turn_count = coalesce(c.turn_count, 0) + 1
            """,
            {
                **common_params,
                "content": ai_response,
                "preview": assistant_preview,
            },
        ),
        (
            """
            MERGE (src:DataSource {platform: 'thrivesight', workspace_id: $workspace_id})
            ON CREATE SET src.uid = randomUUID(),
                          src.owner_user_id = $owner_user_id,
                          src.imported_at = datetime()
            ON MATCH SET src.last_updated = datetime()
            WITH src
            MATCH (c:Conversation {conversation_id: $conv_id, workspace_id: $workspace_id})
            MERGE (c)-[:IMPORTED_FROM]->(src)
            """,
            common_params,
        ),
        (
            """
            MERGE (u:UserProfile {workspace_id: $workspace_id, owner_user_id: $owner_user_id})
            ON CREATE SET u.uid = randomUUID(),
                          u.username = $username,
                          u.email = $email,
                          u.created_at = datetime()
            SET u.total_messages = coalesce(u.total_messages, 0) + 1,
                u.username = coalesce($username, u.username),
                u.email = coalesce($email, u.email),
                u.last_active = datetime()
            WITH u
            MATCH (c:Conversation {conversation_id: $conv_id, workspace_id: $workspace_id})
            MERGE (u)-[:OWNS_CONVERSATION]->(c)
            """,
            {
                **common_params,
                "username": username or None,
                "email": email or None,
            },
        ),
        (
            """
            MATCH (p:Person {workspace_id: $workspace_id})
            WHERE toLower(trim(p.name)) IN $self_aliases
            DETACH DELETE p
            """,
            common_params,
        ),
    ]

    if topics:
        queries.append(
            (
                """
                UNWIND $topics AS t
                MERGE (topic:Topic {word: t.word, workspace_id: $workspace_id})
                ON CREATE SET topic.uid = randomUUID(),
                              topic.owner_user_id = $owner_user_id,
                              topic.total_count = 0,
                              topic.conversation_count = 0
                SET topic.total_count = coalesce(topic.total_count, 0) + t.count
                WITH topic
                MATCH (c:Conversation {conversation_id: $conv_id, workspace_id: $workspace_id})
                MERGE (topic)-[:DISCUSSED_IN]->(c)
                ON CREATE SET topic.conversation_count = coalesce(topic.conversation_count, 0) + 1
                """,
                {
                    **common_params,
                    "topics": topics,
                },
            )
        )

    queries.append(
        (
            """
            MATCH (c:Conversation {conversation_id: $conv_id, workspace_id: $workspace_id})
            OPTIONAL MATCH (c)-[:CONTAINS]->(turn)
            RETURN c.title AS title,
                   coalesce(c.turn_count, 0) AS turn_count,
                   count(turn) AS stored_turns
            """,
            common_params,
        )
    )

    results = _run_write_queries(queries)
    verification_rows, verification_columns = results[-1]
    verification = dict(zip(verification_columns, verification_rows[0])) if verification_rows else {}
    stored_turns = verification.get("stored_turns", 0) or 0
    turn_count = verification.get("turn_count", 0) or 0
    saved_title = verification.get("title") or title

    return {
        "updated": stored_turns >= 2 and turn_count >= 2,
        "conversation_title": saved_title,
        "topics": [topic["word"] for topic in topics],
        "topic_count": len(topics),
        "stored_turns": stored_turns,
    }


def write_pipeline_trace(
    *,
    workspace_id: str,
    conversation_id: str,
    entities: dict | None = None,
    persona: str | None = None,
    context_summary: str | None = None,
    signals_referenced: int = 0,
    clusters_referenced: int = 0,
    token_count: int = 0,
) -> None:
    """
    Persist a PipelineTrace node linked to the most recent UserTurn
    in the given conversation.

    This captures what the context assembly layer did for a single
    message — which entities were extracted, which persona was used,
    how many signals/clusters fed into the context packet, and how
    many tokens it consumed.
    """
    import json as _json

    entities_str = _json.dumps(entities or {})

    cypher_query(
        """
        MATCH (c:Conversation {conversation_id: $conv_id, workspace_id: $workspace_id})
              -[:CONTAINS]->(turn:UserTurn {workspace_id: $workspace_id})
        WITH turn ORDER BY turn.create_time DESC LIMIT 1
        CREATE (pt:PipelineTrace {
            uid: randomUUID(),
            workspace_id: $workspace_id,
            entities_extracted: $entities,
            persona_used: $persona,
            context_packet_summary: $summary,
            signals_referenced: $signal_count,
            clusters_referenced: $cluster_count,
            token_count: $token_count,
            created_at: timestamp() / 1000.0
        })
        CREATE (pt)-[:TRACES]->(turn)
        """,
        {
            "conv_id": conversation_id,
            "workspace_id": workspace_id,
            "entities": entities_str,
            "persona": persona or "neutral_observer",
            "summary": context_summary or "",
            "signal_count": signals_referenced,
            "cluster_count": clusters_referenced,
            "token_count": token_count,
        },
    )
