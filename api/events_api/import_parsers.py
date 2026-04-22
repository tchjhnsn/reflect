"""
ThriveSight — AI Conversation Import Parsers

Parses exported conversation data from ChatGPT, Claude, and Gemini into
ThriveSight's normalized conversation model, then writes to the Neo4j
knowledge graph.

Each parser normalizes platform-specific formats into a common structure:
    {
        "conversations": [
            {
                "title": str,
                "create_time": float (epoch),
                "platform": str,
                "turns": [
                    {
                        "role": "user" | "assistant" | "system",
                        "content": str,
                        "create_time": float | None,
                        "turn_number": int,
                    }
                ]
            }
        ],
        "metadata": {
            "platform": str,
            "total_conversations": int,
            "total_turns": int,
            "date_range": { "earliest": float, "latest": float },
        }
    }
"""

import json
import logging
import re
from collections import Counter
from typing import Optional

logger = logging.getLogger(__name__)


# ── Stop words for topic extraction ──────────────────────────────────

STOP_WORDS = frozenset(
    "i me my myself we our ours ourselves you your yours yourself yourselves "
    "he him his himself she her hers herself it its itself they them their "
    "theirs themselves what which who whom this that these those am is are was "
    "were be been being have has had having do does did doing a an the and but "
    "if or because as until while of at by for with about against between "
    "through during before after above below to from up down in out on off "
    "over under again further then once here there when where why how all both "
    "each few more most other some such no nor not only own same so than too "
    "very s t can will just don should now d ll m o re ve y ain aren couldn "
    "didn doesn hadn hasn haven isn ma mightn mustn needn shan shouldn wasn "
    "weren won wouldn could would also like get got going go know think want "
    "need really well even still much way things thing let yes right okay sure "
    "something anything everything nothing maybe actually pretty already always "
    "never sometimes often usually try trying tried use using used make makes "
    "making made one two three first last new good great best better ".split()
)


# ═══════════════════════════════════════════════════════════════════════
# ChatGPT Parser
# ═══════════════════════════════════════════════════════════════════════


def parse_chatgpt_export(raw_json: list) -> dict:
    """
    Parse ChatGPT's conversations.json export.

    ChatGPT exports a JSON array of conversation objects. Each conversation
    has a tree-structured `mapping` where nodes contain messages linked by
    parent/children references. We traverse from the root to `current_node`
    to extract the linear conversation.

    Args:
        raw_json: Parsed JSON array from ChatGPT's conversations.json

    Returns:
        Normalized conversation data dict.
    """
    conversations = []
    total_turns = 0
    earliest = float("inf")
    latest = float("-inf")

    for conv in raw_json:
        title = conv.get("title", "Untitled")
        create_time = conv.get("create_time")
        mapping = conv.get("mapping", {})
        current_node = conv.get("current_node")

        if not mapping:
            continue

        # Traverse the tree to extract linear message sequence
        turns = _traverse_chatgpt_mapping(mapping, current_node)

        if not turns:
            continue

        # Track date range
        if create_time:
            earliest = min(earliest, create_time)
            latest = max(latest, create_time)

        conversations.append({
            "title": title,
            "create_time": create_time,
            "platform": "chatgpt",
            "turns": turns,
        })
        total_turns += len(turns)

    return {
        "conversations": conversations,
        "metadata": {
            "platform": "chatgpt",
            "total_conversations": len(conversations),
            "total_turns": total_turns,
            "date_range": {
                "earliest": earliest if earliest != float("inf") else None,
                "latest": latest if latest != float("-inf") else None,
            },
        },
    }


def _traverse_chatgpt_mapping(mapping: dict, current_node: Optional[str]) -> list:
    """
    Traverse ChatGPT's tree-structured mapping to extract a linear
    conversation. Walks from root to current_node following the
    parent chain, then reverses to get chronological order.
    """
    if not current_node or current_node not in mapping:
        # Fallback: sort all messages by create_time
        return _fallback_sort_messages(mapping)

    # Walk backward from current_node to root
    chain = []
    node_id = current_node
    visited = set()

    while node_id and node_id in mapping and node_id not in visited:
        visited.add(node_id)
        node = mapping[node_id]
        msg = node.get("message")
        if msg:
            chain.append(msg)
        node_id = node.get("parent")

    # Reverse to get chronological order
    chain.reverse()

    # Convert to normalized turns
    turns = []
    turn_number = 0
    for msg in chain:
        author = msg.get("author", {})
        role = author.get("role", "unknown")

        # Skip system messages and tool results
        if role not in ("user", "assistant"):
            continue

        content = _extract_chatgpt_content(msg)
        if not content or not content.strip():
            continue

        turn_number += 1
        turns.append({
            "role": role,
            "content": content.strip(),
            "create_time": msg.get("create_time"),
            "turn_number": turn_number,
        })

    return turns


def _fallback_sort_messages(mapping: dict) -> list:
    """Sort all messages by create_time when tree traversal isn't possible."""
    messages = []
    for node in mapping.values():
        msg = node.get("message")
        if msg and msg.get("author", {}).get("role") in ("user", "assistant"):
            content = _extract_chatgpt_content(msg)
            if content and content.strip():
                messages.append(msg)

    messages.sort(key=lambda m: m.get("create_time") or 0)

    turns = []
    for i, msg in enumerate(messages, 1):
        turns.append({
            "role": msg["author"]["role"],
            "content": _extract_chatgpt_content(msg).strip(),
            "create_time": msg.get("create_time"),
            "turn_number": i,
        })
    return turns


def _extract_chatgpt_content(msg: dict) -> str:
    """Extract text content from a ChatGPT message, handling various content types."""
    content = msg.get("content", {})
    parts = content.get("parts", [])

    text_parts = []
    for part in parts:
        if isinstance(part, str):
            text_parts.append(part)
        elif isinstance(part, dict):
            # Handle multimodal content (images, etc.) — extract text if present
            if "text" in part:
                text_parts.append(part["text"])

    return "\n".join(text_parts)


# ═══════════════════════════════════════════════════════════════════════
# Claude Parser
# ═══════════════════════════════════════════════════════════════════════


def parse_claude_export(raw_json: list) -> dict:
    """
    Parse Claude's conversation export.

    Claude exports conversations as a JSON array. Each conversation has
    a `chat_messages` array with `sender` ("human"/"assistant") and `text`.

    Args:
        raw_json: Parsed JSON array from Claude's export

    Returns:
        Normalized conversation data dict.
    """
    conversations = []
    total_turns = 0
    earliest = float("inf")
    latest = float("-inf")

    for conv in raw_json:
        title = conv.get("name", conv.get("title", "Untitled"))
        create_time = conv.get("created_at") or conv.get("create_time")

        # Handle ISO timestamp strings
        if isinstance(create_time, str):
            try:
                from datetime import datetime
                dt = datetime.fromisoformat(create_time.replace("Z", "+00:00"))
                create_time = dt.timestamp()
            except (ValueError, TypeError):
                create_time = None

        messages = conv.get("chat_messages", [])
        if not messages:
            continue

        turns = []
        turn_number = 0
        for msg in messages:
            sender = msg.get("sender", "")
            role = "user" if sender == "human" else "assistant" if sender == "assistant" else None
            if not role:
                continue

            text = msg.get("text", "")
            if not text or not text.strip():
                continue

            turn_number += 1
            msg_time = msg.get("created_at") or msg.get("create_time")
            if isinstance(msg_time, str):
                try:
                    from datetime import datetime
                    dt = datetime.fromisoformat(msg_time.replace("Z", "+00:00"))
                    msg_time = dt.timestamp()
                except (ValueError, TypeError):
                    msg_time = None

            turns.append({
                "role": role,
                "content": text.strip(),
                "create_time": msg_time,
                "turn_number": turn_number,
            })

        if not turns:
            continue

        if create_time:
            earliest = min(earliest, create_time)
            latest = max(latest, create_time)

        conversations.append({
            "title": title,
            "create_time": create_time,
            "platform": "claude",
            "turns": turns,
        })
        total_turns += len(turns)

    return {
        "conversations": conversations,
        "metadata": {
            "platform": "claude",
            "total_conversations": len(conversations),
            "total_turns": total_turns,
            "date_range": {
                "earliest": earliest if earliest != float("inf") else None,
                "latest": latest if latest != float("-inf") else None,
            },
        },
    }


# ═══════════════════════════════════════════════════════════════════════
# Topic & Entity Extraction (lightweight, no LLM)
# ═══════════════════════════════════════════════════════════════════════


def extract_topics_from_conversations(normalized_data: dict) -> dict:
    """
    Extract topic nodes and basic entity data from normalized conversations.
    This is a lightweight extraction — frequency-based, no LLM calls.
    Used for initial graph population before deeper LLM analysis.

    Returns:
        {
            "topics": [ { "topic": str, "count": int, "conversations": [str] } ],
            "user_patterns": {
                "total_messages": int,
                "avg_message_length": float,
                "question_ratio": float,
                "top_bigrams": [ (str, int) ],
            }
        }
    """
    all_user_words = []
    all_user_bigrams = []
    user_message_count = 0
    user_total_length = 0
    question_count = 0
    topic_conv_map = {}  # word → set of conversation titles

    for conv in normalized_data.get("conversations", []):
        title = conv.get("title", "Untitled")
        for turn in conv.get("turns", []):
            if turn["role"] != "user":
                continue

            content = turn["content"]
            user_message_count += 1
            user_total_length += len(content)

            if "?" in content:
                question_count += 1

            # Tokenize
            words = _tokenize(content)
            meaningful = [w for w in words if w not in STOP_WORDS and len(w) > 2]
            all_user_words.extend(meaningful)

            for w in meaningful:
                if w not in topic_conv_map:
                    topic_conv_map[w] = set()
                topic_conv_map[w].add(title)

            # Bigrams
            for i in range(len(meaningful) - 1):
                bigram = f"{meaningful[i]} {meaningful[i+1]}"
                all_user_bigrams.append(bigram)

    # Count topics
    word_counts = Counter(all_user_words)
    bigram_counts = Counter(all_user_bigrams)

    # Build topic list — words that appear in multiple conversations are more topical
    topics = []
    for word, count in word_counts.most_common(100):
        convs = topic_conv_map.get(word, set())
        if len(convs) >= 2 or count >= 3:  # Appears in 2+ convos or 3+ times
            topics.append({
                "topic": word,
                "count": count,
                "conversation_count": len(convs),
                "conversations": sorted(convs)[:10],  # Limit for response size
            })

    return {
        "topics": topics[:50],  # Top 50 topics
        "user_patterns": {
            "total_messages": user_message_count,
            "avg_message_length": (
                user_total_length / user_message_count if user_message_count else 0
            ),
            "question_ratio": (
                question_count / user_message_count if user_message_count else 0
            ),
            "top_bigrams": bigram_counts.most_common(20),
        },
    }


def _tokenize(text: str) -> list:
    """Simple word tokenizer — lowercase, strip punctuation."""
    text = text.lower()
    text = re.sub(r"[\u2018\u2019\u201c\u201d]", "", text)  # Smart quotes
    words = re.findall(r"[a-z]+(?:'[a-z]+)?", text)
    return words


# ═══════════════════════════════════════════════════════════════════════
# Graph Writer — Write imported conversations to Neo4j
# ═══════════════════════════════════════════════════════════════════════


def write_import_to_graph(
    normalized_data: dict,
    topics_data: dict,
    *,
    workspace_id: str,
    owner_user_id: int,
) -> dict:
    """
    Write imported AI conversation data to the Neo4j knowledge graph.

    Creates:
    - A UserProfile node (or merges with existing)
    - Conversation nodes for each imported conversation
    - Turn nodes within conversations
    - Topic nodes from frequency analysis
    - DISCUSSED_IN edges connecting topics to conversations
    - TEMPORAL edges connecting sequential conversations

    Args:
        normalized_data: Output from parse_chatgpt_export() or parse_claude_export()
        topics_data: Output from extract_topics_from_conversations()

    Returns:
        Summary dict with counts of nodes/edges created.
    """
    from neomodel import db

    platform = normalized_data["metadata"]["platform"]
    summary = {"nodes_created": 0, "edges_created": 0, "errors": []}

    try:
        # 1. Create platform source node
        db.cypher_query(
            """
            MERGE (src:DataSource {platform: $platform, workspace_id: $workspace_id})
            ON CREATE SET src.uid = randomUUID(),
                          src.owner_user_id = $owner_user_id,
                          src.total_conversations = $total_convs,
                          src.total_turns = $total_turns,
                          src.imported_at = datetime()
            ON MATCH SET src.total_conversations = $total_convs,
                         src.total_turns = $total_turns,
                         src.last_updated = datetime()
            """,
            {
                "platform": platform,
                "workspace_id": workspace_id,
                "owner_user_id": owner_user_id,
                "total_convs": normalized_data["metadata"]["total_conversations"],
                "total_turns": normalized_data["metadata"]["total_turns"],
            },
        )
        summary["nodes_created"] += 1

        # 2. Batch create conversation nodes
        conv_params = []
        for i, conv in enumerate(normalized_data["conversations"]):
            conv_params.append({
                "title": conv["title"],
                "create_time": conv.get("create_time"),
                "platform": platform,
                "workspace_id": workspace_id,
                "owner_user_id": owner_user_id,
                "turn_count": len(conv["turns"]),
                "index": i,
            })

        if conv_params:
            # Chunk to avoid huge queries
            chunk_size = 50
            for i in range(0, len(conv_params), chunk_size):
                chunk = conv_params[i:i + chunk_size]
                db.cypher_query(
                    """
                    UNWIND $convs AS c
                    MERGE (conv:Conversation {title: c.title, platform: c.platform, workspace_id: c.workspace_id})
                    ON CREATE SET conv.uid = randomUUID(),
                                  conv.owner_user_id = c.owner_user_id,
                                  conv.create_time = c.create_time,
                                  conv.turn_count = c.turn_count
                    WITH conv, c
                    MATCH (src:DataSource {platform: c.platform, workspace_id: c.workspace_id})
                    MERGE (conv)-[:IMPORTED_FROM]->(src)
                    """,
                    {"convs": chunk},
                )
                summary["nodes_created"] += len(chunk)

        # 3. Batch create turn nodes (user turns only — they represent the user's voice)
        turn_params = []
        for conv in normalized_data["conversations"]:
            for turn in conv["turns"]:
                if turn["role"] == "user":
                    # Truncate content for graph storage (full text too large)
                    content = turn["content"]
                    preview = content[:500] if len(content) > 500 else content
                    turn_params.append({
                        "conv_title": conv["title"],
                        "platform": platform,
                        "workspace_id": workspace_id,
                        "owner_user_id": owner_user_id,
                        "turn_number": turn["turn_number"],
                        "content_preview": preview,
                        "word_count": len(content.split()),
                        "has_question": "?" in content,
                        "create_time": turn.get("create_time"),
                    })

        if turn_params:
            for i in range(0, len(turn_params), chunk_size):
                chunk = turn_params[i:i + chunk_size]
                db.cypher_query(
                    """
                    UNWIND $turns AS t
                    MATCH (conv:Conversation {title: t.conv_title, platform: t.platform, workspace_id: t.workspace_id})
                    CREATE (turn:UserTurn {
                        uid: randomUUID(),
                        workspace_id: t.workspace_id,
                        owner_user_id: t.owner_user_id,
                        turn_number: t.turn_number,
                        content_preview: t.content_preview,
                        word_count: t.word_count,
                        has_question: t.has_question,
                        create_time: t.create_time
                    })
                    MERGE (conv)-[:CONTAINS]->(turn)
                    """,
                    {"turns": chunk},
                )
                summary["nodes_created"] += len(chunk)

        # 4. Create topic nodes and connect to conversations
        topic_params = []
        for topic in topics_data.get("topics", []):
            topic_params.append({
                "word": topic["topic"],
                "workspace_id": workspace_id,
                "owner_user_id": owner_user_id,
                "count": topic["count"],
                "conversation_count": topic["conversation_count"],
                "conversations": topic["conversations"],
            })

        if topic_params:
            db.cypher_query(
                """
                UNWIND $topics AS t
                MERGE (topic:Topic {word: t.word, workspace_id: t.workspace_id})
                ON CREATE SET topic.uid = randomUUID(),
                              topic.owner_user_id = t.owner_user_id,
                              topic.total_count = t.count,
                              topic.conversation_count = t.conversation_count
                ON MATCH SET topic.total_count = t.count,
                             topic.conversation_count = t.conversation_count
                """,
                {"topics": topic_params},
            )
            summary["nodes_created"] += len(topic_params)

            # Connect topics to conversations
            topic_conv_edges = []
            for topic in topics_data.get("topics", []):
                for conv_title in topic["conversations"]:
                    topic_conv_edges.append({
                        "word": topic["topic"],
                        "conv_title": conv_title,
                        "platform": platform,
                        "workspace_id": workspace_id,
                    })

            if topic_conv_edges:
                for i in range(0, len(topic_conv_edges), chunk_size):
                    chunk = topic_conv_edges[i:i + chunk_size]
                    db.cypher_query(
                        """
                        UNWIND $edges AS e
                        MATCH (topic:Topic {word: e.word, workspace_id: e.workspace_id})
                        MATCH (conv:Conversation {title: e.conv_title, platform: e.platform, workspace_id: e.workspace_id})
                        MERGE (topic)-[:DISCUSSED_IN]->(conv)
                        """,
                        {"edges": chunk},
                    )
                    summary["edges_created"] += len(chunk)

        # 5. Store user patterns as a UserProfile node
        patterns = topics_data.get("user_patterns", {})
        db.cypher_query(
            """
            MERGE (u:UserProfile {workspace_id: $workspace_id, owner_user_id: $owner_user_id})
            ON CREATE SET u.uid = randomUUID(),
                          u.created_at = datetime()
            SET u.total_messages = $total_messages,
                u.owner_user_id = $owner_user_id,
                u.avg_message_length = $avg_length,
                u.question_ratio = $q_ratio,
                u.last_import = datetime()
            """,
            {
                "workspace_id": workspace_id,
                "owner_user_id": owner_user_id,
                "total_messages": patterns.get("total_messages", 0),
                "avg_length": patterns.get("avg_message_length", 0),
                "q_ratio": patterns.get("question_ratio", 0),
            },
        )
        summary["nodes_created"] += 1

        logger.info(f"Import graph write complete: {summary}")

    except Exception as e:
        logger.error(f"Import graph write failed: {e}")
        summary["errors"].append(str(e))

    return summary


# ═══════════════════════════════════════════════════════════════════════
# Auto-detect platform from export data
# ═══════════════════════════════════════════════════════════════════════


def detect_platform(raw_json) -> str:
    """
    Detect which AI platform produced an export based on JSON structure.

    Returns: "chatgpt", "claude", or "unknown"
    """
    if not isinstance(raw_json, list) or len(raw_json) == 0:
        return "unknown"

    sample = raw_json[0]

    # ChatGPT: has 'mapping' and 'current_node'
    if "mapping" in sample and "current_node" in sample:
        return "chatgpt"

    # Claude: has 'chat_messages' with 'sender' field
    if "chat_messages" in sample:
        msgs = sample["chat_messages"]
        if msgs and isinstance(msgs, list) and "sender" in msgs[0]:
            return "claude"

    return "unknown"
