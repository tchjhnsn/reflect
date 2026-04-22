"""
ThriveSight Context Assembly — Entity Extraction and Context Packet Assembly.

The context assembly layer bridges the graph and the LLM by:
1. Extracting entities from user messages (rule-based first, LLM fallback)
2. Querying the graph for related signals, clusters, and insights
3. Assembling a token-efficient context packet for the LLM prompt
4. Compressing the packet to fit within a persona's token budget

Zone 1: Proprietary — internal module.
"""

import json
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Rule-Based Entity Extraction Patterns
# ──────────────────────────────────────────────────────────────────────────────

# Person roles and common references
_PERSON_PATTERNS = {
    "manager": ["manager", "boss", "supervisor", "lead"],
    "colleague": ["colleague", "coworker", "co-worker", "teammate"],
    "partner": ["partner", "spouse", "husband", "wife", "boyfriend", "girlfriend"],
    "friend": ["friend", "buddy", "pal"],
    "family": ["mother", "father", "mom", "dad", "sister", "brother", "parent",
               "daughter", "son", "aunt", "uncle", "grandparent", "grandmother",
               "grandfather", "cousin", "family"],
    "therapist": ["therapist", "counselor", "doctor", "psychiatrist"],
    "self": ["myself", "i myself", "me personally"],
}

# Context keywords
_CONTEXT_PATTERNS = {
    "work": ["work", "office", "meeting", "job", "workplace", "company",
             "project", "team", "client", "deadline", "standup"],
    "home": ["home", "house", "apartment", "kitchen", "bedroom", "living room"],
    "social": ["party", "gathering", "social", "event", "dinner", "bar",
               "restaurant", "outing"],
    "family": ["family dinner", "family gathering", "holiday", "thanksgiving",
               "christmas", "reunion"],
    "health": ["hospital", "doctor", "appointment", "therapy", "gym",
               "exercise", "health", "medication"],
    "self": ["journal", "alone", "by myself", "meditation", "reflection"],
}

# Action/trigger keywords mapping to trigger categories
_ACTION_PATTERNS = {
    "dismissal": ["dismissed", "dismissal", "ignored", "minimized", "brushed off",
                  "shut down", "overlooked", "not listening"],
    "accusation": ["blamed", "accusation", "accused", "your fault",
                   "you always", "you never"],
    "deflection": ["changed the subject", "deflected", "redirected",
                   "avoided", "dodged"],
    "withdrawal": ["walked away", "silent treatment", "withdrew", "shut down",
                   "stopped talking", "disengaged"],
    "demand": ["demanded", "insisted", "ultimatum", "must", "have to"],
    "questioning": ["asked", "questioned", "wondered", "inquired"],
    "validation": ["validated", "acknowledged", "understood", "heard me",
                   "recognized"],
    "acknowledgment": ["acknowledged", "noted", "recognized", "saw that"],
    "concession": ["agreed", "compromised", "yielded", "admitted",
                   "took responsibility"],
    "sarcasm": ["sarcastic", "mocking", "ironic", "snide"],
    "praise": ["praised", "complimented", "recognized", "thanked",
               "appreciated"],
    "interruption": ["interrupted", "cut off", "talked over", "spoke over"],
}

# Temporal keywords
_TEMPORAL_PATTERNS = {
    "yesterday": ["yesterday"],
    "today": ["today", "this morning", "this afternoon", "this evening",
              "tonight", "earlier today"],
    "last_week": ["last week", "a week ago"],
    "this_week": ["this week"],
    "last_month": ["last month", "a month ago"],
    "recently": ["recently", "lately", "the other day"],
    "always": ["always", "every time", "constantly", "repeatedly"],
    "monday": ["monday"],
    "tuesday": ["tuesday"],
    "wednesday": ["wednesday"],
    "thursday": ["thursday"],
    "friday": ["friday"],
    "weekend": ["weekend", "saturday", "sunday"],
    "morning": ["morning", "mornings"],
    "evening": ["evening", "evenings", "night", "nights"],
}


# ──────────────────────────────────────────────────────────────────────────────
# ContextAssembler
# ──────────────────────────────────────────────────────────────────────────────


class ContextAssembler:
    """
    Extracts entities from user messages and assembles context packets.

    Rule-based entity extraction runs first. Graph enrichment queries
    Neo4j for related signals, clusters, and pending insights, then
    assembles everything into a token-efficient context packet.
    """

    def __init__(self, llm_client=None):
        self.llm_client = llm_client

    def extract_entities(self, message: str) -> dict:
        """
        Extract persons, contexts, actions, and temporal references from a message.

        Uses rule-based pattern matching. No graph or LLM required.

        Args:
            message: The user's raw message text.

        Returns:
            dict with keys: persons, contexts, actions, temporal
            Each value is a list of dicts with extraction details.
        """
        if not message or not message.strip():
            return {
                "persons": [],
                "contexts": [],
                "actions": [],
                "temporal": [],
            }

        message_lower = message.lower()

        persons = self._extract_persons(message_lower)
        contexts = self._extract_contexts(message_lower)
        actions = self._extract_actions(message_lower)
        temporal = self._extract_temporal(message_lower)

        return {
            "persons": persons,
            "contexts": contexts,
            "actions": actions,
            "temporal": temporal,
        }

    def enrich_from_graph(self, entities, workspace_id, persona=None,
                          conversation_id=None):
        """
        Query Neo4j for signals, clusters, and insights related to
        the extracted entities.

        Respects the persona's context_depth, confidence_threshold,
        and cluster_surfacing_threshold to control how much context
        the LLM receives.

        When conversation_id is provided, also queries observation bias
        flags from the current conversation's signals for feed-forward.
        Unlike the cross-conversation bias extraction (which requires 2+
        occurrences), intra-conversation biases are surfaced at 1+
        occurrence since they were JUST detected and are immediately
        relevant to the next turn.

        Args:
            entities: Dict from extract_entities().
            workspace_id: The user's workspace_id for scoping queries.
            persona: Optional PersonaConfig controlling depth/thresholds.
            conversation_id: Optional current conversation ID for
                intra-conversation bias feed-forward.

        Returns:
            dict with keys: signals, clusters, insights, bias_flags
        """
        # Persona defaults
        context_depth = 15
        confidence_threshold = 0.5
        cluster_threshold = 5
        include_insights = True

        if persona:
            context_depth = getattr(persona, "context_depth", 15)
            confidence_threshold = getattr(persona, "confidence_threshold", 0.5)
            cluster_threshold = getattr(persona, "cluster_surfacing_threshold", 5)
            include_insights = getattr(persona, "include_pending_insights", True)

        result = {
            "signals": [],
            "clusters": [],
            "insights": [],
            "bias_flags": [],
        }

        try:
            from events_api.neo4j_client import cypher_query
        except ImportError:
            logger.warning("neo4j_client not available; skipping graph enrichment")
            return result

        # ── Query related signals ────────────────────────────────────
        # Find signals that share coordinate values with the current
        # message's entities. Ordered by recency, capped at context_depth.
        try:
            result["signals"] = self._query_related_signals(
                cypher_query, entities, workspace_id,
                limit=context_depth,
                min_confidence=confidence_threshold,
            )
        except Exception as e:
            logger.warning(f"Signal enrichment query failed: {e}")

        # ── Extract observation bias flags from related signals ──────
        try:
            result["bias_flags"] = self._extract_bias_flags(result["signals"])
        except Exception as e:
            logger.warning(f"Bias flag extraction failed: {e}")

        # ── Feed-forward: intra-conversation bias flags ──────────────
        # When a conversation_id is provided, also check the *current*
        # conversation's signals for biases. These are surfaced even if
        # they only appeared once (threshold=1) because they were just
        # detected and are immediately relevant to the next turn.
        if conversation_id:
            try:
                conv_biases = self._query_conversation_bias_flags(
                    cypher_query, workspace_id, conversation_id
                )
                # Merge with cross-conversation biases, preferring higher counts
                existing_types = {b["type"] for b in result["bias_flags"]}
                for cb in conv_biases:
                    if cb["type"] not in existing_types:
                        result["bias_flags"].append(cb)
                    else:
                        # Update count if conversation-local count is higher
                        for existing in result["bias_flags"]:
                            if (existing["type"] == cb["type"]
                                    and cb["count"] > existing["count"]):
                                existing["count"] = cb["count"]
                                existing["example"] = cb.get("example", existing.get("example", ""))
            except Exception as e:
                logger.warning(f"Conversation bias feed-forward failed: {e}")

        # ── Query clusters linked to related signals ─────────────────
        signal_uids = [s.get("uid") for s in result["signals"] if s.get("uid")]
        if signal_uids:
            try:
                result["clusters"] = self._query_related_clusters(
                    cypher_query, workspace_id, signal_uids,
                    min_member_count=cluster_threshold,
                )
            except Exception as e:
                logger.warning(f"Cluster enrichment query failed: {e}")

        # ── Query pending insights ───────────────────────────────────
        if include_insights:
            try:
                result["insights"] = self._query_pending_insights(
                    cypher_query, workspace_id,
                    min_confidence=confidence_threshold,
                )
            except Exception as e:
                logger.warning(f"Insight enrichment query failed: {e}")

        return result

    def _query_related_signals(
        self, cypher_query, entities, workspace_id, limit=15, min_confidence=0.5
    ):
        """
        Find signals that share coordinate values with the current entities.

        Match strategy: find signals connected to the same Person, ContextNode,
        or ActionNode names that appear in the extracted entities. This gives
        the LLM cross-conversation awareness — "you've mentioned your manager
        before, and here's what happened."
        """
        # Gather coordinate names to match against
        person_names = [p["normalized"] for p in entities.get("persons", [])]
        context_names = [c["context"] for c in entities.get("contexts", [])]
        action_names = [a["category"] for a in entities.get("actions", [])]

        all_names = person_names + context_names + action_names
        if not all_names:
            # No entities to match — return recent signals as general context
            return self._query_recent_signals(cypher_query, workspace_id, limit)

        # Build a UNION query: find signals connected to any matching
        # coordinate node, deduplicate, order by recency
        query = """
            // Signals connected to matching Person nodes
            OPTIONAL MATCH (p:Person {workspace_id: $ws})-[:PARTICIPANT_IN]->(s1:Signal {workspace_id: $ws})
            WHERE p.name IN $person_names AND s1.confidence_score >= $min_conf
            WITH collect(DISTINCT s1) AS person_signals

            // Signals connected to matching ContextNode nodes
            OPTIONAL MATCH (s2:Signal {workspace_id: $ws})-[:IN_CONTEXT]->(c:ContextNode)
            WHERE c.name IN $context_names AND s2.confidence_score >= $min_conf
            WITH person_signals, collect(DISTINCT s2) AS context_signals

            // Signals connected to matching ActionNode nodes
            OPTIONAL MATCH (s3:Signal {workspace_id: $ws})-[:INVOLVES_ACTION]->(a:ActionNode)
            WHERE a.name IN $action_names AND s3.confidence_score >= $min_conf
            WITH person_signals + context_signals + collect(DISTINCT s3) AS all_signals

            // Deduplicate and return
            UNWIND all_signals AS s
            WITH DISTINCT s
            WHERE s IS NOT NULL
            RETURN s.uid AS uid,
                   s.signal_address AS address,
                   s.emotions AS emotions,
                   s.confidence_score AS confidence,
                   s.observation_bias_flags AS bias_flags,
                   s.content_preview AS preview,
                   s.created_at AS created_at
            ORDER BY s.created_at DESC
            LIMIT $limit
        """

        rows, cols = cypher_query(query, {
            "ws": workspace_id,
            "person_names": person_names or [],
            "context_names": context_names or [],
            "action_names": action_names or [],
            "min_conf": min_confidence,
            "limit": limit,
        })

        return self._rows_to_signal_dicts(rows, cols)

    def _query_recent_signals(self, cypher_query, workspace_id, limit=10):
        """Fallback: return the most recent signals when no entities match."""
        query = """
            MATCH (s:Signal {workspace_id: $ws})
            WHERE s.confidence_score >= 0.3
            RETURN s.uid AS uid,
                   s.signal_address AS address,
                   s.emotions AS emotions,
                   s.confidence_score AS confidence,
                   s.observation_bias_flags AS bias_flags,
                   s.content_preview AS preview,
                   s.created_at AS created_at
            ORDER BY s.created_at DESC
            LIMIT $limit
        """
        rows, cols = cypher_query(query, {"ws": workspace_id, "limit": limit})
        return self._rows_to_signal_dicts(rows, cols)

    def _rows_to_signal_dicts(self, rows, cols):
        """Convert Cypher result rows to signal dicts."""
        columns = cols if isinstance(cols, list) else []
        signals = []
        for row in rows:
            d = dict(zip(columns, row)) if columns else {}
            # Parse JSON fields
            emotions_raw = d.get("emotions")
            if isinstance(emotions_raw, str):
                try:
                    emotions_raw = json.loads(emotions_raw)
                except (json.JSONDecodeError, TypeError):
                    emotions_raw = []

            bias_raw = d.get("bias_flags")
            if isinstance(bias_raw, str):
                try:
                    bias_raw = json.loads(bias_raw)
                except (json.JSONDecodeError, TypeError):
                    bias_raw = []

            signals.append({
                "uid": d.get("uid"),
                "address": d.get("address", ""),
                "emotions": emotions_raw if isinstance(emotions_raw, list) else [],
                "confidence": d.get("confidence", 0.5),
                "bias_flags": bias_raw if isinstance(bias_raw, list) else [],
                "preview": d.get("preview", ""),
                "created_at": str(d.get("created_at", "")),
            })
        return signals

    def _extract_bias_flags(self, signals):
        """
        Aggregate observation bias flags across related signals.

        Returns a list of dicts: [{type, count, example_preview}]
        for biases that appear more than once (indicating a pattern).
        """
        bias_counts = {}
        bias_examples = {}

        for sig in signals:
            for flag in sig.get("bias_flags", []):
                flag_type = flag if isinstance(flag, str) else str(flag)
                bias_counts[flag_type] = bias_counts.get(flag_type, 0) + 1
                if flag_type not in bias_examples:
                    bias_examples[flag_type] = sig.get("preview", "")[:100]

        # Only surface biases that appear in multiple signals (pattern, not one-off)
        return [
            {
                "type": bias_type,
                "count": count,
                "example": bias_examples.get(bias_type, ""),
            }
            for bias_type, count in bias_counts.items()
            if count >= 2
        ]

    def _query_conversation_bias_flags(
        self, cypher_query, workspace_id, conversation_id
    ):
        """
        Query observation bias flags from the current conversation's signals.

        Unlike _extract_bias_flags (which aggregates across all related signals
        and requires 2+ occurrences), this queries biases from the *current*
        conversation only and surfaces them at threshold=1. This enables
        feed-forward: biases detected on turn N influence the AI on turn N+1.

        Returns:
            List of dicts: [{type, count, example, source: "current_conversation"}]
        """
        query = """
            MATCH (conv:Conversation {conversation_id: $conv_id, workspace_id: $ws})
                  -[:CONTAINS_SIGNAL]->(s:Signal)
            WHERE s.observation_bias_flags IS NOT NULL
                  AND s.observation_bias_flags <> '[]'
            RETURN s.observation_bias_flags AS bias_flags,
                   s.signal_address AS address,
                   substring(coalesce(s.content_preview, ''), 0, 100) AS preview
            ORDER BY s.created_at DESC
        """
        rows, cols = cypher_query(query, {
            "conv_id": conversation_id,
            "ws": workspace_id,
        })

        columns = cols if isinstance(cols, list) else []
        bias_counts = {}
        bias_examples = {}

        for row in rows:
            d = dict(zip(columns, row)) if columns else {}
            raw = d.get("bias_flags", "[]")
            if isinstance(raw, str):
                try:
                    flags = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    flags = []
            else:
                flags = raw if isinstance(raw, list) else []

            for flag in flags:
                flag_type = flag if isinstance(flag, str) else str(flag)
                bias_counts[flag_type] = bias_counts.get(flag_type, 0) + 1
                if flag_type not in bias_examples:
                    bias_examples[flag_type] = d.get("preview", "")

        # Threshold = 1 for intra-conversation (immediate feed-forward)
        return [
            {
                "type": bias_type,
                "count": count,
                "example": bias_examples.get(bias_type, ""),
                "source": "current_conversation",
            }
            for bias_type, count in bias_counts.items()
            if count >= 1
        ]

    def _query_related_clusters(
        self, cypher_query, workspace_id, signal_uids, min_member_count=5
    ):
        """
        Find active clusters that contain any of the related signals.
        """
        if not signal_uids:
            return []

        query = """
            MATCH (s:Signal)-[:MEMBER_OF]->(c:Cluster {workspace_id: $ws})
            WHERE s.uid IN $uids
                  AND c.status = 'active'
                  AND c.member_count >= $min_members
            WITH DISTINCT c
            RETURN c.cluster_id AS cluster_id,
                   c.cluster_type AS cluster_type,
                   c.shared_coordinates AS shared_coordinates,
                   c.divergent_dimensions AS divergent_dimensions,
                   c.strength AS strength,
                   c.member_count AS member_count
            ORDER BY c.strength DESC
            LIMIT 3
        """
        rows, cols = cypher_query(query, {
            "ws": workspace_id,
            "uids": signal_uids,
            "min_members": min_member_count,
        })

        columns = cols if isinstance(cols, list) else []
        clusters = []
        for row in rows:
            d = dict(zip(columns, row)) if columns else {}
            # Parse JSON fields
            shared = d.get("shared_coordinates")
            if isinstance(shared, str):
                try:
                    shared = json.loads(shared)
                except (json.JSONDecodeError, TypeError):
                    shared = {}
            divergent = d.get("divergent_dimensions")
            if isinstance(divergent, str):
                try:
                    divergent = json.loads(divergent)
                except (json.JSONDecodeError, TypeError):
                    divergent = {}

            clusters.append({
                "cluster_id": d.get("cluster_id", ""),
                "cluster_type": d.get("cluster_type", ""),
                "shared_coordinates": shared if isinstance(shared, dict) else {},
                "divergent_dimensions": divergent if isinstance(divergent, dict) else {},
                "strength": d.get("strength", 0.0),
                "member_count": d.get("member_count", 0),
            })
        return clusters

    def _query_pending_insights(self, cypher_query, workspace_id, min_confidence=0.5):
        """
        Find pending insights that haven't been surfaced yet.
        """
        query = """
            MATCH (pi:PendingInsight {workspace_id: $ws})
            WHERE pi.status = 'pending'
                  AND pi.confidence >= $min_conf
            RETURN pi.uid AS uid,
                   pi.detection_type AS detection_type,
                   pi.description AS description,
                   pi.confidence AS confidence
            ORDER BY pi.confidence DESC, pi.created_at DESC
            LIMIT 3
        """
        rows, cols = cypher_query(query, {
            "ws": workspace_id,
            "min_conf": min_confidence,
        })

        columns = cols if isinstance(cols, list) else []
        insights = []
        for row in rows:
            d = dict(zip(columns, row)) if columns else {}
            insights.append({
                "uid": d.get("uid"),
                "detection_type": d.get("detection_type", ""),
                "description": d.get("description", ""),
                "confidence": d.get("confidence", 0.5),
            })
        return insights

    # ── Private extraction helpers ────────────────────────────────────

    def _extract_persons(self, text: str) -> list:
        """Extract person mentions from text."""
        found = []
        for role, keywords in _PERSON_PATTERNS.items():
            for keyword in keywords:
                if keyword in text:
                    found.append({
                        "mention": keyword,
                        "normalized": role,
                        "role": role,
                        "confidence": 0.8,
                    })
                    break  # One match per role category
        return found

    def _extract_contexts(self, text: str) -> list:
        """Extract context mentions from text."""
        found = []
        for context, keywords in _CONTEXT_PATTERNS.items():
            for keyword in keywords:
                if keyword in text:
                    found.append({
                        "mention": keyword,
                        "context": context,
                        "confidence": 0.8,
                    })
                    break
        return found

    def _extract_actions(self, text: str) -> list:
        """Extract action/trigger mentions from text."""
        found = []
        for category, keywords in _ACTION_PATTERNS.items():
            for keyword in keywords:
                if keyword in text:
                    found.append({
                        "mention": keyword,
                        "category": category,
                        "confidence": 0.7,
                    })
                    break
        return found

    def _extract_temporal(self, text: str) -> list:
        """Extract temporal references from text."""
        found = []
        for temporal, keywords in _TEMPORAL_PATTERNS.items():
            for keyword in keywords:
                if keyword in text:
                    found.append({
                        "mention": keyword,
                        "temporal": temporal,
                        "confidence": 0.8,
                    })
                    break
        return found


# ──────────────────────────────────────────────────────────────────────────────
# Context Packet Assembly
# ──────────────────────────────────────────────────────────────────────────────

# Human-readable cluster type descriptions
_CLUSTER_TYPE_LABELS = {
    "same_time_diff_emotion": "mixed emotions in the same moment",
    "same_person_diff_time": "recurring pattern with the same person",
    "same_context_diff_person": "this setting triggers feelings regardless of who's involved",
    "same_action_diff_everything": "this trigger keeps appearing across different situations",
    "same_emotion_diff_source": "the same feeling coming from different sources",
    "cross_dimensional": "a complex pattern across multiple dimensions",
}

# Human-readable bias type descriptions
_BIAS_TYPE_LABELS = {
    "projection": "attributing feelings to others without clear evidence",
    "rumination_amplification": "repeated thinking that may be inflating the emotional intensity",
    "confirmation_bias": "selectively noticing evidence that confirms an existing belief",
    "narrative_construction": "building a causal story that may not fully match events",
}


def assemble_context(
    entities: dict,
    signals: Optional[list] = None,
    clusters: Optional[list] = None,
    insights: Optional[list] = None,
    bias_flags: Optional[list] = None,
    token_budget: int = 2000,
) -> str:
    """
    Assemble a context packet from extracted entities and graph data.

    The output is natural language that reads like a briefing note for
    the LLM — it should be conversational, not clinical.

    Priority order when truncating for token budget:
    1. Entity summary (always included)
    2. Related signals (capped by context_depth)
    3. Observation bias patterns (if recurring)
    4. Active clusters (capped at 3)
    5. Pending insights (if persona allows)

    Args:
        entities: Dict from ContextAssembler.extract_entities().
        signals: Optional list of related signal dicts.
        clusters: Optional list of related cluster dicts.
        insights: Optional list of pending insight dicts.
        bias_flags: Optional list of aggregated bias flag dicts.
        token_budget: Maximum approximate tokens for the packet.

    Returns:
        Assembled context string for LLM prompt injection.
    """
    parts = []

    # ── 1. Entity summary (always included) ──────────────────────
    entity_lines = []
    if entities.get("persons"):
        person_names = [p["normalized"] for p in entities["persons"]]
        entity_lines.append(f"People mentioned: {', '.join(person_names)}")

    if entities.get("contexts"):
        context_names = [c["context"] for c in entities["contexts"]]
        entity_lines.append(f"Contexts: {', '.join(context_names)}")

    if entities.get("actions"):
        action_names = [a["category"] for a in entities["actions"]]
        entity_lines.append(f"Actions detected: {', '.join(action_names)}")

    if entities.get("temporal"):
        time_refs = [t["temporal"] for t in entities["temporal"]]
        entity_lines.append(f"Time references: {', '.join(time_refs)}")

    if entity_lines:
        parts.append("\n".join(entity_lines))

    # ── 2. Related signals from past conversations ───────────────
    if signals:
        sig_lines = [f"\nEmotional history ({len(signals)} related signals found):"]
        for sig in signals[:10]:
            addr = sig.get("address", "unknown")
            emotions = sig.get("emotions", [])
            preview = sig.get("preview", "")[:80]

            # Format emotions concisely
            emo_names = []
            for e in emotions[:3]:
                if isinstance(e, dict):
                    name = e.get("emotion") or e.get("name", "")
                    intensity = e.get("intensity")
                    if name:
                        emo_names.append(f"{name}" + (f" ({intensity}/10)" if intensity else ""))
                elif isinstance(e, str):
                    emo_names.append(e)

            emo_str = ", ".join(emo_names) if emo_names else "unspecified"
            line = f"  - {addr}: {emo_str}"
            if preview:
                line += f' — "{preview}"'
            sig_lines.append(line)

        parts.append("\n".join(sig_lines))

    # ── 3. Observation bias patterns ─────────────────────────────
    if bias_flags:
        # Separate current-conversation biases from historical ones
        conv_biases = [b for b in bias_flags if b.get("source") == "current_conversation"]
        hist_biases = [b for b in bias_flags if b.get("source") != "current_conversation"]

        bias_lines = ["\nObservation patterns to be aware of:"]

        if conv_biases:
            bias_lines.append("  From this conversation:")
            for bf in conv_biases[:3]:
                bias_type = bf.get("type", "")
                count = bf.get("count", 0)
                label = _BIAS_TYPE_LABELS.get(bias_type, bias_type)
                example = bf.get("example", "")
                line = f"    - {label} (detected {count}x in this conversation)"
                if example:
                    line += f' — e.g., "{example[:60]}"'
                bias_lines.append(line)

        if hist_biases:
            bias_lines.append("  From past conversations:")
            for bf in hist_biases[:3]:
                bias_type = bf.get("type", "")
                count = bf.get("count", 0)
                label = _BIAS_TYPE_LABELS.get(bias_type, bias_type)
                bias_lines.append(
                    f"    - {label} (appeared in {count} past signals)"
                )

        bias_lines.append(
            "  IMPORTANT: Do not name these biases directly. Instead, gently "
            "probe the underlying assumptions through questions. For biases "
            "from this conversation, address them sooner — they are fresh."
        )
        parts.append("\n".join(bias_lines))

    # ── 4. Active clusters ───────────────────────────────────────
    if clusters:
        cluster_lines = [f"\nDetected patterns ({len(clusters)} clusters):"]
        for cl in clusters[:3]:
            ctype = cl.get("cluster_type", "")
            label = _CLUSTER_TYPE_LABELS.get(ctype, ctype)
            strength = cl.get("strength", 0.0)
            shared = cl.get("shared_coordinates", {})
            member_count = cl.get("member_count", 0)

            # Format shared coordinates naturally
            shared_desc = []
            for dim, vals in shared.items():
                if isinstance(vals, list):
                    shared_desc.append(f"{dim}: {', '.join(str(v) for v in vals)}")
                else:
                    shared_desc.append(f"{dim}: {vals}")
            shared_str = "; ".join(shared_desc) if shared_desc else "multiple dimensions"

            cluster_lines.append(
                f"  - Pattern: {label} ({shared_str}) — "
                f"strength {strength:.1f}, {member_count} instances"
            )
        parts.append("\n".join(cluster_lines))

    # ── 5. Pending insights ──────────────────────────────────────
    if insights:
        insight_lines = ["\nInsights awaiting exploration:"]
        for ins in insights[:3]:
            desc = ins.get("description", "")
            dtype = ins.get("detection_type", "")
            if desc:
                insight_lines.append(f"  - [{dtype}] {desc}")
        parts.append("\n".join(insight_lines))

    # ── Assemble and truncate ────────────────────────────────────
    full_packet = "\n".join(parts)

    # Rough token estimate (1 token ≈ 4 chars for English)
    approx_tokens = len(full_packet) // 4
    if approx_tokens > token_budget:
        # Truncate from the bottom (insights first, then clusters)
        char_limit = token_budget * 4
        full_packet = full_packet[:char_limit]

    return full_packet
