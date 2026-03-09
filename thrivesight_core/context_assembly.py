"""
ThriveSight Context Assembly Layer — Token-efficient bridge between graph and LLM.

This is the most critical engineering component in the system. It:
1. Extracts entities from the user's message (persons, contexts, actions, temporal)
2. Queries the graph for related signals, clusters, and insights
3. Compresses everything into a token-efficient context packet
4. Injects the packet into the LLM system prompt

The context assembly layer is the reason ThriveSight conversations feel informed
by history — it's what makes the AI "remember" and connect patterns.

Token budget strategy:
- Each persona defines a max_context_tokens budget (default 2000)
- The assembler prioritizes: active clusters > recent signals > pending insights > reasoning traces
- Compression strategies: cluster summaries, trajectory one-liners, citation references
- Only signals above the persona's confidence_threshold are included

Architecture:
    User message → ContextAssembler.assemble()
        → extract_entities(message)
        → query_related_signals(entities)
        → query_active_clusters(entities)
        → query_pending_insights()
        → query_reasoning_history()
        → compress(packet, token_budget)
        → Return context string for LLM injection
"""

import json
import logging
import re
from typing import Any, Optional

from neomodel import db

from .persona_config import PersonaConfig, get_persona, DEFAULT_PERSONA

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────

# Approximate tokens per character (rough estimate for English text)
CHARS_PER_TOKEN = 4

# Entity extraction patterns (rule-based, LLM fallback for ambiguity)
TEMPORAL_PATTERNS = [
    (r"\b(today|yesterday|this morning|this afternoon|this evening|tonight|last night)\b", "relative"),
    (r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", "cyclical"),
    (r"\b(this week|last week|next week|this month|last month)\b", "period"),
    (r"\b(always|never|usually|often|sometimes|rarely|every time)\b", "cyclical"),
    (r"\b(recently|lately|these days|nowadays)\b", "relative"),
    (r"\b(\d{1,2}/\d{1,2}/\d{2,4})\b", "specific"),
    (r"\b(morning|afternoon|evening|night|dawn|dusk)\b", "cyclical"),
]

CONTEXT_INDICATORS = {
    "work": ["work", "office", "meeting", "boss", "manager", "colleague", "project",
             "deadline", "job", "email", "presentation", "performance", "review",
             "promotion", "fired", "hired", "interview", "client", "team"],
    "home": ["home", "house", "apartment", "kitchen", "bedroom", "roommate",
             "chores", "cooking", "rent", "mortgage", "neighbor"],
    "social": ["party", "dinner", "friends", "gathering", "event", "hangout",
               "bar", "restaurant", "concert", "movie", "club"],
    "family": ["family", "mother", "father", "mom", "dad", "sister", "brother",
               "parent", "child", "kid", "son", "daughter", "uncle", "aunt",
               "grandparent", "cousin", "in-law", "holiday"],
    "health": ["doctor", "hospital", "therapy", "therapist", "health", "sick",
               "pain", "medication", "diagnosis", "symptoms", "exercise", "diet",
               "sleep", "insomnia", "headache", "anxiety", "depression"],
    "self": ["myself", "self-esteem", "confidence", "identity", "values",
             "goals", "purpose", "meaning", "growth", "habit"],
}

ROLE_INDICATORS = [
    "manager", "boss", "supervisor", "colleague", "coworker", "co-worker",
    "friend", "partner", "spouse", "wife", "husband", "parent", "mother",
    "father", "mom", "dad", "sibling", "brother", "sister", "therapist",
    "doctor", "teacher", "mentor", "team lead", "teammate", "ex",
    "roommate", "neighbor", "child", "son", "daughter",
]


class ContextAssembler:
    """
    Assembles token-efficient context packets from the knowledge graph.

    The assembler is the bridge between the graph and the LLM. It extracts
    entities from the user's message, queries the graph for relevant history,
    and compresses it into a context string that fits within the persona's
    token budget.

    Usage:
        assembler = ContextAssembler(persona_id="gentle_explorer")
        context = assembler.assemble(
            message="I'm frustrated with my manager again",
            conversation_id="abc123",
        )
        # context is a string ready for injection into the LLM system prompt
    """

    def __init__(
        self,
        persona_id: str = DEFAULT_PERSONA,
        persona_overrides: Optional[dict] = None,
    ):
        self.persona = get_persona(persona_id, persona_overrides)
        self._last_entities = None
        self._last_packet = None

    def assemble(
        self,
        message: str,
        conversation_id: str = "",
        include_system_prompt: bool = True,
    ) -> str:
        """
        Assemble a complete context string for LLM injection.

        This is the main entry point. It:
        1. Extracts entities from the message
        2. Queries the graph for relevant data
        3. Compresses into a token-efficient string
        4. Optionally prepends the persona's system prompt modifier

        Args:
            message: User's current message
            conversation_id: Current conversation ID
            include_system_prompt: Whether to include persona system prompt

        Returns:
            Context string ready for LLM system prompt injection
        """
        # Step 1: Extract entities
        entities = self.extract_entities(message)
        self._last_entities = entities

        # Step 2: Query graph for related data
        packet = self._build_context_packet(entities, conversation_id)
        self._last_packet = packet

        # Step 3: Compress to token budget
        compressed = self.compress(packet)

        # Step 4: Build final context string
        parts = []

        if include_system_prompt and self.persona.system_prompt_modifier:
            parts.append(self.persona.system_prompt_modifier)

        if compressed:
            parts.append("\n--- KNOWLEDGE GRAPH CONTEXT ---\n")
            parts.append(compressed)
            parts.append("\n--- END CONTEXT ---")

        return "\n\n".join(parts) if parts else ""

    def extract_entities(self, message: str) -> dict:
        """
        Extract entity references from a user message.

        Uses rule-based extraction first (names, time words, context cues),
        with LLM fallback available for ambiguity.

        Args:
            message: User's message text

        Returns:
            Dict with keys: persons, contexts, actions, temporal
        """
        text_lower = message.lower()

        # Extract persons
        persons = self._extract_persons(message, text_lower)

        # Extract contexts
        contexts = self._extract_contexts(text_lower)

        # Extract temporal references
        temporal = self._extract_temporal(text_lower)

        # Extract actions (keywords that map to trigger categories)
        actions = self._extract_actions(text_lower)

        return {
            "persons": persons,
            "contexts": contexts,
            "actions": actions,
            "temporal": temporal,
        }

    def compress(self, packet: dict, token_budget: Optional[int] = None) -> str:
        """
        Compress a context packet into a token-efficient string.

        Prioritization order:
        1. Active clusters (most important — show patterns)
        2. Recent related signals (direct evidence)
        3. Pending insights (background detections)
        4. Reasoning traces (previous conversation insights)

        Args:
            packet: Context packet from _build_context_packet()
            token_budget: Override token budget (default from persona)

        Returns:
            Compressed context string
        """
        budget = token_budget or self.persona.max_context_tokens
        max_chars = budget * CHARS_PER_TOKEN
        parts = []
        chars_used = 0

        # 1. Active clusters (highest priority)
        clusters = packet.get("clusters", [])
        if clusters:
            cluster_text = self._compress_clusters(clusters)
            if chars_used + len(cluster_text) < max_chars:
                parts.append(cluster_text)
                chars_used += len(cluster_text)

        # 2. Recent signals
        signals = packet.get("signals", [])
        if signals:
            signal_text = self._compress_signals(signals, max_chars - chars_used)
            if signal_text:
                parts.append(signal_text)
                chars_used += len(signal_text)

        # 3. Pending insights
        if self.persona.include_pending_insights:
            pending = packet.get("pending_insights", [])
            if pending:
                pending_text = self._compress_pending_insights(pending)
                if chars_used + len(pending_text) < max_chars:
                    parts.append(pending_text)
                    chars_used += len(pending_text)

        # 4. Reasoning traces
        if self.persona.include_reasoning_traces:
            traces = packet.get("reasoning_traces", [])
            if traces:
                trace_text = self._compress_reasoning_traces(traces, max_chars - chars_used)
                if trace_text:
                    parts.append(trace_text)
                    chars_used += len(trace_text)

        return "\n".join(parts)

    # ──────────────────────────────────────────────────────────────────────
    # INTERNAL: Entity Extraction
    # ──────────────────────────────────────────────────────────────────────

    def _extract_persons(self, message: str, text_lower: str) -> list[dict]:
        """Extract person references from message."""
        persons = []

        # Check for role indicators
        for role in ROLE_INDICATORS:
            if role in text_lower:
                persons.append({
                    "mention": role,
                    "normalized": role.lower().replace(" ", "_"),
                    "type": "role",
                })

        # Check for capitalized names (simple heuristic)
        # Match capitalized words that aren't at sentence starts
        words = message.split()
        for i, word in enumerate(words):
            clean = word.strip(".,!?;:'\"")
            if (
                clean
                and clean[0].isupper()
                and len(clean) > 1
                and clean.lower() not in {"i", "the", "a", "an", "my", "we", "they", "it"}
                and clean.lower() not in {r for r in ROLE_INDICATORS}
                and i > 0  # Skip first word (sentence start)
            ):
                # Check against known graph persons
                persons.append({
                    "mention": clean,
                    "normalized": clean.lower(),
                    "type": "name",
                })

        # Deduplicate
        seen = set()
        unique = []
        for p in persons:
            key = p["normalized"]
            if key not in seen:
                seen.add(key)
                unique.append(p)

        return unique

    def _extract_contexts(self, text_lower: str) -> list[dict]:
        """Extract context references from message."""
        contexts = []

        for context, indicators in CONTEXT_INDICATORS.items():
            matches = [ind for ind in indicators if ind in text_lower]
            if matches:
                contexts.append({
                    "context": context,
                    "indicators": matches[:3],  # Top 3 matching indicators
                    "confidence": min(len(matches) * 0.3, 1.0),
                })

        # Sort by confidence
        contexts.sort(key=lambda c: c["confidence"], reverse=True)
        return contexts

    def _extract_temporal(self, text_lower: str) -> list[dict]:
        """Extract temporal references from message."""
        temporal = []

        for pattern, temp_type in TEMPORAL_PATTERNS:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                temporal.append({
                    "mention": match,
                    "type": temp_type,
                    "normalized": match.lower().strip(),
                })

        # Deduplicate
        seen = set()
        unique = []
        for t in temporal:
            if t["normalized"] not in seen:
                seen.add(t["normalized"])
                unique.append(t)

        return unique

    def _extract_actions(self, text_lower: str) -> list[dict]:
        """Extract action references from message."""
        # Action keywords mapped to trigger categories
        action_keywords = {
            "dismissed": "dismissal", "ignored": "dismissal", "minimized": "dismissal",
            "blamed": "accusation", "accused": "accusation", "attacked": "accusation",
            "changed the subject": "deflection", "deflected": "deflection",
            "shut down": "withdrawal", "walked away": "withdrawal", "left": "withdrawal",
            "demanded": "demand", "insisted": "demand", "ultimatum": "demand",
            "asked": "questioning", "questioned": "questioning",
            "validated": "validation", "acknowledged": "acknowledgment",
            "apologized": "concession", "admitted": "concession",
            "mocked": "sarcasm", "sarcastic": "sarcasm",
        }

        actions = []
        for keyword, category in action_keywords.items():
            if keyword in text_lower:
                actions.append({
                    "mention": keyword,
                    "category": category,
                })

        return actions

    # ──────────────────────────────────────────────────────────────────────
    # INTERNAL: Graph Queries
    # ──────────────────────────────────────────────────────────────────────

    def _build_context_packet(self, entities: dict, conversation_id: str) -> dict:
        """
        Query the graph and build a structured context packet.
        """
        packet = {
            "signals": [],
            "clusters": [],
            "pending_insights": [],
            "reasoning_traces": [],
        }

        try:
            packet["signals"] = self._query_related_signals(entities)
        except Exception as e:
            logger.warning(f"Signal query failed: {e}")

        try:
            packet["clusters"] = self._query_active_clusters(entities)
        except Exception as e:
            logger.warning(f"Cluster query failed: {e}")

        if self.persona.include_pending_insights:
            try:
                packet["pending_insights"] = self._query_pending_insights()
            except Exception as e:
                logger.warning(f"Pending insights query failed: {e}")

        if self.persona.include_reasoning_traces:
            try:
                packet["reasoning_traces"] = self._query_reasoning_history(
                    conversation_id
                )
            except Exception as e:
                logger.warning(f"Reasoning history query failed: {e}")

        return packet

    def _query_related_signals(self, entities: dict) -> list[dict]:
        """
        Query graph for signals related to extracted entities.

        Returns signals above the persona's confidence threshold,
        limited to context_depth count.
        """
        limit = self.persona.context_depth
        confidence_min = self.persona.confidence_threshold

        # Build entity match conditions
        person_names = [p["normalized"] for p in entities.get("persons", [])]
        context_names = [c["context"] for c in entities.get("contexts", [])]

        signals = []

        # Query by person
        if person_names:
            results, _ = db.cypher_query(
                """
                MATCH (p:Person)-[:PARTICIPANT_IN]->(s:Signal)
                WHERE p.name IN $names
                  AND (s.confidence_score IS NULL OR s.confidence_score >= $min_conf)
                RETURN s.uid, s.signal_address, s.emotion, s.intensity,
                       s.emotions, s.confidence_score, s.observation_bias_flags,
                       s.created_at
                ORDER BY s.created_at DESC
                LIMIT $limit
                """,
                {
                    "names": person_names,
                    "min_conf": confidence_min,
                    "limit": limit,
                },
            )
            for row in results:
                signals.append({
                    "uid": row[0],
                    "signal_address": row[1],
                    "emotion": row[2],
                    "intensity": row[3],
                    "emotions": row[4],
                    "confidence_score": row[5],
                    "bias_flags": row[6],
                    "created_at": row[7],
                    "match_type": "person",
                })

        # Query by context
        if context_names:
            results, _ = db.cypher_query(
                """
                MATCH (s:Signal)-[:IN_CONTEXT]->(ctx:ContextNode)
                WHERE ctx.name IN $names
                  AND (s.confidence_score IS NULL OR s.confidence_score >= $min_conf)
                RETURN s.uid, s.signal_address, s.emotion, s.intensity,
                       s.emotions, s.confidence_score, s.observation_bias_flags,
                       s.created_at
                ORDER BY s.created_at DESC
                LIMIT $limit
                """,
                {
                    "names": context_names,
                    "min_conf": confidence_min,
                    "limit": limit,
                },
            )
            for row in results:
                signals.append({
                    "uid": row[0],
                    "signal_address": row[1],
                    "emotion": row[2],
                    "intensity": row[3],
                    "emotions": row[4],
                    "confidence_score": row[5],
                    "bias_flags": row[6],
                    "created_at": row[7],
                    "match_type": "context",
                })

        # Deduplicate by uid
        seen = set()
        unique = []
        for s in signals:
            if s["uid"] not in seen:
                seen.add(s["uid"])
                unique.append(s)

        return unique[:limit]

    def _query_active_clusters(self, entities: dict) -> list[dict]:
        """Query for active clusters involving the extracted entities."""
        min_members = self.persona.cluster_surfacing_threshold
        min_strength = self.persona.cluster_strength_threshold

        person_names = [p["normalized"] for p in entities.get("persons", [])]
        context_names = [c["context"] for c in entities.get("contexts", [])]

        clusters = []

        if person_names or context_names:
            # Find clusters containing signals that match our entities
            results, _ = db.cypher_query(
                """
                MATCH (s:Signal)-[r:MEMBER_OF]->(c:Cluster)
                WHERE c.status IN ['active', 'weakening']
                  AND r.active = true
                  AND c.member_count >= $min_members
                  AND (c.strength IS NULL OR c.strength >= $min_strength)
                WITH c, collect(DISTINCT s.uid) AS member_uids
                OPTIONAL MATCH (p:Person)-[:PARTICIPANT_IN]->(s2:Signal)-[:MEMBER_OF]->(c)
                WHERE p.name IN $person_names
                WITH c, member_uids, collect(DISTINCT p.name) AS matching_persons
                RETURN c.cluster_id, c.cluster_type, c.shared_coordinates,
                       c.divergent_dimensions, c.strength, c.confidence_score,
                       c.member_count, c.trajectory_history, matching_persons
                LIMIT 10
                """,
                {
                    "min_members": min_members,
                    "min_strength": min_strength,
                    "person_names": person_names,
                },
            )

            for row in results:
                clusters.append({
                    "cluster_id": row[0],
                    "cluster_type": row[1],
                    "shared_coordinates": row[2],
                    "divergent_dimensions": row[3],
                    "strength": row[4],
                    "confidence_score": row[5],
                    "member_count": row[6],
                    "trajectory_history": row[7],
                    "matching_persons": row[8],
                })

        return clusters

    def _query_pending_insights(self) -> list[dict]:
        """Query for unsurfaced background detections."""
        results, _ = db.cypher_query(
            """
            MATCH (pi:PendingInsight)
            WHERE pi.status = 'pending'
            RETURN pi.uid, pi.detection_type, pi.description,
                   pi.confidence, pi.created_at
            ORDER BY pi.confidence DESC, pi.created_at DESC
            LIMIT 5
            """,
        )

        return [
            {
                "uid": row[0],
                "detection_type": row[1],
                "description": row[2],
                "confidence": row[3],
                "created_at": row[4],
            }
            for row in results
        ]

    def _query_reasoning_history(self, conversation_id: str) -> list[dict]:
        """Query for recent Insight and Reflection nodes from this conversation."""
        if not conversation_id:
            return []

        results, _ = db.cypher_query(
            """
            MATCH (c:Conversation {conversation_id: $conv_id})<-[:GENERATED_DURING]-(i:Insight)
            OPTIONAL MATCH (r:Reflection)-[:RESPONDS_TO]->(i)
            RETURN i.uid, i.reasoning_text, i.persona, i.confidence,
                   i.validation_status, r.text, r.reflection_type
            ORDER BY i.generated_at DESC
            LIMIT 10
            """,
            {"conv_id": conversation_id},
        )

        traces = []
        for row in results:
            trace = {
                "insight_uid": row[0],
                "reasoning_text": row[1],
                "persona": row[2],
                "confidence": row[3],
                "validation_status": row[4],
            }
            if row[5]:  # Has reflection
                trace["user_reflection"] = row[5]
                trace["reflection_type"] = row[6]
            traces.append(trace)

        return traces

    # ──────────────────────────────────────────────────────────────────────
    # INTERNAL: Compression
    # ──────────────────────────────────────────────────────────────────────

    def _compress_clusters(self, clusters: list[dict]) -> str:
        """Compress cluster data into token-efficient summaries."""
        if not clusters:
            return ""

        lines = ["ACTIVE CLUSTERS:"]
        for c in clusters:
            cluster_id = c.get("cluster_id", "unknown")
            cluster_type = c.get("cluster_type", "general")
            member_count = c.get("member_count", 0)
            strength = c.get("strength", 0)
            shared = c.get("shared_coordinates", [])
            divergent = c.get("divergent_dimensions", [])

            # One-liner summary
            shared_str = ", ".join(shared) if shared else "none"
            divergent_str = ", ".join(divergent) if divergent else "none"

            line = (
                f"- [{cluster_id}] {cluster_type}: {member_count} signals, "
                f"strength={strength:.1f}, shared=[{shared_str}], "
                f"divergent=[{divergent_str}]"
            )
            lines.append(line)

            # Add trajectory one-liner if available
            trajectory = c.get("trajectory_history", [])
            if trajectory and len(trajectory) >= 2:
                first_strength = trajectory[0].get("strength", 0)
                last_strength = trajectory[-1].get("strength", 0)
                direction = "strengthening" if last_strength > first_strength else "weakening"
                lines.append(
                    f"  Trajectory: {first_strength:.1f} → {last_strength:.1f} ({direction})"
                )

        return "\n".join(lines)

    def _compress_signals(self, signals: list[dict], max_chars: int) -> str:
        """Compress signal data into token-efficient format."""
        if not signals or max_chars <= 0:
            return ""

        lines = ["RELATED SIGNALS:"]
        chars = len(lines[0])

        for s in signals:
            sa = s.get("signal_address", "unknown")
            emotion = s.get("emotion", "unknown")
            intensity = s.get("intensity", 0)
            confidence = s.get("confidence_score", 0)
            match_type = s.get("match_type", "")

            line = f"- {sa} | {emotion} (i={intensity:.1f}, c={confidence:.1f}) [{match_type}]"

            if chars + len(line) + 1 > max_chars:
                break

            lines.append(line)
            chars += len(line) + 1

        return "\n".join(lines) if len(lines) > 1 else ""

    def _compress_pending_insights(self, insights: list[dict]) -> str:
        """Compress pending insights into brief notes."""
        if not insights:
            return ""

        lines = ["BACKGROUND DETECTIONS:"]
        for ins in insights:
            detection_type = ins.get("detection_type", "unknown")
            description = ins.get("description", "")[:100]
            confidence = ins.get("confidence", 0)
            lines.append(f"- [{detection_type}] {description} (confidence={confidence:.1f})")

        return "\n".join(lines)

    def _compress_reasoning_traces(self, traces: list[dict], max_chars: int) -> str:
        """Compress reasoning history into brief summaries."""
        if not traces or max_chars <= 0:
            return ""

        lines = ["PREVIOUS REASONING:"]
        chars = len(lines[0])

        for t in traces:
            reasoning = (t.get("reasoning_text") or "")[:150]
            status = t.get("validation_status", "pending")

            line = f"- [{status}] {reasoning}"
            if t.get("user_reflection"):
                reflection = t["user_reflection"][:80]
                line += f" → User: \"{reflection}\""

            if chars + len(line) + 1 > max_chars:
                break

            lines.append(line)
            chars += len(line) + 1

        return "\n".join(lines) if len(lines) > 1 else ""


# ──────────────────────────────────────────────────────────────────────────────
# CONVENIENCE FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────


def assemble_context(
    message: str,
    conversation_id: str = "",
    persona_id: str = DEFAULT_PERSONA,
    persona_overrides: Optional[dict] = None,
) -> str:
    """
    Convenience function for assembling context.

    Args:
        message: User's message
        conversation_id: Current conversation ID
        persona_id: Persona to use
        persona_overrides: Optional setting overrides

    Returns:
        Context string ready for LLM system prompt injection
    """
    assembler = ContextAssembler(
        persona_id=persona_id,
        persona_overrides=persona_overrides,
    )
    return assembler.assemble(message, conversation_id)


def extract_entities(message: str) -> dict:
    """Convenience function for entity extraction only."""
    assembler = ContextAssembler()
    return assembler.extract_entities(message)
