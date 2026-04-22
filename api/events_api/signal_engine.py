"""
ThriveSight Signal Engine — Real-Time Signal Generation.

Generates emotional signals from user messages with multi-emotion support,
wildcard detection, and observation bias tracking.

This module provides:
- SignalGenerator: main class for signal generation
- LLM-powered signal generation (primary path)
- Keyword-based fallback when LLM is unavailable
- Signal persistence to Neo4j graph
- Embedding computation for semantic similarity
"""

import json
import logging
import re
from typing import Optional

from events_api.coordinate_system import (
    COORDINATE_NAMES,
    WILDCARD,
    build_signal_address,
    detect_wildcards,
    parse_signal_address,
)
from events_api.identity import normalize_graph_person_name

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Emotion Keyword Map (for fallback detection)
# ──────────────────────────────────────────────────────────────────────────────

_EMOTION_KEYWORDS = {
    "anger": ["angry", "anger", "furious", "rage", "mad", "livid", "outraged",
              "infuriated", "enraged", "irate"],
    "frustration": ["frustrated", "frustration", "annoyed", "irritated",
                    "exasperated"],
    "sadness": ["sad", "sadness", "depressed", "down", "unhappy", "miserable",
                "heartbroken", "devastated", "grief"],
    "fear": ["afraid", "fear", "scared", "anxious", "worried", "terrified",
             "nervous", "panic", "dread"],
    "shame": ["ashamed", "shame", "embarrassed", "humiliated", "mortified"],
    "guilt": ["guilty", "guilt", "remorse", "regret"],
    "joy": ["happy", "joy", "joyful", "excited", "thrilled", "elated",
            "ecstatic", "delighted", "pleased"],
    "warmth": ["warm", "warmth", "love", "affection", "tender", "caring",
               "compassion"],
    "surprise": ["surprised", "surprise", "shocked", "astonished", "amazed",
                 "stunned"],
    "disgust": ["disgusted", "disgust", "repulsed", "revolted", "sickened"],
    "contempt": ["contempt", "disdain", "scorn", "disdainful"],
    "hope": ["hopeful", "hope", "optimistic", "encouraged"],
    "relief": ["relieved", "relief"],
    "confusion": ["confused", "confusion", "bewildered", "puzzled", "lost"],
    "loneliness": ["lonely", "loneliness", "isolated", "alone"],
    "resentment": ["resentful", "resentment", "bitter", "bitterness"],
    "pride": ["proud", "pride", "accomplished"],
    "gratitude": ["grateful", "gratitude", "thankful", "appreciative"],
}

# Context keyword map (for fallback coordinate detection)
_CONTEXT_KEYWORDS = {
    "work": ["work", "office", "meeting", "job", "boss", "manager", "team",
             "project", "deadline", "colleague"],
    "home": ["home", "house", "apartment"],
    "social": ["party", "friends", "dinner", "gathering"],
    "family": ["family", "mother", "father", "sibling", "parent"],
    "health": ["doctor", "hospital", "therapy", "exercise"],
    "self": ["myself", "alone", "journal"],
}

# Person keyword map (for fallback participant detection)
_PERSON_KEYWORDS = {
    "manager": ["manager", "boss", "supervisor"],
    "colleague": ["colleague", "coworker", "teammate"],
    "partner": ["partner", "spouse", "husband", "wife"],
    "friend": ["friend"],
    "family_member": ["mother", "father", "sister", "brother", "parent"],
}


# ──────────────────────────────────────────────────────────────────────────────
# SignalGenerator
# ──────────────────────────────────────────────────────────────────────────────


class SignalGenerator:
    """
    Generates emotional signals from user messages.

    When use_llm=True (default), uses LLM prompts for accurate signal
    generation. When use_llm=False, falls back to keyword-based detection
    for testing and degraded-mode operation.
    """

    def __init__(
        self,
        use_llm: bool = True,
        llm_client=None,
        workspace_id: str = "",
        current_username: str = "",
        owner_user_id: int | None = None,
    ):
        self.use_llm = use_llm
        self.llm_client = llm_client
        self._workspace_id = workspace_id or ""
        self._current_username = current_username or ""
        self._owner_user_id = owner_user_id

    def generate_from_message(
        self,
        message: str,
        conversation_context: Optional[str] = None,
        participants: Optional[list] = None,
    ) -> dict:
        """
        Generate signals from a user message.

        Args:
            message: The user's raw message text.
            conversation_context: Optional summary of conversation so far.
            participants: Optional list of known participant names.

        Returns:
            dict with keys: signals (list of signal dicts)
        """
        if self.use_llm and self.llm_client:
            return self._generate_signals_llm(
                message, conversation_context, participants
            )
        return self._generate_signals_fallback(
            message, participants or []
        )

    def _generate_signals_llm(
        self,
        message: str,
        conversation_context: Optional[str],
        participants: Optional[list],
    ) -> dict:
        """
        Generate signals using LLM-powered analysis.

        Pipeline:
        1. Call LLM for multi-emotion signal generation
        2. Assess observation bias and confidence per signal
        3. Compute embedding for semantic similarity
        4. Persist Signal nodes to Neo4j graph with relationships

        Falls back to keyword detection if LLM call fails.
        """
        from events_api.llm_client import (
            generate_signal_from_message,
            assess_signal_confidence,
            compute_text_embedding,
        )

        try:
            # Step 1: LLM signal generation
            result = generate_signal_from_message(
                message=message,
                conversation_context=conversation_context,
                participants=participants,
            )

            signals = result.get("signals", [])
            if not signals:
                logger.warning("LLM returned no signals, falling back to keyword")
                return self._generate_signals_fallback(message, participants or [])

            # Step 2: Confidence assessment and embedding for each signal
            enriched_signals = []
            for sig in signals:
                # Assess observation bias
                try:
                    confidence_result = assess_signal_confidence(
                        signal_data=sig,
                        conversation_context=conversation_context,
                    )
                    sig["confidence"] = confidence_result["confidence"]
                    sig["observation_bias_flags"] = confidence_result["observation_bias_flags"]
                except Exception as e:
                    logger.warning(f"Confidence assessment failed: {e}")
                    # Keep existing confidence from signal generation

                # Compute embedding
                try:
                    embedding = compute_text_embedding(message)
                    sig["embedding"] = embedding
                except Exception as e:
                    logger.warning(f"Embedding computation failed: {e}")
                    sig["embedding"] = []

                # Detect wildcards from signal address
                try:
                    parsed = parse_signal_address(sig.get("signal_address", "SA(*, *, *, *)"))
                    sig["wildcards"] = detect_wildcards(parsed)
                except Exception:
                    sig["wildcards"] = sig.get("wildcards", [])

                enriched_signals.append(sig)

            # Step 3: Persist to graph
            self._persist_signals_to_graph(enriched_signals, message)

            return {"signals": enriched_signals}

        except Exception as e:
            logger.error(f"LLM signal generation failed, falling back to keyword: {e}")
            return self._generate_signals_fallback(message, participants or [])

    def _persist_signals_to_graph(
        self,
        signals: list,
        message: str,
    ) -> None:
        """
        Persist generated signals to the Neo4j graph.

        Creates Signal nodes with full SA coordinates and connects
        them to coordinate nodes via relationships.
        """
        try:
            from neomodel import db

            for sig in signals:
                address = sig.get("signal_address", "SA(*, *, *, *)")
                emotions_json = json.dumps(sig.get("emotions", []))
                embedding_json = json.dumps(sig.get("embedding", []))
                bias_flags_json = json.dumps(sig.get("observation_bias_flags", []))
                confidence = sig.get("confidence", 0.7)
                provenance = sig.get("provenance", "llm_inferred")
                wildcards = sig.get("wildcards", [])

                # Parse address for coordinate values
                try:
                    parsed = parse_signal_address(address)
                except Exception:
                    parsed = {"context": "*", "person": "*", "action": "*", "temporal": "*"}

                # Create Signal node (workspace_id required for graph queries)
                result, _ = db.cypher_query(
                    """
                    CREATE (s:Signal {
                        uid: randomUUID(),
                        workspace_id: $workspace_id,
                        signal_address: $address,
                        emotions: $emotions,
                        confidence_score: $confidence,
                        provenance: $provenance,
                        observation_bias_flags: $bias_flags,
                        embedding: $embedding,
                        is_resolved: $is_resolved,
                        content_preview: $preview,
                        created_at: datetime()
                    })
                    RETURN s.uid AS uid
                    """,
                    {
                        "workspace_id": self._workspace_id,
                        "address": address,
                        "emotions": emotions_json,
                        "confidence": confidence,
                        "provenance": provenance,
                        "bias_flags": bias_flags_json,
                        "embedding": embedding_json,
                        "is_resolved": len(wildcards) == 0,
                        "preview": message[:200],
                    },
                )

                if not result:
                    continue

                signal_uid = result[0][0]

                # Connect to coordinate nodes (each gets a stable uid via ON CREATE)
                ctx = parsed.get("context", "*")
                if ctx != "*":
                    db.cypher_query(
                        """
                        MATCH (s:Signal {uid: $uid})
                        MERGE (c:ContextNode {name: $name, workspace_id: $workspace_id})
                        ON CREATE SET c.uid = randomUUID(),
                                      c.owner_user_id = $owner_user_id,
                                      c.created_at = datetime()
                        ON MATCH SET c.owner_user_id = coalesce(c.owner_user_id, $owner_user_id)
                        MERGE (s)-[:IN_CONTEXT]->(c)
                        """,
                        {
                            "uid": signal_uid,
                            "name": ctx,
                            "workspace_id": self._workspace_id,
                            "owner_user_id": self._owner_user_id,
                        },
                    )

                person = normalize_graph_person_name(
                    parsed.get("person", "*"),
                    current_username=self._current_username,
                )
                if person:
                    db.cypher_query(
                        """
                        MATCH (s:Signal {uid: $uid})
                        MERGE (p:Person {name: $name, workspace_id: $workspace_id})
                        ON CREATE SET p.uid = randomUUID(),
                                      p.owner_user_id = $owner_user_id,
                                      p.role_type = 'specific_person',
                                      p.created_at = datetime()
                        ON MATCH SET p.owner_user_id = coalesce(p.owner_user_id, $owner_user_id),
                                     p.role_type = coalesce(p.role_type, 'specific_person')
                        MERGE (p)-[:PARTICIPANT_IN {role: 'primary_actor'}]->(s)
                        """,
                        {
                            "uid": signal_uid,
                            "name": person,
                            "workspace_id": self._workspace_id,
                            "owner_user_id": self._owner_user_id,
                        },
                    )

                action = parsed.get("action", "*")
                if action != "*":
                    db.cypher_query(
                        """
                        MATCH (s:Signal {uid: $uid})
                        MERGE (a:ActionNode {name: $name, workspace_id: $workspace_id})
                        ON CREATE SET a.uid = randomUUID(),
                                      a.owner_user_id = $owner_user_id,
                                      a.created_at = datetime()
                        ON MATCH SET a.owner_user_id = coalesce(a.owner_user_id, $owner_user_id)
                        MERGE (s)-[:INVOLVES_ACTION]->(a)
                        """,
                        {
                            "uid": signal_uid,
                            "name": action,
                            "workspace_id": self._workspace_id,
                            "owner_user_id": self._owner_user_id,
                        },
                    )

                temporal = parsed.get("temporal", "*")
                if temporal != "*":
                    db.cypher_query(
                        """
                        MATCH (s:Signal {uid: $uid})
                        MERGE (t:TemporalNode {name: $name, workspace_id: $workspace_id})
                        ON CREATE SET t.uid = randomUUID(),
                                      t.owner_user_id = $owner_user_id,
                                      t.created_at = datetime()
                        ON MATCH SET t.owner_user_id = coalesce(t.owner_user_id, $owner_user_id)
                        MERGE (s)-[:AT_TIME]->(t)
                        """,
                        {
                            "uid": signal_uid,
                            "name": temporal,
                            "workspace_id": self._workspace_id,
                            "owner_user_id": self._owner_user_id,
                        },
                    )

                # Materialize Emotion nodes from the emotions JSON array
                raw_emotions = sig.get("emotions", [])
                if isinstance(raw_emotions, str):
                    try:
                        raw_emotions = json.loads(raw_emotions)
                    except (json.JSONDecodeError, TypeError):
                        raw_emotions = []

                for emo in raw_emotions:
                    emo_name = (emo.get("emotion") or "").strip() if isinstance(emo, dict) else str(emo).strip()
                    if not emo_name:
                        continue
                    emo_props = {}
                    if isinstance(emo, dict):
                        if emo.get("intensity") is not None:
                            emo_props["valence"] = emo["intensity"]
                        if emo.get("source_description"):
                            emo_props["description"] = emo["source_description"]
                    db.cypher_query(
                        """
                        MATCH (s:Signal {uid: $uid})
                        MERGE (e:Emotion {name: $name, workspace_id: $workspace_id})
                        ON CREATE SET e.uid = randomUUID(),
                                      e.owner_user_id = $owner_user_id,
                                      e.created_at = datetime()
                        ON MATCH SET e.owner_user_id = coalesce(e.owner_user_id, $owner_user_id)
                        SET e.valence = coalesce($valence, e.valence),
                            e.description = coalesce($description, e.description)
                        MERGE (s)-[:EXPRESSES_EMOTION]->(e)
                        """,
                        {
                            "uid": signal_uid,
                            "name": emo_name,
                            "workspace_id": self._workspace_id,
                            "owner_user_id": self._owner_user_id,
                            "valence": emo_props.get("valence"),
                            "description": emo_props.get("description"),
                        },
                    )

        except Exception as e:
            logger.warning(f"Signal graph persistence failed (non-fatal): {e}")

    def _generate_signals_fallback(
        self,
        message: str,
        participants: list,
    ) -> dict:
        """
        Generate signals using keyword matching (no LLM required).

        This is the degraded-mode path for testing and when LLM is
        unavailable. It produces reasonable but less accurate signals.

        Args:
            message: The user's raw message text.
            participants: List of known participant names.

        Returns:
            dict with keys: signals (list of signal dicts)
        """
        message_lower = message.lower()

        # Detect emotions
        emotions = self._detect_emotions_keyword(message_lower)
        if not emotions:
            emotions = [{"emotion": "neutral", "intensity": 3.0,
                         "source_coordinate": "unknown",
                         "source_description": "No clear emotion detected",
                         "confidence": 0.3}]

        # Detect context
        context = self._detect_context_keyword(message_lower)

        # Detect person
        person = WILDCARD
        if participants:
            person = participants[0]
        else:
            for role, keywords in _PERSON_KEYWORDS.items():
                if any(kw in message_lower for kw in keywords):
                    person = role
                    break

        # Detect action (simplified — use first emotion as proxy)
        action = WILDCARD

        # Temporal is almost always wildcard in fallback
        temporal = WILDCARD

        # Build signal address
        signal_address = build_signal_address(
            context=context,
            person=person,
            action=action,
            temporal=temporal,
        )

        # Detect wildcards
        wildcards = []
        for name, val in [("context", context), ("person", person),
                          ("action", action), ("temporal", temporal)]:
            if val == WILDCARD:
                wildcards.append(name)

        signal = {
            "signal_address": signal_address,
            "emotions": emotions,
            "participants": [
                {"name": p, "role": "mentioned", "confidence": 0.5}
                for p in participants
            ],
            "confidence": 0.5,
            "provenance": "system_detected",
            "observation_bias_flags": [],
            "wildcards": wildcards,
        }

        return {"signals": [signal]}

    def _detect_emotions_keyword(self, text: str) -> list:
        """Detect emotions from text using keyword matching."""
        found = []
        for emotion, keywords in _EMOTION_KEYWORDS.items():
            for keyword in keywords:
                # Word boundary match to avoid partial matches
                pattern = r'\b' + re.escape(keyword) + r'\b'
                if re.search(pattern, text):
                    found.append({
                        "emotion": emotion,
                        "intensity": 5.0,
                        "source_coordinate": "unknown",
                        "source_description": f"Keyword match: '{keyword}'",
                        "confidence": 0.5,
                    })
                    break  # One match per emotion category
        return found

    def _detect_context_keyword(self, text: str) -> str:
        """Detect context from text using keyword matching."""
        for context, keywords in _CONTEXT_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text:
                    return context
        return WILDCARD


def generate_signals(message: str, **kwargs) -> dict:
    """
    Convenience function for signal generation.

    Args:
        message: The user's raw message text.
        **kwargs: Passed to SignalGenerator.generate_from_message().

    Returns:
        dict with keys: signals (list of signal dicts)
    """
    generator = SignalGenerator(use_llm=False)
    return generator.generate_from_message(message, **kwargs)


# Alias for backward compatibility with stub imports
SignalEngine = SignalGenerator
