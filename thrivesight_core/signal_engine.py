"""
ThriveSight Signal Engine — Real-time signal generation from conversation messages.

The signal engine is the V3 replacement for DialogueSignalGenerator. It generates
Signal Address objects on every conversation message with:
- Multi-emotion support (emotions traced to specific coordinates/participants)
- Wildcard detection (incomplete coordinates flagged for exploration)
- Observation bias tracking (confidence scoring, projection detection)
- Signal derivation (child signals from wildcard resolution)
- Participant detection (extracting person mentions, assigning roles)
- Lightweight embedding computation for semantic similarity

Architecture:
    Message → SignalGenerator.generate_from_message()
        → LLM call (SIGNAL_GENERATION prompt)
        → Signal node(s) created in graph
        → Coordinates resolved to hierarchy nodes
        → Participants linked via PARTICIPANT_IN
        → Wildcards flagged for exploration
        → Embedding computed for cluster matching

    The signal engine does NOT handle clusters — that's the ClusterEngine's job.
    The signal engine does NOT assemble context — that's the ContextAssembler's job.
    It generates signals and writes them to the graph.
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime
from typing import Any, Optional

from neomodel import db

from . import llm_client
from .coordinate_system import (
    WILDCARD,
    build_signal_address,
    detect_wildcards,
    parse_signal_address,
    resolve_coordinates,
)
from .graph_models import (
    ActionNode,
    Cluster,
    ContextNode,
    Conversation,
    Insight,
    Person,
    Signal,
    TemporalNode,
    TriggerAction,
    TriggerCategory,
    Turn,
)
from .llm_prompts import get_prompt

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# KNOWN EMOTIONS (extended V3 vocabulary)
# ──────────────────────────────────────────────────────────────────────────────

KNOWN_EMOTIONS = {
    "frustration", "defensiveness", "anger", "sadness", "anxiety", "hurt",
    "contempt", "warmth", "humor", "resignation", "guilt", "relief",
    "hope", "resentment", "vulnerability", "confusion", "empathy",
    "indifference", "shame", "pride", "grief", "fear", "joy", "surprise",
    "disgust", "embarrassment", "loneliness", "betrayal", "gratitude",
    "admiration", "jealousy", "envy",
}

# Participant roles
PARTICIPANT_ROLES = {
    "primary_actor", "amplifier", "witness", "subject", "mentioned",
}

# Observation bias types
BIAS_TYPES = {
    "projection", "rumination_amplification", "confirmation_bias",
    "narrative_construction",
}

# Exploration geometry types
GEOMETRY_TYPES = {
    "circle", "spiral", "starburst", "line", None,
}


class SignalGenerator:
    """
    Generates Signal nodes from conversation messages.

    This is the core V3 signal generation engine. It processes each user
    message and produces one or more Signal nodes with:
    - Multi-emotion payloads
    - Coordinate resolution to graph nodes
    - Participant linking
    - Wildcard detection
    - Observation bias assessment
    - Lightweight embeddings

    Usage:
        generator = SignalGenerator()
        signals = generator.generate_from_message(
            message="My manager dismissed my work again today",
            conversation_id="abc123",
            participants=["manager"],
        )
    """

    def __init__(self, use_llm: bool = True):
        """
        Args:
            use_llm: If True, use LLM for signal generation. If False, use
                    keyword-based fallback only.
        """
        self.use_llm = use_llm
        self._last_llm_response = None

    def generate_from_message(
        self,
        message: str,
        conversation_id: str,
        conversation_context: str = "",
        participants: Optional[list[str]] = None,
        turn_number: int = 0,
    ) -> list[Signal]:
        """
        Generate Signal nodes from a user message.

        This is the main entry point. It:
        1. Calls the LLM to extract signal data from the message
        2. Creates Signal nodes in the graph
        3. Resolves coordinates to hierarchy nodes
        4. Links participants
        5. Returns the created Signal nodes

        Args:
            message: The user's message text
            conversation_id: ID of the current conversation
            conversation_context: Previous messages for context (formatted text)
            participants: Optional list of known participant names
            turn_number: Current turn number in conversation

        Returns:
            List of created Signal nodes
        """
        if not message or not message.strip():
            return []

        # Step 1: Extract signal data
        try:
            if self.use_llm:
                signal_data = self._generate_signals_llm(
                    message, conversation_context, participants or []
                )
            else:
                signal_data = self._generate_signals_fallback(
                    message, participants or []
                )
        except Exception as e:
            logger.warning(
                f"Signal generation failed ({type(e).__name__}: {e}); "
                f"using keyword fallback"
            )
            signal_data = self._generate_signals_fallback(
                message, participants or []
            )

        # Step 2: Create Signal nodes from extracted data
        created_signals = []
        for raw_signal in signal_data.get("signals", []):
            try:
                signal_node = self._create_signal_node(
                    raw_signal, conversation_id, turn_number
                )
                if signal_node:
                    created_signals.append(signal_node)
            except Exception as e:
                logger.error(f"Failed to create signal node: {e}")
                continue

        logger.info(
            f"Generated {len(created_signals)} signals from message "
            f"(conversation: {conversation_id}, turn: {turn_number})"
        )

        return created_signals

    def detect_signal_wildcards(self, signal: Signal) -> list[str]:
        """
        Identify which coordinates in a Signal are wildcards.

        Args:
            signal: A Signal node

        Returns:
            List of coordinate names that are wildcards
        """
        if signal.wildcard_coordinates:
            return signal.wildcard_coordinates

        # Parse the signal address and check
        if signal.signal_address:
            try:
                parsed = parse_signal_address(signal.signal_address)
                return detect_wildcards(parsed)
            except ValueError:
                pass

        return []

    def derive_child_signal(
        self,
        parent_signal: Signal,
        new_coordinate_info: dict,
    ) -> Optional[Signal]:
        """
        Create a child signal from a parent, filling in wildcard coordinates.

        This is called when a user's response resolves a wildcard in an
        existing signal, producing a more specific child signal.

        Args:
            parent_signal: The parent Signal node
            new_coordinate_info: Dict with coordinate names and their resolved values
                e.g., {"person": "Sarah", "temporal": "last Monday"}

        Returns:
            The created child Signal node, or None on failure
        """
        if not parent_signal.signal_address:
            logger.warning("Cannot derive child signal: parent has no signal_address")
            return None

        try:
            parsed = parse_signal_address(parent_signal.signal_address)
        except ValueError:
            logger.warning(f"Cannot parse parent signal address: {parent_signal.signal_address}")
            return None

        # Fill in wildcards with new values
        for coord_name, value in new_coordinate_info.items():
            if coord_name in parsed and parsed[coord_name] == WILDCARD:
                parsed[coord_name] = value

        # Build new signal address
        new_sa = build_signal_address(
            context=parsed["context"],
            person=parsed["person"],
            action=parsed["action"],
            temporal=parsed["temporal"],
            emotion=parsed.get("emotion"),
        )

        # Create child signal node
        child = Signal(
            signal_address=new_sa,
            emotions=parent_signal.emotions,  # Inherit emotions initially
            confidence_score=parent_signal.confidence_score,
            provenance="derived",
            observation_bias_flags=parent_signal.observation_bias_flags or [],
            wildcard_coordinates=detect_wildcards(parsed) if detect_wildcards(parsed) else [],
            is_resolved=len(detect_wildcards(parsed)) == 0,
        )
        child.save()

        # Link to parent via DERIVED_FROM
        child.derived_from.connect(parent_signal, {
            "derivation_type": "wildcard_resolution",
        })

        # Resolve and link new coordinates
        self._link_coordinates(child, parsed)

        logger.info(
            f"Derived child signal {child.uid} from parent {parent_signal.uid}: "
            f"{parent_signal.signal_address} → {new_sa}"
        )

        return child

    def assess_confidence(
        self,
        signal: Signal,
        conversation_history: str = "",
        user_confidence: Optional[float] = None,
    ) -> dict:
        """
        Assess confidence and check for observation bias on a signal.

        This uses the CONFIDENCE_ASSESSMENT prompt to evaluate whether
        the signal's emotions are reliably reported or potentially biased.

        Args:
            signal: The Signal node to assess
            conversation_history: Previous conversation text
            user_confidence: User's self-reported confidence (if any)

        Returns:
            Dict with assessed_confidence, bias_flags, accountability_note
        """
        if not self.use_llm:
            return {
                "assessed_confidence": signal.confidence_score or 0.5,
                "bias_flags": [],
                "accountability_note": None,
                "confidence_reasoning": "LLM assessment not available",
            }

        try:
            prompt = get_prompt(
                "confidence_assessment",
                signal_address=signal.signal_address or "unknown",
                emotions=json.dumps(signal.emotions or []),
                history=conversation_history[:2000],  # Truncate for token budget
                user_confidence=str(user_confidence or "not stated"),
            )

            result = llm_client.analyze_with_retry(prompt, "Assess this signal.")

            # Update signal with assessment
            if "assessed_confidence" in result:
                signal.confidence_score = result["assessed_confidence"]

            bias_flags = []
            for flag in result.get("bias_flags", []):
                if flag.get("type") in BIAS_TYPES:
                    bias_flags.append(flag["type"])

            if bias_flags:
                signal.observation_bias_flags = bias_flags

            signal.save()

            return result

        except Exception as e:
            logger.warning(f"Confidence assessment failed: {e}")
            return {
                "assessed_confidence": signal.confidence_score or 0.5,
                "bias_flags": [],
                "accountability_note": None,
                "confidence_reasoning": f"Assessment failed: {e}",
            }

    def compute_embedding(self, signal: Signal) -> Optional[list[float]]:
        """
        Compute a lightweight semantic embedding for a signal.

        The embedding is used by the cluster engine for semantic similarity
        matching when coordinate overlap alone isn't sufficient.

        For now, we use a simple hash-based fingerprint. This will be
        replaced with a proper embedding model (e.g., text-embedding-3-small)
        in a future iteration.

        Args:
            signal: The Signal node

        Returns:
            List of floats representing the embedding, or None
        """
        # Build a text representation of the signal
        components = []

        if signal.signal_address:
            components.append(signal.signal_address)

        if signal.emotions:
            for e in signal.emotions:
                if isinstance(e, dict):
                    components.append(f"{e.get('emotion', '')}:{e.get('intensity', 0)}")
                elif isinstance(e, str):
                    components.append(e)

        text = " ".join(components)
        if not text:
            return None

        # Simple hash-based fingerprint (placeholder for real embeddings)
        # Each character of the SHA-256 hash maps to a float in [0, 1]
        hash_hex = hashlib.sha256(text.encode()).hexdigest()
        embedding = [int(hash_hex[i:i+2], 16) / 255.0 for i in range(0, 64, 2)]

        signal.embedding = embedding
        signal.save()

        return embedding

    # ──────────────────────────────────────────────────────────────────────
    # INTERNAL: LLM Signal Generation
    # ──────────────────────────────────────────────────────────────────────

    def _generate_signals_llm(
        self,
        message: str,
        conversation_context: str,
        participants: list[str],
    ) -> dict:
        """
        Call LLM to generate signal data from a message.

        Returns:
            Dict matching the SIGNAL_GENERATION output schema
        """
        system_prompt = get_prompt("signal_generation")

        user_prompt = self._build_signal_user_prompt(
            message, conversation_context, participants
        )

        result = llm_client.analyze_with_retry(system_prompt, user_prompt)
        self._last_llm_response = result

        # Validate structure
        if not isinstance(result, dict):
            raise ValueError(f"Expected dict from LLM, got {type(result)}")

        if "signals" not in result:
            # Try wrapping single signal
            if "signal_address" in result:
                result = {"signals": [result], "entity_mentions": {}}
            else:
                raise ValueError("LLM response missing 'signals' key")

        return result

    def _build_signal_user_prompt(
        self,
        message: str,
        conversation_context: str,
        participants: list[str],
    ) -> str:
        """Build the user prompt for signal generation."""
        parts = []

        if conversation_context:
            parts.append(f"Previous conversation context:\n{conversation_context[:3000]}\n")

        if participants:
            parts.append(f"Known participants: {', '.join(participants)}\n")

        parts.append(f"Current message to analyze:\n{message}")

        parts.append(
            "\nGenerate signal address objects for the emotional moments in this message. "
            "Identify ALL emotions (not just the primary one), trace each to its source, "
            "and flag any observation bias you detect."
        )

        return "\n".join(parts)

    # ──────────────────────────────────────────────────────────────────────
    # INTERNAL: Keyword Fallback
    # ──────────────────────────────────────────────────────────────────────

    # Emotion keyword lexicon (from V2 DialogueSignalGenerator)
    EMOTION_KEYWORDS = {
        "frustration": ["frustrated", "annoying", "sick of", "tired of", "can't believe"],
        "anger": ["angry", "furious", "mad", "pissed", "rage", "how dare"],
        "defensiveness": ["not my fault", "i didn't", "don't blame", "that's not fair"],
        "sadness": ["sad", "hurt", "miss", "cry", "lonely", "depressed"],
        "anxiety": ["worried", "scared", "nervous", "anxious", "afraid", "what if"],
        "contempt": ["pathetic", "ridiculous", "whatever", "you always", "you never"],
        "warmth": ["love", "appreciate", "thank", "grateful", "proud of you"],
        "hope": ["maybe we could", "what if we", "we could try", "good idea"],
        "guilt": ["sorry", "my fault", "i shouldn't have", "i apologize"],
        "vulnerability": ["i feel", "it makes me", "i need", "i'm afraid"],
        "resignation": ["fine", "whatever", "i give up", "doesn't matter", "forget it"],
        "shame": ["ashamed", "embarrassed", "humiliated", "mortified"],
        "pride": ["proud", "accomplished", "achieved", "nailed it"],
        "grief": ["loss", "gone", "miss them", "passed away"],
        "fear": ["terrified", "dread", "panic", "frightened"],
        "joy": ["happy", "excited", "thrilled", "overjoyed", "delighted"],
        "betrayal": ["betrayed", "stabbed in the back", "trusted", "lied to"],
        "loneliness": ["alone", "isolated", "no one", "nobody"],
        "gratitude": ["grateful", "thankful", "blessed", "appreciate"],
    }

    # Context keyword lexicon
    CONTEXT_KEYWORDS = {
        "work": ["work", "office", "meeting", "boss", "manager", "colleague", "project", "deadline", "job"],
        "home": ["home", "house", "apartment", "kitchen", "bedroom", "roommate"],
        "social": ["party", "dinner", "friends", "gathering", "event", "hangout"],
        "family": ["family", "mother", "father", "sister", "brother", "parent", "child", "kid"],
        "health": ["doctor", "hospital", "therapy", "therapist", "health", "sick", "pain"],
        "self": ["myself", "i feel", "i think", "i need", "i want", "i am"],
    }

    # Person role keywords
    ROLE_KEYWORDS = {
        "manager": "primary_actor",
        "boss": "primary_actor",
        "colleague": "amplifier",
        "friend": "amplifier",
        "partner": "primary_actor",
        "spouse": "primary_actor",
        "parent": "primary_actor",
        "therapist": "witness",
    }

    def _generate_signals_fallback(
        self,
        message: str,
        participants: list[str],
    ) -> dict:
        """
        Generate signals using keyword-based fallback when LLM is unavailable.

        Returns:
            Dict matching the SIGNAL_GENERATION output schema (simplified)
        """
        text_lower = message.lower()

        # Detect emotions
        emotions = []
        for emotion, keywords in self.EMOTION_KEYWORDS.items():
            matches = sum(1 for kw in keywords if kw in text_lower)
            if matches > 0:
                intensity = min(1.0 + matches * 0.8, 5.0)
                emotions.append({
                    "emotion": emotion,
                    "intensity": round(intensity, 1),
                    "source_coordinate": "action",
                    "source_description": f"Keyword match ({matches} hits)",
                    "confidence": 0.4,
                })

        if not emotions:
            emotions = [{
                "emotion": "neutral",
                "intensity": 1.0,
                "source_coordinate": "unknown",
                "source_description": "No keyword matches",
                "confidence": 0.2,
            }]

        # Detect context
        context = WILDCARD
        for ctx, keywords in self.CONTEXT_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                context = ctx
                break

        # Detect participants
        detected_participants = []
        for participant in participants:
            if participant.lower() in text_lower:
                role = self.ROLE_KEYWORDS.get(participant.lower(), "mentioned")
                detected_participants.append({
                    "name": participant,
                    "role": role,
                })

        # Build signal
        sa = build_signal_address(
            context=context,
            person=participants[0] if participants else WILDCARD,
            action=WILDCARD,
            temporal="recent",
        )

        signal = {
            "signal_address": sa,
            "emotions": emotions,
            "participants": detected_participants,
            "wildcards": [],
            "provenance": "llm_inferred",
            "confidence_score": 0.3,
            "observation_bias_flags": [],
            "exploration_geometry": None,
        }

        # Detect wildcards
        try:
            parsed = parse_signal_address(sa)
            signal["wildcards"] = detect_wildcards(parsed)
        except ValueError:
            pass

        return {
            "signals": [signal],
            "entity_mentions": {
                "persons": participants,
                "contexts": [context] if context != WILDCARD else [],
                "actions": [],
                "temporal": ["recent"],
            },
        }

    # ──────────────────────────────────────────────────────────────────────
    # INTERNAL: Graph Node Creation
    # ──────────────────────────────────────────────────────────────────────

    def _create_signal_node(
        self,
        raw_signal: dict,
        conversation_id: str,
        turn_number: int,
    ) -> Optional[Signal]:
        """
        Create a Signal node in the graph from raw LLM output.

        Args:
            raw_signal: Dict from LLM matching signal schema
            conversation_id: Conversation this signal belongs to
            turn_number: Turn number within conversation

        Returns:
            Created Signal node, or None on failure
        """
        sa_string = raw_signal.get("signal_address", "")
        emotions = raw_signal.get("emotions", [])
        participants = raw_signal.get("participants", [])
        wildcards = raw_signal.get("wildcards", [])
        provenance = raw_signal.get("provenance", "llm_inferred")
        confidence = raw_signal.get("confidence_score", 0.5)
        bias_flags = raw_signal.get("observation_bias_flags", [])
        geometry = raw_signal.get("exploration_geometry")

        # Validate emotions
        validated_emotions = self._validate_emotions(emotions)

        # Determine primary emotion for backward-compatible 'emotion' field
        primary_emotion = "neutral"
        primary_intensity = 0.0
        if validated_emotions:
            # Highest intensity emotion is primary
            best = max(validated_emotions, key=lambda e: e.get("intensity", 0))
            primary_emotion = best.get("emotion", "neutral")
            primary_intensity = best.get("intensity", 1.0)

        # Create Signal node
        signal = Signal(
            signal_address=sa_string,
            emotion=primary_emotion,
            intensity=primary_intensity,
            emotions=validated_emotions,
            confidence_score=min(max(confidence, 0.0), 1.0),
            provenance=provenance if provenance in ("user_stated", "llm_inferred", "derived") else "llm_inferred",
            observation_bias_flags=[f for f in bias_flags if f in BIAS_TYPES] if bias_flags else [],
            wildcard_coordinates=wildcards,
            is_resolved=len(wildcards) == 0,
            exploration_geometry=geometry if geometry in GEOMETRY_TYPES else None,
        )
        signal.save()

        # Resolve and link coordinates
        try:
            parsed = parse_signal_address(sa_string)
            self._link_coordinates(signal, parsed)
        except ValueError:
            logger.warning(f"Could not parse signal address: {sa_string}")

        # Link participants
        self._link_participants(signal, participants)

        # Link to conversation
        self._link_to_conversation(signal, conversation_id)

        # Compute embedding
        try:
            self.compute_embedding(signal)
        except Exception as e:
            logger.debug(f"Embedding computation skipped: {e}")

        return signal

    def _validate_emotions(self, emotions: list) -> list[dict]:
        """Validate and normalize emotion data from LLM output."""
        validated = []
        for e in emotions:
            if isinstance(e, str):
                # Simple string emotion → convert to dict
                validated.append({
                    "emotion": e if e in KNOWN_EMOTIONS else "unclassified",
                    "intensity": 2.0,
                    "source_coordinate": "unknown",
                    "source_description": "",
                    "confidence": 0.5,
                })
            elif isinstance(e, dict):
                emotion_name = e.get("emotion", "unclassified")
                validated.append({
                    "emotion": emotion_name if emotion_name in KNOWN_EMOTIONS else "unclassified",
                    "intensity": min(max(float(e.get("intensity", 2.0)), 1.0), 5.0),
                    "source_coordinate": e.get("source_coordinate", "unknown"),
                    "source_description": e.get("source_description", ""),
                    "confidence": min(max(float(e.get("confidence", 0.5)), 0.0), 1.0),
                })
        return validated

    def _link_coordinates(self, signal: Signal, parsed_sa: dict):
        """Resolve parsed SA coordinates to graph nodes and link to signal."""
        resolved = resolve_coordinates(parsed_sa)

        if resolved.get("context"):
            signal.in_context.connect(resolved["context"])

        if resolved.get("action"):
            signal.involves_action.connect(resolved["action"])

        if resolved.get("temporal"):
            signal.at_time.connect(resolved["temporal"])

        # Person is handled via PARTICIPANT_IN, not direct link
        # (Person resolution happens in _link_participants)

    def _link_participants(self, signal: Signal, participants: list[dict]):
        """Create or find Person nodes and link to signal with roles."""
        for p in participants:
            name = p.get("name", "").strip()
            role = p.get("role", "mentioned")

            if not name:
                continue

            # Validate role
            if role not in PARTICIPANT_ROLES:
                role = "mentioned"

            # Find or create person node
            normalized = name.lower()
            existing = Person.nodes.filter(name=normalized)
            if existing:
                person_node = existing[0]
            else:
                # Detect if this is a role reference
                role_keywords = [
                    "manager", "boss", "supervisor", "colleague", "coworker",
                    "friend", "partner", "spouse", "parent", "mother", "father",
                    "sibling", "therapist", "doctor", "teacher", "mentor",
                ]
                is_role = any(kw in normalized for kw in role_keywords)

                person_node = Person(
                    name=normalized,
                    role_type="role_category" if is_role else "specific_person",
                    role=normalized if is_role else None,
                )
                person_node.save()

            # Connect person to signal with role
            person_node.signals.connect(signal, {"role": role})

    def _link_to_conversation(self, signal: Signal, conversation_id: str):
        """Link signal to its conversation via Cypher (conversation may not have neomodel node)."""
        try:
            db.cypher_query(
                """
                MATCH (c:Conversation {conversation_id: $conv_id})
                MATCH (s:Signal {uid: $signal_uid})
                MERGE (c)-[:CONTAINS_SIGNAL]->(s)
                """,
                {
                    "conv_id": conversation_id,
                    "signal_uid": signal.uid,
                },
            )
        except Exception as e:
            logger.debug(f"Could not link signal to conversation {conversation_id}: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# CONVENIENCE FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────


def generate_signals(
    message: str,
    conversation_id: str,
    conversation_context: str = "",
    participants: Optional[list[str]] = None,
    turn_number: int = 0,
    use_llm: bool = True,
) -> list[Signal]:
    """
    Convenience function for generating signals from a message.

    Args:
        message: User message text
        conversation_id: Conversation ID
        conversation_context: Previous conversation text
        participants: Known participant names
        turn_number: Current turn number
        use_llm: Whether to use LLM (True) or keyword fallback (False)

    Returns:
        List of created Signal nodes
    """
    generator = SignalGenerator(use_llm=use_llm)
    return generator.generate_from_message(
        message=message,
        conversation_id=conversation_id,
        conversation_context=conversation_context,
        participants=participants,
        turn_number=turn_number,
    )


def explore_wildcard(
    signal: Signal,
    conversation_context: str = "",
) -> dict:
    """
    Generate an exploration question for a signal's wildcards.

    Args:
        signal: Signal node with wildcard coordinates
        conversation_context: Recent conversation text

    Returns:
        Dict with target_wildcard, question, reasoning, suggested_values
    """
    wildcards = signal.wildcard_coordinates or []
    if not wildcards:
        return {"target_wildcard": None, "question": None, "reasoning": "No wildcards to explore"}

    try:
        prompt = get_prompt(
            "wildcard_exploration",
            signal_address=signal.signal_address or "SA(*, *, *, *)",
            wildcards=", ".join(wildcards),
            context=conversation_context[:2000],
        )

        result = llm_client.analyze_with_retry(prompt, "Help me explore this signal.")
        return result

    except Exception as e:
        logger.warning(f"Wildcard exploration failed: {e}")
        # Return a generic question
        target = wildcards[0]
        generic_questions = {
            "person": "Who was involved in this moment?",
            "context": "Where were you when this happened?",
            "action": "What specifically happened?",
            "temporal": "When did this happen?",
        }
        return {
            "target_wildcard": target,
            "question": generic_questions.get(target, f"Can you tell me more about the {target}?"),
            "reasoning": f"Exploring {target} wildcard (fallback question)",
            "suggested_values": [],
        }
