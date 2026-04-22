"""
ThriveSight Conversation Analysis Pipeline — First Three Stages

This module implements the first three stages of the ThriveSight analysis pipeline:

1. ConversationParser: Converts raw input into normalized Turn Schema
2. DialogueSignalGenerator: Classifies emotional signals and trigger actions
3. TriggerActionInference: Enriches signals with trigger action categorization

The module also provides:
- ConversationGraphWriter (optional): Writes results to Memgraph
- analyze_conversation: Top-level orchestration function

Every output is validated against the JSON contracts before entering the next stage.
"""

import json
import logging
import re
from difflib import SequenceMatcher
from datetime import datetime
from typing import Any, Optional

from . import llm_client
from . import validators

logger = logging.getLogger(__name__)


# ============================================================================
# DETERMINISTIC PARSING PATTERNS
# ============================================================================

# Regex for "Speaker: text" format
LABELED_DIALOGUE_PATTERN = re.compile(
    r"^\s*([A-Za-z0-9_\-\s]+?):\s+(.+?)$",
    re.MULTILINE
)

# Regex for "[timestamp] Speaker: text" format
TIMESTAMPED_CHAT_PATTERN = re.compile(
    r"^\s*\[([^\]]+)\]\s+([A-Za-z0-9_\-\s]+?):\s+(.+?)$",
    re.MULTILINE
)

# Reaction type enum (fixed vocabulary)
VALID_REACTIONS = [
    "defended",
    "counter_attacked",
    "withdrew",
    "de_escalated",
    "acknowledged",
    "escalated",
    "deflected",
    "conceded",
]

# Known emotions (extended set from constrained-generation.md)
KNOWN_EMOTIONS = {
    "frustration",
    "defensiveness",
    "anger",
    "sadness",
    "anxiety",
    "hurt",
    "contempt",
    "warmth",
    "humor",
    "resignation",
    "guilt",
    "relief",
    "hope",
    "resentment",
    "vulnerability",
    "confusion",
    "empathy",
    "indifference",
}

# Seed trigger categories (static vocabulary)
SEED_CATEGORIES = {
    "dismissal": "Minimizing or ignoring the other person's concern.",
    "accusation": "Directly blaming the other person with absolutist language.",
    "deflection": "Changing the subject or redirecting blame.",
    "withdrawal": "Emotionally or physically disengaging.",
    "demand": "Issuing an ultimatum or insisting on action.",
    "questioning": "Asking for clarification or information.",
    "validation": "Acknowledging the other person's perspective as legitimate.",
    "acknowledgment": "Recognizing the situation without necessarily agreeing.",
    "concession": "Yielding ground or accepting responsibility.",
    "sarcasm": "Using irony or mocking tone to undermine.",
    "initiation": "Opening a conversation topic.",
}


# ============================================================================
# CLASS 1: CONVERSATION PARSER
# ============================================================================

class ConversationParser:
    """
    Converts raw input into normalized Turn Schema (array of turns with speakers).

    Uses deterministic regex parsing for known formats, falls back to LLM
    for unstructured text.

    Output: Conversation envelope with speakers array and turns array.
    """

    def process(self, raw_input: str, source_type: str = "text") -> dict[str, Any]:
        """
        Parse raw input into conversation turns.

        Args:
            raw_input: Raw conversation text or JSON
            source_type: Format hint ('text', 'json', 'jsonl', 'auto')

        Returns:
            Conversation envelope dict with keys:
            - conversation_title: str
            - speakers: list[str]
            - turns: list[dict] conforming to Turn Schema
            - parse_metadata: dict with format_detected, llm_assisted, total_turns
        """
        # Attempt deterministic parsing first
        parsed = self._try_deterministic_parse(raw_input, source_type)

        if parsed:
            return parsed

        # Fall back to LLM-assisted parsing
        logger.info("No deterministic format match; using LLM-assisted parsing")
        return self._llm_assisted_parse(raw_input)

    def _try_deterministic_parse(
        self, raw_input: str, source_type: str
    ) -> Optional[dict[str, Any]]:
        """
        Try deterministic parsing for known formats.
        Returns None if no format matches.
        """
        # JSON format
        if source_type in ("json", "auto"):
            try:
                data = json.loads(raw_input)
                if isinstance(data, dict) and "turns" in data:
                    return self._parse_json_envelope(data)
                elif isinstance(data, list):
                    return self._parse_json_list(data)
            except json.JSONDecodeError:
                pass

        # JSONL format
        if source_type in ("jsonl", "auto"):
            result = self._try_jsonl_parse(raw_input)
            if result:
                return result

        # Timestamped chat format: [HH:MM] Speaker: text
        if source_type in ("text", "auto"):
            result = self._try_timestamped_parse(raw_input)
            if result:
                return result

        # Labeled dialogue format: Speaker: text
        if source_type in ("text", "auto"):
            result = self._try_labeled_parse(raw_input)
            if result:
                return result

        return None

    def _parse_json_envelope(self, data: dict) -> dict[str, Any]:
        """Parse JSON object with 'turns' key."""
        turns = data.get("turns", [])
        speakers = data.get("speakers", self._extract_speakers_from_turns(turns))
        title = data.get("conversation_title", "Imported Conversation")

        # Validate turns and fix if needed
        turns = self._normalize_turns(turns)

        return {
            "conversation_title": title,
            "speakers": speakers,
            "turns": turns,
            "parse_metadata": {
                "format_detected": "json",
                "llm_assisted": False,
                "total_turns": len(turns),
            },
        }

    def _parse_json_list(self, data: list) -> dict[str, Any]:
        """Parse JSON array of turn objects."""
        turns = self._normalize_turns(data)
        speakers = self._extract_speakers_from_turns(turns)

        return {
            "conversation_title": "Imported Conversation",
            "speakers": speakers,
            "turns": turns,
            "parse_metadata": {
                "format_detected": "json",
                "llm_assisted": False,
                "total_turns": len(turns),
            },
        }

    def _try_jsonl_parse(self, raw_input: str) -> Optional[dict[str, Any]]:
        """Try parsing JSONL (one JSON object per line)."""
        lines = raw_input.strip().split("\n")
        turns = []

        try:
            for line in lines:
                if line.strip():
                    obj = json.loads(line)
                    turns.append(obj)
        except json.JSONDecodeError:
            return None

        if not turns:
            return None

        turns = self._normalize_turns(turns)
        speakers = self._extract_speakers_from_turns(turns)

        return {
            "conversation_title": "Imported Conversation",
            "speakers": speakers,
            "turns": turns,
            "parse_metadata": {
                "format_detected": "jsonl",
                "llm_assisted": False,
                "total_turns": len(turns),
            },
        }

    def _try_timestamped_parse(self, raw_input: str) -> Optional[dict[str, Any]]:
        """Try parsing [timestamp] Speaker: text format."""
        matches = TIMESTAMPED_CHAT_PATTERN.findall(raw_input)

        if not matches:
            return None

        turns = []
        speakers = set()
        raw_offset = 0

        for turn_number, (timestamp, speaker, text) in enumerate(matches, start=1):
            speakers.add(speaker)
            turns.append(
                {
                    "turn_number": turn_number,
                    "speaker": speaker,
                    "text": text.strip(),
                    "timestamp": timestamp,
                    "raw_offset": raw_input.find(f"[{timestamp}]"),
                }
            )

        if not turns:
            return None

        speakers_list = sorted(list(speakers))

        return {
            "conversation_title": "Chat Conversation",
            "speakers": speakers_list,
            "turns": turns,
            "parse_metadata": {
                "format_detected": "timestamped_chat",
                "llm_assisted": False,
                "total_turns": len(turns),
            },
        }

    def _try_labeled_parse(self, raw_input: str) -> Optional[dict[str, Any]]:
        """Try parsing Speaker: text format."""
        matches = LABELED_DIALOGUE_PATTERN.findall(raw_input)

        if not matches:
            return None

        turns = []
        speakers = set()

        for turn_number, (speaker, text) in enumerate(matches, start=1):
            speaker = speaker.strip()
            speakers.add(speaker)

            # Find raw offset in original text
            search_pattern = f"{speaker}:"
            raw_offset = raw_input.find(search_pattern)

            turns.append(
                {
                    "turn_number": turn_number,
                    "speaker": speaker,
                    "text": text.strip(),
                    "timestamp": None,
                    "raw_offset": raw_offset,
                }
            )

        if not turns:
            return None

        speakers_list = sorted(list(speakers))

        return {
            "conversation_title": "Dialogue",
            "speakers": speakers_list,
            "turns": turns,
            "parse_metadata": {
                "format_detected": "labeled_dialogue",
                "llm_assisted": False,
                "total_turns": len(turns),
            },
        }

    def _llm_assisted_parse(self, raw_input: str) -> dict[str, Any]:
        """Use LLM to parse unstructured text."""
        system_prompt = llm_client.PARSER_SYSTEM
        user_prompt = f"Parse this conversation:\n\n{raw_input}"

        try:
            result = llm_client.analyze_with_retry(system_prompt, user_prompt)
        except ValueError as e:
            logger.error(f"LLM parsing failed: {e}")
            # Fallback: treat the entire input as a single turn from an unknown speaker
            return {
                "conversation_title": "Unparseable Conversation",
                "speakers": ["Unknown"],
                "turns": [
                    {
                        "turn_number": 1,
                        "speaker": "Unknown",
                        "text": raw_input[:500],  # Truncate for safety
                        "timestamp": None,
                        "raw_offset": 0,
                    }
                ],
                "parse_metadata": {
                    "format_detected": "unparseable",
                    "llm_assisted": False,
                    "total_turns": 1,
                },
            }

        speakers = result.get("speakers", ["Speaker A", "Speaker B"])
        turns = result.get("turns", [])
        turns = self._normalize_turns(turns)

        return {
            "conversation_title": "LLM-Parsed Conversation",
            "speakers": speakers,
            "turns": turns,
            "parse_metadata": {
                "format_detected": "unstructured",
                "llm_assisted": True,
                "total_turns": len(turns),
            },
        }

    def _normalize_turns(self, turns: list) -> list[dict[str, Any]]:
        """Ensure all turns conform to Turn Schema."""
        normalized = []

        for i, turn in enumerate(turns, start=1):
            if not isinstance(turn, dict):
                continue

            normalized_turn = {
                "turn_number": turn.get("turn_number", i),
                "speaker": turn.get("speaker", f"Speaker {chr(65 + (i % 26))}"),
                "text": turn.get("text", ""),
                "timestamp": turn.get("timestamp"),
                "raw_offset": turn.get("raw_offset", 0),
            }

            # Validate
            errors = validators.validate_turns([normalized_turn])
            if not errors:
                normalized.append(normalized_turn)

        return normalized

    def _extract_speakers_from_turns(self, turns: list) -> list[str]:
        """Extract unique speakers from turns in order of appearance."""
        speakers = []
        seen = set()

        for turn in turns:
            speaker = turn.get("speaker", "Unknown")
            if speaker not in seen:
                speakers.append(speaker)
                seen.add(speaker)

        return speakers


# ============================================================================
# CLASS 2: DIALOGUE SIGNAL GENERATOR
# ============================================================================

class DialogueSignalGenerator:
    """
    Generates Signal Schema objects from parsed conversation turns.

    Calls the LLM to classify emotion, intensity, reaction, and trigger action
    for each turn. Computes signal_address SA(c, p, a, t).

    Validates output through validators.validate_signals().
    """

    def __init__(self, categories: Optional[dict[str, str]] = None):
        """
        Initialize signal generator.

        Args:
            categories: Optional dict of {category_name: description}.
                       Defaults to seed categories.
        """
        self.categories = categories or SEED_CATEGORIES
        self.used_fallback = False

    def process(self, parsed_conversation: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Generate signals from parsed conversation.

        Args:
            parsed_conversation: Output from ConversationParser.process()

        Returns:
            List of Signal Schema objects
        """
        turns = parsed_conversation.get("turns", [])
        speakers = parsed_conversation.get("speakers", [])

        if not turns:
            logger.warning("No turns to process")
            return []

        # Build conversation text for LLM context
        conversation_text = self._build_conversation_text(turns)

        # Call LLM to generate signals
        try:
            signals = self._generate_signals_llm(conversation_text)
            self.used_fallback = False
        except Exception as e:
            logger.warning(f"LLM signal generation failed ({type(e).__name__}: {e}); using keyword fallback")
            signals = self._apply_fallback_signals(turns)
            self.used_fallback = True

        # Validate signals
        validation_errors = validators.validate_signals(signals)
        if validation_errors:
            logger.warning(f"Signal validation errors: {validation_errors}")

        return signals

    def _build_conversation_text(self, turns: list) -> str:
        """Format turns as readable conversation text."""
        lines = []
        for turn in turns:
            speaker = turn.get("speaker", "Unknown")
            text = turn.get("text", "")
            lines.append(f"{speaker}: {text}")

        return "\n".join(lines)

    def _generate_signals_llm(self, conversation_text: str) -> list[dict[str, Any]]:
        """Call LLM to generate signals."""
        # Prepare categories list for prompt
        categories_list = [
            {"name": k, "description": v} for k, v in self.categories.items()
        ]

        system_prompt, user_prompt = llm_client.build_signal_prompt(
            conversation_text, categories_list
        )

        try:
            result = llm_client.analyze_with_retry(system_prompt, user_prompt)
        except ValueError as e:
            raise ValueError(f"LLM signal generation failed: {e}")

        # Result should be a list of signals
        if isinstance(result, dict) and "signals" in result:
            return result["signals"]
        elif isinstance(result, list):
            return result
        else:
            raise ValueError(f"Unexpected LLM response format: {type(result)}")

    # Keyword lexicons for deterministic fallback
    EMOTION_KEYWORDS = {
        "frustration": ["frustrated", "annoying", "sick of", "tired of", "can't believe", "ugh"],
        "anger": ["angry", "furious", "mad", "pissed", "rage", "how dare"],
        "defensiveness": ["not my fault", "i didn't", "don't blame", "that's not fair", "it's not like", "defensive"],
        "sadness": ["sad", "hurt", "miss", "cry", "lonely", "depressed", "heartbroken"],
        "anxiety": ["worried", "scared", "nervous", "anxious", "afraid", "what if"],
        "contempt": ["pathetic", "ridiculous", "whatever", "you always", "you never"],
        "warmth": ["love", "appreciate", "thank", "grateful", "proud of you", "care about"],
        "hope": ["maybe we could", "what if we", "we could try", "i think we can", "good idea"],
        "guilt": ["sorry", "my fault", "i shouldn't have", "i apologize", "i feel bad"],
        "vulnerability": ["i feel", "it makes me", "i need", "i'm afraid", "it hurts when"],
        "resignation": ["fine", "whatever", "i give up", "doesn't matter", "forget it", "never mind"],
    }

    TRIGGER_KEYWORDS = {
        "dismissal": ["not that bad", "overreacting", "big deal", "not a big deal", "calm down", "it's nothing", "get over it"],
        "accusation": ["you always", "you never", "your fault", "you're the one", "you did this", "blame"],
        "deflection": ["what about", "that's different", "changing the subject", "not the point", "you're deflecting"],
        "withdrawal": ["i'm done", "leave me alone", "whatever", "i don't care", "shut down", "not talking"],
        "demand": ["you need to", "you have to", "you must", "i forbid", "do it now", "that's final"],
        "questioning": ["can we talk", "why did you", "what happened", "how come", "could you explain", "i want to understand"],
        "validation": ["i hear you", "i understand", "that makes sense", "you're right", "i see your point", "i get it"],
        "acknowledgment": ["i see that", "i can see", "i hear that", "okay", "you have a point"],
        "concession": ["i'm sorry", "you're right", "i was wrong", "i shouldn't have", "my fault", "i apologize"],
        "sarcasm": ["oh great", "sure, right", "yeah right", "how wonderful", "of course you"],
    }

    REACTION_KEYWORDS = {
        "defended": ["i didn't", "that's not", "it's not like", "not what i said"],
        "escalated": ["well maybe if you", "you're the one", "don't even", "how dare"],
        "withdrew": ["fine", "whatever", "i'm done", "forget it"],
        "de_escalated": ["let's", "maybe we could", "i understand", "i hear you", "you're right"],
        "acknowledged": ["i see", "okay", "i get it", "that makes sense"],
        "deflected": ["what about", "that's not the point", "you're changing"],
        "conceded": ["i'm sorry", "you're right", "my fault", "i apologize"],
    }

    def _apply_fallback_signals(self, turns: list) -> list[dict[str, Any]]:
        """Apply keyword-based deterministic fallback when LLM fails."""
        signals = []

        for i, turn in enumerate(turns):
            turn_number = turn.get("turn_number", i + 1)
            speaker = turn.get("speaker", "Unknown")
            text = turn.get("text", "").lower()

            # Detect emotion via keyword matching
            emotion = self._detect_emotion_keywords(text)

            # Detect trigger category via keyword matching
            category = self._detect_trigger_keywords(text)

            # Detect reaction via keyword matching
            reaction = self._detect_reaction_keywords(text)

            # Compute intensity from punctuation + keywords + caps
            intensity = self._compute_intensity(turn.get("text", ""))

            # Build action text from the turn (what the OTHER person did that triggered this)
            if i > 0:
                prev_text = turns[i - 1].get("text", "")
                action_text = prev_text[:80] if prev_text else "previous statement"
            else:
                action_text = "conversation opener"
                category = category or "initiation"

            signal = {
                "turn_number": turn_number,
                "speaker": speaker,
                "text": turn.get("text", ""),
                "emotion": emotion or "neutral",
                "intensity": intensity,
                "reaction": reaction or "acknowledged",
                "trigger_action": {
                    "action_text": action_text,
                    "category": category or "questioning",
                    "is_new_category": False,
                    "category_description": None,
                },
                "signal_address": f"SA(*, {speaker}, {category or 'unclassified'}, turn_{turn_number - 1})",
                "_fallback": True,
            }
            signals.append(signal)

        return signals

    def _detect_emotion_keywords(self, text: str) -> Optional[str]:
        """Match text against emotion keyword lexicon."""
        best_match = None
        best_count = 0
        for emotion, keywords in self.EMOTION_KEYWORDS.items():
            count = sum(1 for kw in keywords if kw in text)
            if count > best_count:
                best_count = count
                best_match = emotion
        return best_match

    def _detect_trigger_keywords(self, text: str) -> Optional[str]:
        """Match text against trigger category keyword lexicon."""
        best_match = None
        best_count = 0
        for category, keywords in self.TRIGGER_KEYWORDS.items():
            count = sum(1 for kw in keywords if kw in text)
            if count > best_count:
                best_count = count
                best_match = category
        return best_match

    def _detect_reaction_keywords(self, text: str) -> Optional[str]:
        """Match text against reaction keyword lexicon."""
        best_match = None
        best_count = 0
        for reaction, keywords in self.REACTION_KEYWORDS.items():
            count = sum(1 for kw in keywords if kw in text)
            if count > best_count:
                best_count = count
                best_match = reaction
        return best_match

    def _compute_intensity(self, text: str) -> float:
        """Compute intensity score from linguistic markers."""
        intensity = 1.5  # baseline

        # Exclamation marks
        intensity += min(text.count("!") * 0.5, 1.5)

        # Question marks (mild escalation)
        intensity += min(text.count("?") * 0.2, 0.6)

        # ALL CAPS words
        words = text.split()
        caps_count = sum(1 for w in words if w.isupper() and len(w) > 1)
        intensity += min(caps_count * 0.4, 1.0)

        # Absolutist language
        absolutist = ["always", "never", "every", "nothing", "everything", "nobody"]
        intensity += sum(0.3 for word in absolutist if word in text.lower())

        # Ellipsis (de-escalation/withdrawal)
        if "..." in text:
            intensity -= 0.3

        # Clamp to 1.0-5.0
        return max(1.0, min(5.0, round(intensity, 1)))


# ============================================================================
# CLASS 3: TRIGGER ACTION INFERENCE
# ============================================================================

class TriggerActionInference:
    """
    Enriches signals with trigger action classification.

    Handles new category proposals and checks for similar existing categories
    before creating new ones. Tracks usage_count for existing categories.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.7,
        validation_threshold: int = 3,
    ):
        """
        Initialize trigger action inference.

        Args:
            similarity_threshold: String similarity threshold for duplicate detection
            validation_threshold: Usage count threshold to validate a proposed category
        """
        self.similarity_threshold = similarity_threshold
        self.validation_threshold = validation_threshold
        self.existing_categories = dict(SEED_CATEGORIES)
        self.category_usage = {cat: 0 for cat in SEED_CATEGORIES.keys()}

    def process(self, signals: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Enrich signals with trigger action categorization.

        Args:
            signals: List of Signal objects (output from DialogueSignalGenerator)

        Returns:
            Enriched list of Signal objects with updated trigger_action fields
        """
        enriched = []

        for signal in signals:
            trigger_action = signal.get("trigger_action", {})

            # Check if this is a new category proposal
            if trigger_action.get("is_new_category"):
                trigger_action = self._handle_new_category(trigger_action)
                signal["trigger_action"] = trigger_action

            # Track usage
            category = trigger_action.get("category")
            if category and category in self.category_usage:
                self.category_usage[category] += 1

            enriched.append(signal)

        return enriched

    def _handle_new_category(self, trigger_action: dict[str, Any]) -> dict[str, Any]:
        """
        Handle a proposed new trigger category.

        - Check for similar existing categories
        - If similar, map to existing
        - If new, create and track with usage_count = 1

        Args:
            trigger_action: Trigger action object with is_new_category = True

        Returns:
            Updated trigger_action (may map to existing category)
        """
        proposed_name = trigger_action.get("category", "unknown")
        proposed_desc = trigger_action.get("category_description", "")

        # Check for similar category
        similar = self._find_similar_category(proposed_name, proposed_desc)

        if similar:
            logger.info(
                f"Proposed category '{proposed_name}' maps to existing '{similar}'"
            )
            trigger_action["category"] = similar
            trigger_action["is_new_category"] = False
            trigger_action["category_description"] = None
        else:
            # Accept new category
            logger.info(f"Accepting new category: {proposed_name}")
            self.existing_categories[proposed_name] = proposed_desc
            self.category_usage[proposed_name] = 1

        return trigger_action

    def _find_similar_category(self, proposed_name: str, proposed_desc: str) -> Optional[str]:
        """
        Check if a proposed category is too similar to an existing one.

        Uses string similarity on both name and description.

        Args:
            proposed_name: Proposed category name
            proposed_desc: Proposed category description

        Returns:
            Name of similar category, or None if no match
        """
        for existing_name, existing_desc in self.existing_categories.items():
            # Name similarity
            name_ratio = SequenceMatcher(
                None, proposed_name.lower(), existing_name.lower()
            ).ratio()
            if name_ratio > self.similarity_threshold:
                return existing_name

            # Description similarity
            if proposed_desc and existing_desc:
                desc_ratio = SequenceMatcher(
                    None, proposed_desc.lower(), existing_desc.lower()
                ).ratio()
                if desc_ratio > self.similarity_threshold:
                    return existing_name

        return None

    def get_category_stats(self) -> dict[str, int]:
        """Get current category usage counts."""
        return dict(self.category_usage)


# ============================================================================
# CLASS 4: CONVERSATION GRAPH WRITER (OPTIONAL)
# ============================================================================

class ConversationGraphWriter:
    """
    Optional graph writer for Memgraph.

    Writes parsed conversation and signals to the graph database.
    Gracefully handles connection failures.

    Creates nodes: Conversation, Person, Turn, Signal, TriggerAction, TriggerCategory
    Creates edges: CONTAINS, NEXT, SPOKEN_BY, PRODUCES, TRIGGERED_BY, HAS_CATEGORY
    """

    def write_to_graph(
        self,
        parsed_conversation: dict[str, Any],
        signals: list[dict[str, Any]],
        graph_scope: dict[str, Any] | None = None,
    ) -> bool:
        """
        Write conversation and signals to the graph database.

        Args:
            parsed_conversation: Output from ConversationParser
            signals: Output from DialogueSignalGenerator/TriggerActionInference

        Returns:
            True if write succeeded, False if graph unavailable
        """
        try:
            from .graph_models import (
                Conversation,
                Person,
                Turn,
                Signal,
                TriggerAction,
                TriggerCategory,
            )
        except ImportError:
            logger.warning("Graph models not available; skipping graph write")
            return False

        try:
            graph_scope = graph_scope or {}
            workspace_id = graph_scope.get("workspace_id")
            owner_user_id = graph_scope.get("owner_user_id")
            # Create conversation node
            conv_title = parsed_conversation.get("conversation_title", "Untitled")
            speakers = parsed_conversation.get("speakers", [])
            turns = parsed_conversation.get("turns", [])

            conversation = Conversation(
                workspace_id=workspace_id,
                owner_user_id=owner_user_id,
                title=conv_title,
                speaker_count=len(speakers),
                turn_count=len(turns),
            ).save()

            # Create person nodes and link to conversation
            person_nodes = {}
            for speaker in speakers:
                person = Person(
                    workspace_id=workspace_id,
                    owner_user_id=owner_user_id,
                    name=speaker,
                ).save()
                person_nodes[speaker] = person
                conversation.participants.connect(person)

            # Create turn nodes and link
            turn_nodes = {}
            prev_turn_node = None

            for turn in turns:
                turn_num = turn.get("turn_number", 0)
                speaker = turn.get("speaker", "Unknown")
                text = turn.get("text", "")

                turn_node = Turn(
                    workspace_id=workspace_id,
                    owner_user_id=owner_user_id,
                    turn_number=turn_num,
                    text=text,
                    timestamp=turn.get("timestamp"),
                ).save()

                turn_nodes[turn_num] = turn_node
                conversation.turns.connect(turn_node)
                turn_node.speaker.connect(person_nodes.get(speaker, person_nodes.get("Unknown")))

                # Link to previous turn
                if prev_turn_node:
                    prev_turn_node.next_turn.connect(turn_node)

                prev_turn_node = turn_node

            # Create signal and trigger category nodes
            for signal in signals:
                turn_num = signal.get("turn_number")
                emotion = signal.get("emotion", "unknown")
                intensity = signal.get("intensity", 1.0)
                reaction = signal.get("reaction", "acknowledged")
                signal_address = signal.get("signal_address", "SA(*,*,*,*)")

                signal_node = Signal(
                    workspace_id=workspace_id,
                    owner_user_id=owner_user_id,
                    emotion=emotion,
                    intensity=intensity,
                    reaction=reaction,
                    signal_address=signal_address,
                ).save()

                # Link signal to turn
                if turn_num in turn_nodes:
                    turn_nodes[turn_num].signal.connect(signal_node)

                # Handle trigger category
                trigger_action = signal.get("trigger_action", {})
                if trigger_action:
                    category_name = trigger_action.get("category", "unknown")
                    is_new = trigger_action.get("is_new_category", False)
                    category_desc = trigger_action.get("category_description")

                    # Find or create trigger category
                    category_node = TriggerCategory(
                        name=category_name,
                        description=category_desc or "",
                        is_proposed=is_new,
                        usage_count=1,
                    ).save()

                    signal_node.trigger.connect(category_node)

            logger.info(f"Successfully wrote conversation '{conv_title}' to graph")
            return True

        except Exception as e:
            logger.error(f"Graph write failed: {e}")
            return False


# ============================================================================
# TOP-LEVEL ORCHESTRATION FUNCTION
# ============================================================================

def analyze_conversation(
    raw_input: str,
    source_type: str = "text",
    write_to_graph: bool = False,
    graph_scope: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Orchestrate all three pipeline stages: parse → generate signals → infer triggers.

    Args:
        raw_input: Raw conversation input (text, JSON, JSONL, etc.)
        source_type: Format hint ('text', 'json', 'jsonl', 'auto')
        write_to_graph: Whether to write results to graph database

    Returns:
        Analysis result dict with keys:
        - conversation: Conversation metadata
        - signals: List of Signal objects
        - metadata: Processing metadata
        - graph_written: Boolean (only if write_to_graph=True)
    """
    import time

    start_time = time.time()
    metadata = {
        "processing_time_ms": 0,
        "llm_calls": 0,
        "new_categories_proposed": [],
        "validation_warnings": [],
        "stages": {
            "parser": {"method": "unknown"},
            "signals": {"method": "unknown"},
            "triggers": {"method": "unknown"},
        },
    }

    # Stage 1: Parse — this MUST succeed or we have nothing
    logger.info("Stage 1: Parsing conversation...")
    parser = ConversationParser()
    try:
        parsed_conversation = parser.process(raw_input, source_type)
    except Exception as e:
        logger.error(f"Parser failed: {e}")
        return {
            "conversation": {
                "title": "Parse Failed",
                "speakers": [],
                "total_turns": 0,
                "turns": [],
                "parse_metadata": {},
            },
            "signals": [],
            "metadata": metadata,
            "error": str(e),
        }

    # Build conversation envelope (preserved regardless of downstream failures)
    parse_meta = parsed_conversation.get("parse_metadata", {})
    conversation = {
        "title": parsed_conversation.get("conversation_title", "Untitled"),
        "speakers": parsed_conversation.get("speakers", []),
        "total_turns": len(parsed_conversation.get("turns", [])),
        "turns": parsed_conversation.get("turns", []),
        "parse_metadata": parse_meta,
    }
    metadata["stages"]["parser"] = {
        "method": "llm" if parse_meta.get("llm_assisted") else "deterministic",
        "format": parse_meta.get("format_detected", "unknown"),
    }

    # Stage 2: Generate signals — fallback to keyword-based if LLM fails
    signals = []
    logger.info("Stage 2: Generating signals...")
    try:
        signal_generator = DialogueSignalGenerator()
        signals = signal_generator.process(parsed_conversation)
        if signal_generator.used_fallback:
            metadata["stages"]["signals"] = {"method": "keyword_fallback"}
        else:
            metadata["stages"]["signals"] = {"method": "llm"}
            metadata["llm_calls"] += 1
    except Exception as e:
        logger.error(f"Signal generation completely failed: {e}")
        metadata["stages"]["signals"] = {"method": "failed"}
        metadata["validation_warnings"].append(f"Signal generation failed: {str(e)}")

    # Stage 3: Infer trigger actions
    new_categories = []
    if signals:
        logger.info("Stage 3: Inferring trigger actions...")
        try:
            trigger_inference = TriggerActionInference()
            signals = trigger_inference.process(signals)
            category_stats = trigger_inference.get_category_stats()
            new_categories = [
                cat for cat, count in category_stats.items()
                if cat not in SEED_CATEGORIES
            ]
        except Exception as e:
            logger.error(f"Trigger inference failed: {e}")
            metadata["validation_warnings"].append(f"Trigger inference failed: {str(e)}")

    metadata["new_categories_proposed"] = new_categories

    # Optional: Write to graph
    graph_written = False
    if write_to_graph and signals:
        logger.info("Writing to graph...")
        try:
            writer = ConversationGraphWriter()
            graph_written = writer.write_to_graph(parsed_conversation, signals, graph_scope=graph_scope)
        except Exception as e:
            logger.error(f"Graph write failed: {e}")

    # Prepare response
    end_time = time.time()
    metadata["processing_time_ms"] = int((end_time - start_time) * 1000)

    response = {
        "conversation": conversation,
        "signals": signals,
        "metadata": metadata,
    }

    if write_to_graph:
        response["graph_written"] = graph_written

    logger.info(
        f"Analysis complete: {len(signals)} signals, "
        f"{len(new_categories)} new categories, "
        f"{metadata['processing_time_ms']}ms"
    )

    return response
