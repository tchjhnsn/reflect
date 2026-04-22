"""
ThriveSight Character Network — Layer 2 Allocation Engine

This module is the network builder: it takes the parser's structured output
(skeleton JSON) and exhaustively allocates every piece of data into graph-ready
structures. It does not interpret — it counts, maps, and connects.

The allocation engine produces:
- Character co-occurrence matrices (who appears with whom, how often)
- Vocabulary frequency maps per character (vocabulary_json compact tier)
- Relationship edge data (significance scoring: 40% co-occurrence + 40% shared dialogue + 20% witness)
- Dialogue targeting (DIRECTED_AT inference using Option C: name-match → sequential → parenthetical)
- Scene participation roles (focal, present, witness, mentioned)

The engine operates in two modes:
1. build_network(skeleton_json) → NetworkResult (pure data, no graph writes)
2. write_to_graph(network_result) → writes NetworkResult into Memgraph via neomodel

Mode 1 can run without a graph database and is used for testing and preview.
Mode 2 requires Memgraph to be running.
"""

import json
import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# STOP WORDS — filtered from vocabulary counting
# ──────────────────────────────────────────────────────────────────────────────

STOP_WORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "it", "was", "are", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "this", "that",
    "these", "those", "i", "me", "my", "we", "us", "our", "you", "your",
    "he", "him", "his", "she", "her", "they", "them", "their", "its",
    "what", "which", "who", "whom", "how", "when", "where", "why",
    "not", "no", "so", "if", "then", "than", "too", "very", "just",
    "about", "up", "out", "all", "there", "here", "some", "any", "each",
    "every", "as", "into", "through", "during", "before", "after",
    "above", "below", "between", "same", "other", "such", "only",
    "own", "more", "most", "now", "also", "well", "back", "even",
    "still", "already", "much", "many", "s", "t", "re", "ve", "ll",
    "m", "d", "don", "doesn", "didn", "isn", "wasn", "aren", "won",
    "wouldn", "couldn", "shouldn", "haven", "hasn", "hadn", "ain",
    "oh", "yeah", "yes", "okay", "ok", "uh", "um", "like", "got",
    "get", "go", "going", "went", "come", "came", "know", "think",
    "say", "said", "tell", "told", "want", "need", "let", "make",
    "take", "see", "look", "give", "find", "thing", "things",
})

# Minimum word length for vocabulary tracking
MIN_WORD_LENGTH = 3


# ──────────────────────────────────────────────────────────────────────────────
# DATA CLASSES — Network Result
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class DirectedAtResult:
    """Result of DIRECTED_AT inference for a single turn."""
    turn_number: int
    speaker: str
    target_character: str
    inference_method: str       # "parenthetical" | "name_match" | "sequential" | "action_line_context"
    confidence: str             # "explicit" | "high" | "inferred"


@dataclass
class RelationshipData:
    """Computed relationship data between two characters."""
    source_character: str
    target_character: str
    co_occurrence_count: int = 0
    shared_dialogue_scenes: int = 0
    witness_scenes: int = 0
    first_observed_scene: Optional[int] = None
    last_observed_scene: Optional[int] = None
    significance_score: float = 0.0

    def compute_significance(self, max_co_occurrence: int, max_shared: int, max_witness: int):
        """Compute significance score using the 40/40/20 formula."""
        co_norm = self.co_occurrence_count / max_co_occurrence if max_co_occurrence > 0 else 0
        shared_norm = self.shared_dialogue_scenes / max_shared if max_shared > 0 else 0
        witness_norm = self.witness_scenes / max_witness if max_witness > 0 else 0
        self.significance_score = round(0.4 * co_norm + 0.4 * shared_norm + 0.2 * witness_norm, 4)


@dataclass
class VocabularyEntry:
    """Per-word vocabulary data for a character."""
    count: int = 0
    scenes: list = field(default_factory=list)
    directed_at: dict = field(default_factory=dict)     # { character_name: count }
    first_observed_scene: Optional[int] = None
    last_observed_scene: Optional[int] = None

    def to_dict(self):
        return {
            "count": self.count,
            "scenes": sorted(set(self.scenes)),
            "directed_at": dict(self.directed_at),
            "first_observed_scene": self.first_observed_scene,
            "last_observed_scene": self.last_observed_scene,
        }


@dataclass
class CharacterData:
    """Aggregated data for a single character."""
    name: str
    total_words: int = 0
    unique_words: int = 0
    scene_appearances: list = field(default_factory=list)   # [{ scene_number, role }]
    vocabulary: dict = field(default_factory=dict)           # { word: VocabularyEntry }
    dialogue_turn_count: int = 0

    def build_vocabulary_json(self):
        """Build the vocabulary_json compact tier structure."""
        word_freqs = {}
        for word, entry in self.vocabulary.items():
            word_freqs[word] = entry.to_dict()

        return {
            "total_words": self.total_words,
            "unique_words": self.unique_words,
            "word_frequencies": word_freqs,
            "bigrams": {},                  # Populated in a future pass
            "emotional_context_map": {},    # Populated by LLM audit loop Pass 1
        }


@dataclass
class NetworkResult:
    """Complete output of the allocation engine."""
    # Source metadata
    source_work: str
    focal_character: str
    total_scenes: int
    total_turns: int
    total_characters: int

    # Allocated data
    characters: dict = field(default_factory=dict)          # { name: CharacterData }
    relationships: list = field(default_factory=list)        # [ RelationshipData ]
    directed_at_results: list = field(default_factory=list)  # [ DirectedAtResult ]
    scene_roles: dict = field(default_factory=dict)          # { scene_number: { character: role } }

    def to_summary(self):
        """Human-readable summary of the network build."""
        rel_count = len(self.relationships)
        da_count = len(self.directed_at_results)
        chars_with_vocab = sum(1 for c in self.characters.values() if c.vocabulary)
        return {
            "source_work": self.source_work,
            "focal_character": self.focal_character,
            "total_scenes": self.total_scenes,
            "total_turns": self.total_turns,
            "total_characters": self.total_characters,
            "relationships_computed": rel_count,
            "directed_at_inferred": da_count,
            "characters_with_vocabulary": chars_with_vocab,
        }


# ──────────────────────────────────────────────────────────────────────────────
# CORE ENGINE
# ──────────────────────────────────────────────────────────────────────────────


def build_network(skeleton_data: dict) -> NetworkResult:
    """
    Build the character network from parsed screenplay data.

    This is a pure function — no graph writes, no side effects. It takes
    the skeleton JSON and returns a NetworkResult containing all allocated data.

    Args:
        skeleton_data: Parsed screenplay JSON with keys:
            character: { name, source_work, description }
            scenes: [{ scene_number, slugline, location, time_of_day,
                       characters_present, witnesses, dialogue_turns, action_summary }]

    Returns:
        NetworkResult with all allocated data ready for graph writing.
    """
    character_info = skeleton_data.get("character", {})
    scenes = skeleton_data.get("scenes", [])
    focal_character = character_info.get("name", "UNKNOWN")
    source_work = character_info.get("source_work", "Unknown")

    # Initialize result
    result = NetworkResult(
        source_work=source_work,
        focal_character=focal_character,
        total_scenes=len(scenes),
        total_turns=0,
        total_characters=0,
    )

    # ── Pass 1: Character discovery and scene role allocation ─────
    all_characters = set()
    for scene in scenes:
        scene_num = scene.get("scene_number", 0)
        present = scene.get("characters_present", [])
        witnesses = set(scene.get("witnesses", []))
        speakers = set()
        roles = {}

        # Identify speakers from dialogue
        for turn in scene.get("dialogue_turns", []):
            speakers.add(turn.get("speaker", ""))

        # Assign roles
        for char in present:
            all_characters.add(char)
            if char == focal_character:
                roles[char] = "focal"
            elif char in speakers:
                roles[char] = "present"
            elif char in witnesses:
                roles[char] = "witness"
            else:
                roles[char] = "witness"  # Present but not speaking = witness

        result.scene_roles[scene_num] = roles

    # Initialize CharacterData for all discovered characters
    for char_name in all_characters:
        result.characters[char_name] = CharacterData(name=char_name)

    result.total_characters = len(all_characters)

    # ── Pass 2: Scene appearances ─────────────────────────────────
    for scene in scenes:
        scene_num = scene.get("scene_number", 0)
        roles = result.scene_roles.get(scene_num, {})
        for char_name, role in roles.items():
            if char_name in result.characters:
                result.characters[char_name].scene_appearances.append({
                    "scene_number": scene_num,
                    "role": role,
                })

    # ── Pass 3: DIRECTED_AT inference (Option C) ──────────────────
    for scene in scenes:
        scene_num = scene.get("scene_number", 0)
        present = set(scene.get("characters_present", []))
        turns = scene.get("dialogue_turns", [])
        prev_speaker = None

        for turn in turns:
            speaker = turn.get("speaker", "")
            text = turn.get("text", "")
            parenthetical = turn.get("parenthetical") or ""
            turn_number = turn.get("turn_number", 0)

            target, method, conf = _infer_directed_at(
                speaker=speaker,
                text=text,
                parenthetical=parenthetical,
                characters_present=present,
                prev_speaker=prev_speaker,
            )

            if target:
                result.directed_at_results.append(DirectedAtResult(
                    turn_number=turn_number,
                    speaker=speaker,
                    target_character=target,
                    inference_method=method,
                    confidence=conf,
                ))

            prev_speaker = speaker

    result.total_turns = sum(
        len(scene.get("dialogue_turns", []))
        for scene in scenes
    )

    # ── Pass 4: Vocabulary allocation ─────────────────────────────
    # Build a lookup from (scene_number, turn_number) → directed_at target
    da_lookup = {}
    for da in result.directed_at_results:
        # We don't have scene_number on DirectedAtResult, so use a scene-level pass
        pass

    for scene in scenes:
        scene_num = scene.get("scene_number", 0)
        turns = scene.get("dialogue_turns", [])
        present = set(scene.get("characters_present", []))
        prev_speaker = None

        for turn in turns:
            speaker = turn.get("speaker", "")
            text = turn.get("text", "")
            parenthetical = turn.get("parenthetical") or ""
            turn_number = turn.get("turn_number", 0)

            if speaker not in result.characters:
                continue

            char_data = result.characters[speaker]
            char_data.dialogue_turn_count += 1

            # Determine who this turn is directed at
            target, _, _ = _infer_directed_at(
                speaker=speaker,
                text=text,
                parenthetical=parenthetical,
                characters_present=present,
                prev_speaker=prev_speaker,
            )

            # Tokenize and count words
            words = _tokenize(text)
            char_data.total_words += len(words)

            for word in words:
                if word in STOP_WORDS or len(word) < MIN_WORD_LENGTH:
                    continue

                if word not in char_data.vocabulary:
                    char_data.vocabulary[word] = VocabularyEntry(
                        first_observed_scene=scene_num,
                    )

                entry = char_data.vocabulary[word]
                entry.count += 1
                entry.scenes.append(scene_num)
                entry.last_observed_scene = scene_num

                if target:
                    entry.directed_at[target] = entry.directed_at.get(target, 0) + 1

            prev_speaker = speaker

        # Update unique word counts
        for char_data in result.characters.values():
            char_data.unique_words = len(char_data.vocabulary)

    # ── Pass 5: Relationship computation ──────────────────────────
    result.relationships = _compute_relationships(scenes, result.characters, focal_character)

    return result


# ──────────────────────────────────────────────────────────────────────────────
# DIRECTED_AT INFERENCE — Option C
# ──────────────────────────────────────────────────────────────────────────────


def _infer_directed_at(
    speaker: str,
    text: str,
    parenthetical: str,
    characters_present: set,
    prev_speaker: Optional[str],
) -> tuple[Optional[str], str, str]:
    """
    Infer who a dialogue turn is directed at using Option C priority.

    Priority:
    1. Parenthetical check: "(to Mary Jane)" → explicit
    2. Name match: dialogue text contains a present character's name → high
    3. Sequential fallback: previous speaker → inferred

    Args:
        speaker: Who is speaking
        text: The dialogue text
        parenthetical: Stage direction parenthetical (may be empty)
        characters_present: Set of character names in the scene
        prev_speaker: Previous turn's speaker (for sequential fallback)

    Returns:
        Tuple of (target_character, inference_method, confidence)
        Returns (None, "", "") if no target can be inferred.
    """
    others = characters_present - {speaker}
    if not others:
        return None, "", ""

    # 1. Parenthetical check
    if parenthetical:
        paren_lower = parenthetical.lower()
        for char in others:
            # Check various name forms: full name, first name
            char_parts = char.split()
            for name_part in [char.lower()] + [p.lower() for p in char_parts]:
                if name_part in paren_lower:
                    return char, "parenthetical", "explicit"

    # 2. Name match in dialogue text
    text_lower = text.lower()
    matches = []
    for char in others:
        char_parts = char.split()
        for name_part in [char.lower()] + [p.lower() for p in char_parts if len(p) > 2]:
            if name_part in text_lower:
                matches.append(char)
                break

    if len(matches) == 1:
        return matches[0], "name_match", "high"
    elif len(matches) > 1:
        # Multiple name matches — take the first one mentioned in text
        positions = {}
        for char in matches:
            char_parts = char.split()
            min_pos = len(text_lower)
            for name_part in [char.lower()] + [p.lower() for p in char_parts if len(p) > 2]:
                pos = text_lower.find(name_part)
                if pos != -1 and pos < min_pos:
                    min_pos = pos
            positions[char] = min_pos
        earliest = min(positions, key=positions.get)
        return earliest, "name_match", "high"

    # 3. Sequential fallback
    if prev_speaker and prev_speaker in others:
        return prev_speaker, "sequential", "inferred"

    # If prev_speaker is not in others (e.g., scene just started), pick first other
    if others:
        return sorted(others)[0], "sequential", "inferred"

    return None, "", ""


# ──────────────────────────────────────────────────────────────────────────────
# RELATIONSHIP COMPUTATION
# ──────────────────────────────────────────────────────────────────────────────


def _compute_relationships(
    scenes: list[dict],
    characters: dict[str, CharacterData],
    focal_character: str,
) -> list[RelationshipData]:
    """
    Compute pairwise relationship data for the focal character.

    Uses the 40/40/20 significance formula:
    - 40% co-occurrence (scenes both appear in)
    - 40% shared dialogue (scenes with direct dialogue exchange)
    - 20% witness dynamics (scenes where one witnesses the other)

    Args:
        scenes: List of scene dicts from the skeleton
        characters: Dict of CharacterData objects
        focal_character: Name of the focal character

    Returns:
        List of RelationshipData for all focal character relationships.
    """
    # Build per-pair counters
    pair_data: dict[str, RelationshipData] = {}

    for scene in scenes:
        scene_num = scene.get("scene_number", 0)
        present = set(scene.get("characters_present", []))
        witnesses = set(scene.get("witnesses", []))
        speakers = set(t.get("speaker", "") for t in scene.get("dialogue_turns", []))

        if focal_character not in present:
            continue

        for other in present:
            if other == focal_character:
                continue

            key = other  # Relationships are focal → other
            if key not in pair_data:
                pair_data[key] = RelationshipData(
                    source_character=focal_character,
                    target_character=other,
                )

            rel = pair_data[key]
            rel.co_occurrence_count += 1

            # Track observation window
            if rel.first_observed_scene is None or scene_num < rel.first_observed_scene:
                rel.first_observed_scene = scene_num
            if rel.last_observed_scene is None or scene_num > rel.last_observed_scene:
                rel.last_observed_scene = scene_num

            # Shared dialogue: both focal and other speak in this scene
            if focal_character in speakers and other in speakers:
                rel.shared_dialogue_scenes += 1

            # Witness dynamics: one is witness (present but not speaking)
            focal_is_witness = focal_character in witnesses or (focal_character in present and focal_character not in speakers)
            other_is_witness = other in witnesses or (other in present and other not in speakers)
            if focal_is_witness or other_is_witness:
                rel.witness_scenes += 1

    # Compute normalized significance scores
    relationships = list(pair_data.values())
    if relationships:
        max_co = max(r.co_occurrence_count for r in relationships)
        max_shared = max(r.shared_dialogue_scenes for r in relationships)
        max_witness = max(r.witness_scenes for r in relationships) if any(r.witness_scenes for r in relationships) else 1

        for rel in relationships:
            rel.compute_significance(max_co, max_shared, max_witness)

    # Sort by significance score descending
    relationships.sort(key=lambda r: r.significance_score, reverse=True)

    return relationships


# ──────────────────────────────────────────────────────────────────────────────
# TOKENIZATION
# ──────────────────────────────────────────────────────────────────────────────


def _tokenize(text: str) -> list[str]:
    """
    Tokenize dialogue text into lowercase words.

    Strips punctuation, handles contractions, and produces clean word tokens
    suitable for vocabulary counting.
    """
    # Lowercase and strip common punctuation
    text = text.lower()
    # Replace common contractions before tokenizing
    text = re.sub(r"['\u2019]", "'", text)  # Normalize apostrophes
    text = re.sub(r"[^a-z' ]", " ", text)   # Keep only letters, apostrophes, spaces
    words = text.split()
    # Strip leading/trailing apostrophes
    words = [w.strip("'") for w in words if w.strip("'")]
    return words


# ──────────────────────────────────────────────────────────────────────────────
# GRAPH WRITER
# ──────────────────────────────────────────────────────────────────────────────


def write_to_graph(network_result: NetworkResult, source_work_data: Optional[dict] = None):
    """
    Write a NetworkResult into Neo4j via neomodel.

    This function creates/updates graph nodes and edges from the pre-computed
    network data. It is idempotent — running it twice on the same data produces
    the same graph state.

    Args:
        network_result: Output from build_network()
        source_work_data: Optional dict with additional SourceWork metadata:
            { year, medium, writers, source_material, universe, ontology_type }

    Returns:
        dict: Summary of nodes and edges created/updated.
    """
    from . import graph_models as gm

    summary = {"nodes_created": 0, "edges_created": 0, "errors": []}

    try:
        # 1. Create or get SourceWork
        sw_data = source_work_data or {}
        source_work = _get_or_create(
            gm.SourceWork,
            title=network_result.source_work,
            defaults={
                "year": sw_data.get("year"),
                "medium": sw_data.get("medium", "film"),
                "writers": sw_data.get("writers", []),
                "source_material": sw_data.get("source_material", ""),
                "universe": sw_data.get("universe", ""),
                "ontology_type": sw_data.get("ontology_type", "fictional"),
            },
        )
        summary["nodes_created"] += 1

        # 2. Create MasterDocument
        master_doc = gm.MasterDocument(
            total_scenes=network_result.total_scenes,
            total_dialogue_turns=network_result.total_turns,
            total_characters=network_result.total_characters,
            parser_version="2.0",
        )
        master_doc.save()
        master_doc.source_work.connect(source_work)
        summary["nodes_created"] += 1

        # 3. Create CharacterProfile nodes
        char_nodes = {}
        for char_name, char_data in network_result.characters.items():
            profile = _get_or_create(
                gm.CharacterProfile,
                name=char_name,
                defaults={
                    "source_work": network_result.source_work,
                    "vocabulary_json": char_data.build_vocabulary_json(),
                },
            )
            profile.belongs_to.connect(source_work)
            char_nodes[char_name] = profile
            summary["nodes_created"] += 1

        # 4. Create Scene nodes and connect characters
        scene_nodes = {}
        for scene_data in _iter_scenes_from_result(network_result):
            scene_num = scene_data["scene_number"]
            scene = gm.Scene(
                scene_number=scene_num,
                slugline=scene_data.get("slugline", ""),
                location=scene_data.get("location", ""),
                time_of_day=scene_data.get("time_of_day", ""),
                action_summary=scene_data.get("action_summary", ""),
                narrative_order=scene_num,
                title=scene_data.get("slugline", ""),
            )
            scene.save()
            master_doc.scenes.connect(scene)
            scene_nodes[scene_num] = scene
            summary["nodes_created"] += 1

            # Connect characters to scene with roles
            roles = network_result.scene_roles.get(scene_num, {})
            for char_name, role in roles.items():
                if char_name in char_nodes:
                    char_nodes[char_name].scenes.connect(
                        scene, {"role": role}
                    )
                    summary["edges_created"] += 1

        # 5. Create RELATES_TO edges with significance data
        for rel_data in network_result.relationships:
            source_char = char_nodes.get(rel_data.source_character)
            target_char = char_nodes.get(rel_data.target_character)
            if source_char and target_char:
                source_char.relationships.connect(target_char, {
                    "co_occurrence_count": rel_data.co_occurrence_count,
                    "shared_dialogue_scenes": rel_data.shared_dialogue_scenes,
                    "witness_scenes": rel_data.witness_scenes,
                    "first_observed_scene": rel_data.first_observed_scene,
                    "last_observed_scene": rel_data.last_observed_scene,
                    "significance_score": rel_data.significance_score,
                    "observation_source": "screenplay_parse",
                    "observation_completeness": "full_work",
                })
                summary["edges_created"] += 1

        logger.info(f"Graph write complete: {summary}")

    except Exception as e:
        logger.error(f"Graph write failed: {e}")
        summary["errors"].append(str(e))

    return summary


def _get_or_create(model_class, defaults=None, **kwargs):
    """Get existing node by unique fields or create a new one."""
    try:
        existing = model_class.nodes.get(**kwargs)
        return existing
    except model_class.DoesNotExist:
        node = model_class(**kwargs, **(defaults or {}))
        node.save()
        return node


def _iter_scenes_from_result(network_result: NetworkResult):
    """
    Yield basic scene metadata from the network result.

    Note: The full scene data comes from the original skeleton JSON.
    This helper provides minimal data from what's available in the result.
    """
    # The network result tracks scene_roles, which gives us scene numbers
    # For full scene data, the caller should pass the original skeleton
    for scene_num in sorted(network_result.scene_roles.keys()):
        yield {
            "scene_number": scene_num,
            "slugline": "",
            "location": "",
            "time_of_day": "",
            "action_summary": "",
        }


def write_to_graph_with_scenes(
    network_result: NetworkResult,
    skeleton_data: dict,
    source_work_data: Optional[dict] = None,
    *,
    workspace_id: str,
    owner_user_id: int,
):
    """
    Write to graph with full scene metadata from the original skeleton.

    Uses batch Cypher queries instead of individual neomodel .save() calls
    for performance over cloud connections (AuraDB). Reduces ~500+ round-trips
    to ~8 batched queries.
    """
    from neomodel import db

    scenes = skeleton_data.get("scenes", [])
    scene_lookup = {s["scene_number"]: s for s in scenes}
    sw_data = source_work_data or {}

    summary = {"nodes_created": 0, "edges_created": 0, "errors": []}

    try:
        # 1. SourceWork + MasterDocument (2 queries)
        db.cypher_query(
            """
            MERGE (sw:SourceWork {title: $title, workspace_id: $workspace_id})
            ON CREATE SET sw.year = $year, sw.medium = $medium,
                          sw.ontology_type = $ontology_type,
                          sw.owner_user_id = $owner_user_id,
                          sw.uid = randomUUID()
            MERGE (md:MasterDocument {total_scenes: $total_scenes,
                                      total_dialogue_turns: $total_turns,
                                      total_characters: $total_chars,
                                      workspace_id: $workspace_id})
            ON CREATE SET md.parser_version = '2.0', md.owner_user_id = $owner_user_id, md.uid = randomUUID()
            MERGE (md)-[:FROM_SOURCE]->(sw)
            """,
            {
                "title": network_result.source_work,
                "workspace_id": workspace_id,
                "owner_user_id": owner_user_id,
                "year": sw_data.get("year"),
                "medium": sw_data.get("medium", "film"),
                "ontology_type": sw_data.get("ontology_type", "fictional"),
                "total_scenes": network_result.total_scenes,
                "total_turns": network_result.total_turns,
                "total_chars": network_result.total_characters,
            },
        )
        summary["nodes_created"] += 2

        # 2. Batch CharacterProfile nodes
        char_params = []
        for char_name, char_data in network_result.characters.items():
            char_params.append({
                "name": char_name,
                "source_work": network_result.source_work,
                "workspace_id": workspace_id,
                "owner_user_id": owner_user_id,
                "vocabulary_json": json.dumps(char_data.build_vocabulary_json()),
            })

        if char_params:
            db.cypher_query(
                """
                UNWIND $chars AS c
                MERGE (cp:CharacterProfile {name: c.name, source_work: c.source_work, workspace_id: c.workspace_id})
                ON CREATE SET cp.uid = randomUUID(),
                              cp.owner_user_id = c.owner_user_id,
                              cp.vocabulary_json = c.vocabulary_json
                ON MATCH SET cp.vocabulary_json = c.vocabulary_json
                WITH cp, c
                MATCH (sw:SourceWork {title: c.source_work, workspace_id: c.workspace_id})
                MERGE (cp)-[:BELONGS_TO]->(sw)
                """,
                {"chars": char_params},
            )
            summary["nodes_created"] += len(char_params)

        # 3. Batch Scene nodes
        scene_params = []
        for scene_num in sorted(network_result.scene_roles.keys()):
            s = scene_lookup.get(scene_num, {})
            scene_params.append({
                "scene_number": scene_num,
                "workspace_id": workspace_id,
                "owner_user_id": owner_user_id,
                "slugline": s.get("slugline", ""),
                "location": s.get("location", ""),
                "time_of_day": s.get("time_of_day", ""),
                "action_summary": s.get("action_summary", ""),
                "title": s.get("slugline", ""),
            })

        if scene_params:
            db.cypher_query(
                """
                UNWIND $scenes AS s
                MERGE (sc:Scene {scene_number: s.scene_number, workspace_id: s.workspace_id})
                ON CREATE SET sc.uid = randomUUID(),
                              sc.owner_user_id = s.owner_user_id,
                              sc.slugline = s.slugline,
                              sc.location = s.location,
                              sc.time_of_day = s.time_of_day,
                              sc.action_summary = s.action_summary,
                              sc.narrative_order = s.scene_number,
                              sc.title = s.title
                WITH sc
                MATCH (md:MasterDocument {workspace_id: s.workspace_id})
                MERGE (md)-[:CONTAINS_SCENE]->(sc)
                """,
                {"scenes": scene_params},
            )
            summary["nodes_created"] += len(scene_params)

        # 4. Batch APPEARS_IN edges (character → scene with role)
        appears_params = []
        for scene_num, roles in network_result.scene_roles.items():
            for char_name, role in roles.items():
                appears_params.append({
                    "char_name": char_name,
                    "source_work": network_result.source_work,
                    "workspace_id": workspace_id,
                    "scene_number": scene_num,
                    "role": role,
                })

        if appears_params:
            db.cypher_query(
                """
                UNWIND $edges AS e
                MATCH (cp:CharacterProfile {name: e.char_name, source_work: e.source_work, workspace_id: e.workspace_id})
                MATCH (sc:Scene {scene_number: e.scene_number, workspace_id: e.workspace_id})
                MERGE (cp)-[r:APPEARS_IN]->(sc)
                ON CREATE SET r.role = e.role
                """,
                {"edges": appears_params},
            )
            summary["edges_created"] += len(appears_params)

        # 5. Batch Turn nodes with SPOKEN_BY and scene membership
        turn_params = []
        for scene in scenes:
            scene_num = scene["scene_number"]
            if scene_num not in network_result.scene_roles:
                continue
            present = set(scene.get("characters_present", []))
            prev_speaker = None
            for turn in scene.get("dialogue_turns", []):
                speaker = turn.get("speaker", "")
                text = turn.get("text", "")
                paren = turn.get("parenthetical") or ""
                tn = turn.get("turn_number", 0)

                target, method, conf = _infer_directed_at(
                    speaker, text, paren, present, prev_speaker
                )

                turn_params.append({
                    "scene_number": scene_num,
                    "workspace_id": workspace_id,
                    "owner_user_id": owner_user_id,
                    "turn_number": tn,
                    "text": text,
                    "speaker_name": speaker,
                    "source_work": network_result.source_work,
                    "parenthetical": paren if paren else None,
                    "word_count": len(text.split()),
                    "directed_at": target,
                    "inference_method": method,
                    "confidence": conf,
                })
                prev_speaker = speaker

        # Batch turns in chunks of 50 to avoid huge single queries
        chunk_size = 50
        for i in range(0, len(turn_params), chunk_size):
            chunk = turn_params[i:i + chunk_size]
            db.cypher_query(
                """
                UNWIND $turns AS t
                CREATE (turn:Turn {
                    uid: randomUUID(),
                    workspace_id: t.workspace_id,
                    owner_user_id: t.owner_user_id,
                    text: t.text,
                    turn_number: t.turn_number,
                    speaker_name: t.speaker_name,
                    word_count: t.word_count
                })
                WITH turn, t
                MATCH (sc:Scene {scene_number: t.scene_number, workspace_id: t.workspace_id})
                MERGE (sc)-[:CONTAINS]->(turn)
                WITH turn, t
                MATCH (cp:CharacterProfile {name: t.speaker_name, source_work: t.source_work, workspace_id: t.workspace_id})
                MERGE (turn)-[:SPOKEN_BY]->(cp)
                """,
                {"turns": chunk},
            )
            summary["nodes_created"] += len(chunk)

        # 6. Batch DIRECTED_AT edges (separate pass for turns that have targets)
        da_params = [t for t in turn_params if t["directed_at"] and t["directed_at"] in network_result.characters]
        if da_params:
            for i in range(0, len(da_params), chunk_size):
                chunk = da_params[i:i + chunk_size]
                db.cypher_query(
                    """
                    UNWIND $edges AS e
                    MATCH (turn:Turn {turn_number: e.turn_number, speaker_name: e.speaker_name, workspace_id: e.workspace_id})
                    MATCH (sc:Scene {scene_number: e.scene_number, workspace_id: e.workspace_id})
                    WHERE (sc)-[:CONTAINS]->(turn)
                    MATCH (target:CharacterProfile {name: e.directed_at, source_work: e.source_work, workspace_id: e.workspace_id})
                    MERGE (turn)-[r:DIRECTED_AT]->(target)
                    ON CREATE SET r.inference_method = e.inference_method,
                                  r.confidence = e.confidence
                    """,
                    {"edges": chunk},
                )
                summary["edges_created"] += len(chunk)

        # 7. Batch RELATES_TO edges
        rel_params = []
        for rel_data in network_result.relationships:
            rel_params.append({
                "source": rel_data.source_character,
                "target": rel_data.target_character,
                "source_work": network_result.source_work,
                "workspace_id": workspace_id,
                "co_occurrence_count": rel_data.co_occurrence_count,
                "shared_dialogue_scenes": rel_data.shared_dialogue_scenes,
                "witness_scenes": rel_data.witness_scenes,
                "first_observed_scene": rel_data.first_observed_scene,
                "last_observed_scene": rel_data.last_observed_scene,
                "significance_score": rel_data.significance_score,
            })

        if rel_params:
            db.cypher_query(
                """
                UNWIND $rels AS r
                MATCH (src:CharacterProfile {name: r.source, source_work: r.source_work, workspace_id: r.workspace_id})
                MATCH (tgt:CharacterProfile {name: r.target, source_work: r.source_work, workspace_id: r.workspace_id})
                MERGE (src)-[rel:RELATES_TO]->(tgt)
                ON CREATE SET rel.co_occurrence_count = r.co_occurrence_count,
                              rel.shared_dialogue_scenes = r.shared_dialogue_scenes,
                              rel.witness_scenes = r.witness_scenes,
                              rel.first_observed_scene = r.first_observed_scene,
                              rel.last_observed_scene = r.last_observed_scene,
                              rel.significance_score = r.significance_score,
                              rel.observation_source = 'screenplay_parse',
                              rel.observation_completeness = 'full_work'
                """,
                {"rels": rel_params},
            )
            summary["edges_created"] += len(rel_params)

        logger.info(f"Graph write with scenes complete: {summary}")

    except Exception as e:
        logger.error(f"Graph write failed: {e}")
        summary["errors"].append(str(e))

    return summary
