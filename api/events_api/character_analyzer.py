"""
ThriveSight Character Analyzer — Multi-Lens Analysis Framework

This module implements the multi-lens analytical framework for extracting rich
character data from parsed screenplays. Rather than relying solely on dialogue-level
emotional dynamics, it applies FIVE analytical lenses:

LENS 1: DIALOGUE SIGNAL ANALYSIS
    What it captures: Turn-by-turn emotional dynamics (the existing pipeline)
    Maps to: Signal, TriggerAction, Pattern nodes in the graph

LENS 2: PLOT ANALYSIS
    What it captures: Narrative arc position, stakes, power dynamics, turning points
    Maps to: Scene-level metadata that enriches the conversation context

LENS 3: CHARACTER PSYCHOLOGY
    What it captures: Core motivations, fears, defense mechanisms, attachment style,
                      identity conflicts, internal vs. external self
    Maps to: CharacterProfile metadata + enriched Signal interpretation

LENS 4: RELATIONAL DYNAMICS
    What it captures: How character X relates to character Y across ALL scenes,
                      power balance shifts, trust evolution, role each person plays
    Maps to: RELATES_TO edges with rich metadata between CharacterProfile nodes

LENS 5: CONTEXTUAL METADATA
    What it captures: Who witnesses what, build-up patterns across scenes,
                      environmental stressors, what's NOT said (subtext)
    Maps to: Witness data, temporal patterns, environmental context on Conversation nodes

The framework processes a parsed screenplay (from screenplay_parser.py) and produces
a CharacterDataset — a comprehensive data structure ready to be fed into the
ThriveSight pipeline scene-by-scene, with all the enrichment metadata attached.
"""

import json
import logging
from typing import Any, Optional

from . import llm_client

logger = logging.getLogger(__name__)


# ============================================================================
# LENS DEFINITIONS — LLM PROMPTS
# ============================================================================

LENS_1_DIALOGUE_SYSTEM = """You are a behavioral signal analyst for ThriveSight.

For each dialogue scene, identify the emotional dynamics between speakers.

For EACH dialogue turn, classify:
- emotion: The primary emotion expressed (frustration, defensiveness, anger, sadness, anxiety, hurt, contempt, warmth, humor, resignation, guilt, relief, hope, resentment, vulnerability, confusion, empathy, indifference)
- intensity: 1.0-5.0 scale
- reaction: How the speaker responded to the previous turn (defended, counter_attacked, withdrew, de_escalated, acknowledged, escalated, deflected, conceded)
- trigger_category: What behavior from the OTHER person prompted this (dismissal, accusation, deflection, withdrawal, demand, questioning, validation, acknowledgment, concession, sarcasm, initiation)
- trigger_action: Brief description of the specific triggering behavior

Respond with ONLY this JSON:
{
  "scene_dynamics": {
    "dominant_emotion": "string",
    "escalation_direction": "escalating|de_escalating|stable|volatile",
    "emotional_inflection_point": null or turn_number where the dynamic shifts
  },
  "turn_signals": [
    {
      "turn_number": 1,
      "speaker": "NAME",
      "emotion": "string",
      "intensity": 1.0,
      "reaction": "string",
      "trigger_category": "string",
      "trigger_action": "string"
    }
  ]
}"""

LENS_2_PLOT_SYSTEM = """You are a narrative analyst for ThriveSight.

Given a scene from a screenplay, analyze the PLOT-LEVEL context:

1. NARRATIVE ARC POSITION: Where is this scene in the character's story arc?
   (setup, rising_action, midpoint, crisis, climax, resolution, denouement)

2. STAKES: What is at risk for the focal character in this scene? (emotional, physical, relational, identity, financial, moral)

3. POWER DYNAMICS: Who has power in this scene? How does it shift?

4. SCENE FUNCTION: What narrative purpose does this scene serve?
   (character_introduction, relationship_establishment, conflict_introduction,
    escalation, turning_point, revelation, confrontation, reconciliation,
    transformation, setup_payoff)

5. FORESHADOWING/CALLBACKS: Does this scene set up or pay off something from earlier?

Respond with ONLY this JSON:
{
  "arc_position": "string",
  "stakes": ["string"],
  "power_holder": "CHARACTER_NAME or null",
  "power_shift": null or {"from": "NAME", "to": "NAME", "trigger": "what caused it"},
  "scene_function": "string",
  "narrative_connections": [
    {"type": "foreshadows|calls_back|parallels", "target_description": "what it connects to"}
  ]
}"""

LENS_3_CHARACTER_SYSTEM = """You are a character psychologist for ThriveSight.

Given a scene and background context about a character, analyze their PSYCHOLOGY as revealed in this scene:

1. MOTIVATIONS: What is driving this character's behavior in this scene?
   - Surface motivation (what they say they want)
   - Deep motivation (what they actually need)

2. DEFENSE MECHANISMS: What psychological defenses are active?
   (denial, projection, rationalization, displacement, humor, intellectualization,
    suppression, sublimation, reaction_formation, compartmentalization)

3. IDENTITY CONFLICT: Is the character torn between two versions of themselves?
   If so, which self is dominant in this scene?

4. EMOTIONAL REGULATION: How well is the character managing their emotions?
   (regulated, dysregulated, suppressed, performative, authentic)

5. ATTACHMENT SIGNALS: What does this scene reveal about how the character
   connects with others? (secure, anxious, avoidant, disorganized)

6. UNSPOKEN NEEDS: What does the character need but not explicitly ask for?

Respond with ONLY this JSON:
{
  "surface_motivation": "string",
  "deep_motivation": "string",
  "active_defenses": ["string"],
  "identity_conflict": null or {"self_a": "string", "self_b": "string", "dominant": "a or b"},
  "emotional_regulation": "string",
  "attachment_signals": "string",
  "unspoken_needs": ["string"]
}"""

LENS_4_RELATIONAL_SYSTEM = """You are a relational dynamics analyst for ThriveSight.

Given a scene with two or more characters, analyze the RELATIONSHIP DYNAMICS:

1. RELATIONSHIP TYPE: What kind of relationship is this?
   (romantic, familial, friendship, mentorship, rivalry, authority, transactional, adversarial)

2. POWER BALANCE: Who holds relational power and why?

3. COMMUNICATION PATTERN: How do these people typically communicate?
   (direct, passive_aggressive, avoidant, explosive, manipulative,
    collaborative, performative, guarded)

4. TRUST LEVEL: How much trust exists between these characters in this scene?
   (high, moderate, fragile, broken, rebuilding, absent)

5. ROLE EACH PLAYS: What function does each character serve for the other?
   (supporter, challenger, mirror, enabler, antagonist, protector, dependent)

6. WHAT THIS SCENE CHANGES: How does this interaction alter the relationship?

Respond with ONLY this JSON:
{
  "relationships": [
    {
      "character_a": "NAME",
      "character_b": "NAME",
      "relationship_type": "string",
      "power_balance": {"holder": "NAME or balanced", "reason": "string"},
      "communication_pattern": "string",
      "trust_level": "string",
      "roles": {"character_a_role": "string", "character_b_role": "string"},
      "change": null or "string describing how this scene shifts the relationship"
    }
  ]
}"""

LENS_5_CONTEXT_SYSTEM = """You are a contextual analyst for ThriveSight.

Given a scene, its action/description text, and who is present, analyze the CONTEXTUAL METADATA:

1. WITNESSES: For characters present but NOT speaking, what do they observe?
   Why is their witnessing significant?

2. BUILD-UP CONTRIBUTION: How does this scene contribute to a larger build-up?
   What pressure is accumulating? What will eventually break?

3. ENVIRONMENTAL STRESSORS: What external conditions are affecting behavior?
   (financial pressure, time pressure, physical danger, social exposure,
    unfamiliar territory, audience/witnesses, authority presence)

4. SUBTEXT: What is being communicated WITHOUT being said?
   What do the characters understand that isn't in the dialogue?

5. BODY LANGUAGE / PHYSICAL: What physical details from the action lines
   reveal emotional state? (from screenplay directions, not dialogue)

Respond with ONLY this JSON:
{
  "witness_significance": [
    {"witness": "NAME", "observes": "what they see", "significance": "why it matters"}
  ],
  "buildup_contribution": {
    "pressure_type": "string",
    "accumulation": "what's building",
    "breaking_point_proximity": "early|building|near|at_breaking_point"
  },
  "environmental_stressors": ["string"],
  "subtext": ["string — what's communicated without words"],
  "physical_reveals": ["string — what action lines show about emotional state"]
}"""


# ============================================================================
# CHARACTER DATASET STRUCTURE
# ============================================================================

def create_character_dataset(
    character_name: str,
    source_work: str,
    description: str = "",
) -> dict[str, Any]:
    """Create an empty character dataset structure."""
    return {
        "character": {
            "name": character_name,
            "source_work": source_work,
            "description": description,
        },
        "psychology": {
            "core_motivations": [],
            "core_fears": [],
            "defense_patterns": [],
            "identity_conflicts": [],
            "attachment_style": None,
            "emotional_baseline": None,
        },
        "relationships": {},  # keyed by other character name
        "scenes": [],  # enriched scene data
        "arc": {
            "trajectory": [],  # ordered list of arc positions
            "turning_points": [],
            "transformation": None,
        },
        "buildup_threads": [],  # ongoing pressure/buildup patterns
        "analysis_metadata": {
            "total_scenes_analyzed": 0,
            "total_dialogue_turns": 0,
            "lenses_applied": [],
        },
    }


# ============================================================================
# MULTI-LENS ANALYZER
# ============================================================================

class CharacterAnalyzer:
    """
    Applies multiple analytical lenses to screenplay scenes to build
    rich character datasets.

    Usage:
        analyzer = CharacterAnalyzer(focal_character="PETER")
        dataset = analyzer.analyze_screenplay(parsed_screenplay)
    """

    def __init__(
        self,
        focal_character: str,
        source_work: str,
        description: str = "",
        lenses: Optional[list[str]] = None,
    ):
        """
        Initialize the analyzer.

        Args:
            focal_character: The primary character we're building a profile for.
            source_work: Film/show title.
            description: Brief character description.
            lenses: Which lenses to apply. Default: all five.
                    Options: "dialogue", "plot", "psychology", "relational", "context"
        """
        self.focal_character = focal_character.upper()
        self.source_work = source_work
        self.description = description
        self.lenses = lenses or ["dialogue", "plot", "psychology", "relational", "context"]
        self.dataset = create_character_dataset(focal_character, source_work, description)

    def analyze_screenplay(
        self,
        parsed_screenplay: dict[str, Any],
        batch_size: int = 5,
    ) -> dict[str, Any]:
        """
        Analyze all scenes in a parsed screenplay through all active lenses.

        Args:
            parsed_screenplay: Output from ScreenplayParser.parse()
            batch_size: Number of scenes to process per LLM call (for batching)

        Returns:
            Complete character dataset with all lens data.
        """
        scenes = parsed_screenplay.get("scenes", [])
        film_title = parsed_screenplay.get("film_title", "Unknown")

        logger.info(
            f"Analyzing {len(scenes)} scenes for {self.focal_character} "
            f"in {film_title} with lenses: {self.lenses}"
        )

        # Filter to scenes where focal character is present
        relevant_scenes = [
            s for s in scenes
            if self.focal_character in [c.upper() for c in s.get("characters_present", [])]
            or any(self.focal_character in t.get("speaker", "").upper()
                   for t in s.get("dialogue_turns", []))
        ]

        logger.info(
            f"Found {len(relevant_scenes)} scenes with {self.focal_character} "
            f"(out of {len(scenes)} total)"
        )

        # Analyze each relevant scene
        for i, scene in enumerate(relevant_scenes):
            logger.info(
                f"Analyzing scene {i+1}/{len(relevant_scenes)}: "
                f"{scene.get('slugline', 'UNKNOWN')}"
            )

            enriched_scene = self._analyze_scene(scene, i, relevant_scenes)
            self.dataset["scenes"].append(enriched_scene)

        # Post-process: aggregate psychology, relationships, arc
        self._aggregate_psychology()
        self._aggregate_relationships()
        self._compute_arc()
        self._identify_buildup_threads()

        # Update metadata
        self.dataset["analysis_metadata"]["total_scenes_analyzed"] = len(relevant_scenes)
        self.dataset["analysis_metadata"]["total_dialogue_turns"] = sum(
            len(s.get("dialogue_turns", [])) for s in relevant_scenes
        )
        self.dataset["analysis_metadata"]["lenses_applied"] = self.lenses

        return self.dataset

    def _analyze_scene(
        self,
        scene: dict[str, Any],
        scene_index: int,
        all_scenes: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Apply all active lenses to a single scene."""
        enriched = {
            "scene_number": scene.get("scene_number"),
            "slugline": scene.get("slugline"),
            "location": scene.get("location"),
            "time_of_day": scene.get("time_of_day"),
            "characters_present": scene.get("characters_present", []),
            "witnesses": scene.get("witnesses", []),
            "dialogue_turns": scene.get("dialogue_turns", []),
            "action_summary": " ".join(scene.get("action_lines", [])[:10]),  # First 10 lines
            "lens_data": {},
        }

        # Build scene text for LLM context
        scene_text = self._build_scene_text(scene)
        context_summary = self._build_context_summary(scene_index, all_scenes)

        # Apply each lens
        if "dialogue" in self.lenses and scene.get("dialogue_turns"):
            enriched["lens_data"]["dialogue"] = self._apply_dialogue_lens(scene_text)

        if "plot" in self.lenses:
            enriched["lens_data"]["plot"] = self._apply_plot_lens(scene_text, context_summary)

        if "psychology" in self.lenses and scene.get("dialogue_turns"):
            enriched["lens_data"]["psychology"] = self._apply_psychology_lens(
                scene_text, context_summary
            )

        if "relational" in self.lenses and len(scene.get("characters_present", [])) > 1:
            enriched["lens_data"]["relational"] = self._apply_relational_lens(scene_text)

        if "context" in self.lenses:
            enriched["lens_data"]["context"] = self._apply_context_lens(
                scene_text, scene.get("witnesses", []),
                scene.get("action_lines", [])
            )

        return enriched

    def _build_scene_text(self, scene: dict[str, Any]) -> str:
        """Build a readable scene text for LLM prompts."""
        parts = [f"SCENE: {scene.get('slugline', 'UNKNOWN')}\n"]

        # Include key action lines (first few, to establish setting)
        action = scene.get("action_lines", [])
        if action:
            parts.append("ACTION/DESCRIPTION:")
            for line in action[:8]:
                parts.append(f"  {line}")
            parts.append("")

        # Include dialogue
        turns = scene.get("dialogue_turns", [])
        if turns:
            parts.append("DIALOGUE:")
            for turn in turns:
                speaker = turn.get("speaker", "UNKNOWN")
                text = turn.get("text", "")
                paren = turn.get("parenthetical")
                if paren:
                    parts.append(f"  {speaker} ({paren}): {text}")
                else:
                    parts.append(f"  {speaker}: {text}")

        # Note witnesses
        witnesses = scene.get("witnesses", [])
        if witnesses:
            parts.append(f"\nCHARACTERS PRESENT BUT NOT SPEAKING: {', '.join(witnesses)}")

        return "\n".join(parts)

    def _build_context_summary(
        self, current_index: int, all_scenes: list[dict[str, Any]]
    ) -> str:
        """Build a summary of what happened in prior scenes for context."""
        if current_index == 0:
            return "This is the first scene featuring this character."

        prior_scenes = all_scenes[:current_index]
        summaries = []
        for s in prior_scenes[-5:]:  # Last 5 scenes for context
            slugline = s.get("slugline", "UNKNOWN")
            chars = ", ".join(s.get("characters_present", [])[:4])
            turn_count = len(s.get("dialogue_turns", []))
            summaries.append(f"  - {slugline} (with {chars}, {turn_count} dialogue turns)")

        return f"Previous scenes with {self.focal_character}:\n" + "\n".join(summaries)

    # ========================================================================
    # LENS APPLICATION METHODS
    # ========================================================================

    def _apply_dialogue_lens(self, scene_text: str) -> dict[str, Any]:
        """Lens 1: Dialogue signal analysis."""
        try:
            result = llm_client.analyze_with_retry(
                LENS_1_DIALOGUE_SYSTEM,
                f"Analyze the emotional dynamics in this scene:\n\n{scene_text}"
            )
            return result
        except Exception as e:
            logger.warning(f"Dialogue lens failed: {e}")
            return {"error": str(e), "scene_dynamics": None, "turn_signals": []}

    def _apply_plot_lens(self, scene_text: str, context: str) -> dict[str, Any]:
        """Lens 2: Plot/narrative analysis."""
        try:
            prompt = (
                f"Character context: {context}\n\n"
                f"Analyze the narrative function of this scene:\n\n{scene_text}"
            )
            result = llm_client.analyze_with_retry(LENS_2_PLOT_SYSTEM, prompt)
            return result
        except Exception as e:
            logger.warning(f"Plot lens failed: {e}")
            return {"error": str(e)}

    def _apply_psychology_lens(self, scene_text: str, context: str) -> dict[str, Any]:
        """Lens 3: Character psychology analysis."""
        try:
            prompt = (
                f"Focal character: {self.focal_character}\n"
                f"Background: {self.description}\n"
                f"Prior context: {context}\n\n"
                f"Analyze {self.focal_character}'s psychology in this scene:\n\n{scene_text}"
            )
            result = llm_client.analyze_with_retry(LENS_3_CHARACTER_SYSTEM, prompt)
            return result
        except Exception as e:
            logger.warning(f"Psychology lens failed: {e}")
            return {"error": str(e)}

    def _apply_relational_lens(self, scene_text: str) -> dict[str, Any]:
        """Lens 4: Relational dynamics analysis."""
        try:
            result = llm_client.analyze_with_retry(
                LENS_4_RELATIONAL_SYSTEM,
                f"Analyze the relationship dynamics in this scene:\n\n{scene_text}"
            )
            return result
        except Exception as e:
            logger.warning(f"Relational lens failed: {e}")
            return {"error": str(e), "relationships": []}

    def _apply_context_lens(
        self,
        scene_text: str,
        witnesses: list[str],
        action_lines: list[str],
    ) -> dict[str, Any]:
        """Lens 5: Contextual metadata analysis."""
        try:
            action_text = "\n".join(action_lines[:15])
            prompt = (
                f"Characters who are PRESENT but NOT SPEAKING: {', '.join(witnesses) if witnesses else 'none'}\n\n"
                f"Full action/description text:\n{action_text}\n\n"
                f"Analyze the contextual metadata:\n\n{scene_text}"
            )
            result = llm_client.analyze_with_retry(LENS_5_CONTEXT_SYSTEM, prompt)
            return result
        except Exception as e:
            logger.warning(f"Context lens failed: {e}")
            return {"error": str(e)}

    # ========================================================================
    # AGGREGATION METHODS (Post-Processing)
    # ========================================================================

    def _aggregate_psychology(self):
        """Aggregate psychology lens data across all scenes into character profile."""
        motivations = set()
        fears = set()
        defenses = {}  # defense -> count
        identity_conflicts = []
        attachment_signals = []
        regulation_patterns = []

        for scene in self.dataset["scenes"]:
            psych = scene.get("lens_data", {}).get("psychology", {})
            if not psych or "error" in psych:
                continue

            # Motivations
            deep = psych.get("deep_motivation")
            if deep:
                motivations.add(deep)

            # Defense mechanisms
            for defense in psych.get("active_defenses", []):
                defenses[defense] = defenses.get(defense, 0) + 1

            # Identity conflicts
            conflict = psych.get("identity_conflict")
            if conflict:
                identity_conflicts.append(conflict)

            # Attachment
            attachment = psych.get("attachment_signals")
            if attachment:
                attachment_signals.append(attachment)

            # Regulation
            regulation = psych.get("emotional_regulation")
            if regulation:
                regulation_patterns.append(regulation)

            # Unspoken needs → fears (inverse)
            for need in psych.get("unspoken_needs", []):
                fears.add(need)

        self.dataset["psychology"]["core_motivations"] = sorted(motivations)
        self.dataset["psychology"]["core_fears"] = sorted(fears)
        self.dataset["psychology"]["defense_patterns"] = sorted(
            defenses.keys(), key=lambda d: -defenses[d]
        )
        self.dataset["psychology"]["identity_conflicts"] = identity_conflicts
        self.dataset["psychology"]["attachment_style"] = (
            max(set(attachment_signals), key=attachment_signals.count)
            if attachment_signals else None
        )
        self.dataset["psychology"]["emotional_baseline"] = (
            max(set(regulation_patterns), key=regulation_patterns.count)
            if regulation_patterns else None
        )

    def _aggregate_relationships(self):
        """Aggregate relational lens data across all scenes."""
        relationships = {}  # keyed by (char_a, char_b) sorted tuple

        for scene in self.dataset["scenes"]:
            rel_data = scene.get("lens_data", {}).get("relational", {})
            if not rel_data or "error" in rel_data:
                continue

            for rel in rel_data.get("relationships", []):
                char_a = rel.get("character_a", "")
                char_b = rel.get("character_b", "")
                key = tuple(sorted([char_a, char_b]))

                if key not in relationships:
                    relationships[key] = {
                        "character_a": key[0],
                        "character_b": key[1],
                        "relationship_types": [],
                        "power_shifts": [],
                        "trust_trajectory": [],
                        "communication_patterns": [],
                        "scene_interactions": [],
                    }

                entry = relationships[key]
                entry["relationship_types"].append(rel.get("relationship_type"))
                entry["trust_trajectory"].append(rel.get("trust_level"))
                entry["communication_patterns"].append(rel.get("communication_pattern"))

                power = rel.get("power_balance", {})
                if power:
                    entry["power_shifts"].append(power)

                change = rel.get("change")
                if change:
                    entry["scene_interactions"].append({
                        "scene": scene.get("scene_number"),
                        "change": change,
                    })

        self.dataset["relationships"] = {
            f"{v['character_a']}-{v['character_b']}": v
            for v in relationships.values()
        }

    def _compute_arc(self):
        """Compute the character's narrative arc from plot lens data."""
        arc_positions = []
        turning_points = []

        for scene in self.dataset["scenes"]:
            plot = scene.get("lens_data", {}).get("plot", {})
            if not plot or "error" in plot:
                continue

            position = plot.get("arc_position")
            if position:
                arc_positions.append({
                    "scene_number": scene.get("scene_number"),
                    "position": position,
                    "stakes": plot.get("stakes", []),
                })

            func = plot.get("scene_function")
            if func in ("turning_point", "revelation", "confrontation", "transformation"):
                turning_points.append({
                    "scene_number": scene.get("scene_number"),
                    "function": func,
                    "slugline": scene.get("slugline"),
                })

        self.dataset["arc"]["trajectory"] = arc_positions
        self.dataset["arc"]["turning_points"] = turning_points

    def _identify_buildup_threads(self):
        """Identify build-up threads from context lens data."""
        threads = {}  # pressure_type -> list of contributions

        for scene in self.dataset["scenes"]:
            ctx = scene.get("lens_data", {}).get("context", {})
            if not ctx or "error" in ctx:
                continue

            buildup = ctx.get("buildup_contribution", {})
            if buildup:
                ptype = buildup.get("pressure_type", "unknown")
                if ptype not in threads:
                    threads[ptype] = []
                threads[ptype].append({
                    "scene_number": scene.get("scene_number"),
                    "accumulation": buildup.get("accumulation"),
                    "proximity": buildup.get("breaking_point_proximity"),
                })

        self.dataset["buildup_threads"] = [
            {"pressure_type": ptype, "progression": entries}
            for ptype, entries in threads.items()
        ]


# ============================================================================
# CONVENIENCE FUNCTION: FULL PIPELINE
# ============================================================================

def build_character_dataset(
    raw_screenplay: str,
    film_title: str,
    focal_character: str,
    character_description: str = "",
    known_characters: Optional[list[str]] = None,
    lenses: Optional[list[str]] = None,
) -> dict[str, Any]:
    """
    Full pipeline: raw screenplay text → rich character dataset.

    Args:
        raw_screenplay: Raw screenplay text
        film_title: Title of the film
        focal_character: Character to build profile for
        character_description: Brief description of the character
        known_characters: Optional list of known character names
        lenses: Which lenses to apply (default: all)

    Returns:
        Complete character dataset dict
    """
    from .screenplay_parser import ScreenplayParser

    # Step 1: Parse the screenplay
    parser = ScreenplayParser(known_characters=known_characters)
    parsed = parser.parse(raw_screenplay, film_title=film_title)

    logger.info(
        f"Parsed {parsed['total_scenes']} scenes, "
        f"{parsed['total_dialogue_turns']} dialogue turns, "
        f"{parsed['parse_metadata']['detected_characters']} characters"
    )

    # Step 2: Analyze through all lenses
    analyzer = CharacterAnalyzer(
        focal_character=focal_character,
        source_work=film_title,
        description=character_description,
        lenses=lenses,
    )
    dataset = analyzer.analyze_screenplay(parsed)

    return dataset


def scenes_to_pipeline_input(
    dataset: dict[str, Any],
    scene_numbers: Optional[list[int]] = None,
) -> list[dict[str, Any]]:
    """
    Convert character dataset scenes into inputs for the existing
    ThriveSight analysis pipeline (conversation.analyze_conversation).

    This bridges the screenplay parser output to the existing pipeline,
    allowing us to reuse all six analysis stages.

    Args:
        dataset: Output from build_character_dataset or CharacterAnalyzer
        scene_numbers: Optional list of specific scene numbers to convert.
                      Default: all scenes with dialogue.

    Returns:
        List of dicts, each suitable for analyze_conversation() as raw_input.
        Each dict includes:
        - raw_input: Formatted dialogue text
        - metadata: Scene enrichment data from lenses
        - scene_number: For ordering
        - narrative_order: For graph chronology
    """
    results = []

    for scene in dataset.get("scenes", []):
        sn = scene.get("scene_number")
        if scene_numbers and sn not in scene_numbers:
            continue

        turns = scene.get("dialogue_turns", [])
        if not turns:
            continue

        # Format as "Speaker: text" for the ConversationParser
        lines = []
        for turn in turns:
            speaker = turn.get("speaker", "UNKNOWN")
            text = turn.get("text", "")
            lines.append(f"{speaker}: {text}")

        raw_input = "\n".join(lines)

        results.append({
            "raw_input": raw_input,
            "metadata": {
                "slugline": scene.get("slugline"),
                "location": scene.get("location"),
                "characters_present": scene.get("characters_present"),
                "witnesses": scene.get("witnesses"),
                "lens_data": scene.get("lens_data", {}),
            },
            "scene_number": sn,
            "narrative_order": sn,  # scenes are already in screenplay order
        })

    return results
