"""
ThriveSight Reframe Generator — LLM-powered pattern externalization.

This module generates plain-language reframes that externalize conversational conflicts,
helping people understand the dynamic as a thing external to themselves rather than
as a personal failing.

The ReframeGenerator uses LLM inference with strict constraints defined in the
Reframe Schema. If LLM fails, it falls back to a deterministic template-based approach.

Key principles (non-negotiable):
1. EXTERNALIZE: Describe the dynamic, not the people
2. NO BLAME: Neither person is at fault; the pattern is the problem
3. PLAIN LANGUAGE: No clinical jargon
4. EVIDENCE-BASED: Reference actual patterns detected
5. INTERVENTION POINT: Identify where in the cycle a different response could break it
"""

import logging
from typing import Any, Optional

from . import llm_client
from . import validators

logger = logging.getLogger(__name__)


class ReframeGenerator:
    """
    Generates externalized, plain-language reframes from detected patterns.

    Workflow:
    1. Take top-scoring patterns from pattern engine
    2. Call LLM with REFRAME_SYSTEM prompt
    3. Validate output against Reframe Schema
    4. On failure, apply deterministic fallback
    5. Return Reframe Schema object

    The output must:
    - Reference at least one pattern by name
    - NOT assign blame to either person
    - Externalize the conflict
    - Identify a specific intervention point
    """

    def __init__(self):
        """Initialize reframe generator."""
        self.max_patterns_to_reference = 3
        self.fallback_applied = False

    def process(self, patterns: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Generate a reframe from detected patterns.

        Args:
            patterns: List of Pattern Schema objects with keys:
                     - pattern_name: str
                     - hypothesis: str
                     - score: float
                     - trigger_category: str
                     - response_emotion: str
                     - occurrence_count: int
                     - evidence: list[dict]

        Returns:
            Reframe Schema dict with keys:
            - text: str (the reframe)
            - patterns_referenced: list[str]
            - resolution_elements: dict with externalization, accumulation, intervention_point
        """
        if not patterns:
            logger.warning("No patterns provided; generating fallback reframe")
            return self._generate_deterministic_fallback([])

        # Sort patterns by score (highest first)
        sorted_patterns = sorted(
            patterns, key=lambda p: p.get("score", 0), reverse=True
        )

        # Select top patterns (limit to max_patterns_to_reference)
        selected_patterns = sorted_patterns[: self.max_patterns_to_reference]

        # Try LLM-based generation
        try:
            reframe = self._generate_llm_reframe(selected_patterns)
            self.fallback_applied = False
            return reframe
        except Exception as e:
            logger.warning(f"LLM reframe generation failed ({type(e).__name__}: {e}); using fallback")
            self.fallback_applied = True
            return self._generate_deterministic_fallback(selected_patterns)

    def _generate_llm_reframe(self, patterns: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Call LLM to generate reframe.

        Args:
            patterns: Top-scored patterns to reframe

        Returns:
            Reframe Schema dict

        Raises:
            ValueError: If LLM fails or output is invalid
        """
        # Build prompt
        system_prompt, user_prompt = llm_client.build_reframe_prompt(patterns)

        # Call LLM with retry
        try:
            raw_response = llm_client.analyze_with_retry(system_prompt, user_prompt)
        except ValueError as e:
            raise ValueError(f"LLM API failed: {str(e)}")

        # Ensure response is a dict (not a list)
        if isinstance(raw_response, list):
            if len(raw_response) > 0:
                raw_response = raw_response[0]
            else:
                raise ValueError("LLM returned empty list")

        # Extract required fields
        reframe_text = raw_response.get("text")
        patterns_referenced = raw_response.get("patterns_referenced", [])
        resolution_elements = raw_response.get("resolution_elements", {})

        if not reframe_text:
            raise ValueError("LLM response missing 'text' field")

        # Construct reframe object
        reframe = {
            "text": reframe_text,
            "patterns_referenced": patterns_referenced,
            "resolution_elements": {
                "externalization": resolution_elements.get(
                    "externalization", "The dynamic creates a cycle..."
                ),
                "accumulation": resolution_elements.get(
                    "accumulation", "Over time, this pattern has reinforced itself..."
                ),
                "intervention_point": resolution_elements.get(
                    "intervention_point", "A different response at any point could shift the trajectory."
                ),
            },
        }

        # Validate
        validation_errors = validators.validate_reframe(reframe)
        if validation_errors:
            logger.warning(f"Reframe validation errors: {validation_errors}")
            raise ValueError(f"Reframe validation failed: {validation_errors}")

        return reframe

    def _generate_deterministic_fallback(self, patterns: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Generate a template-based reframe when LLM fails.

        Uses pattern names and hypotheses to construct a plain-language fallback.

        Args:
            patterns: Patterns to base fallback on (may be empty)

        Returns:
            Reframe Schema dict
        """
        if not patterns:
            # No patterns at all; generic fallback
            reframe_text = (
                "This conversation contains patterns that cycle back on themselves. "
                "Both participants are responding naturally to the dynamics, but the cycle "
                "can feel stuck. By understanding the pattern as something external rather than "
                "a personal failure, you open space for a different response at any point. "
                "When one person shifts how they respond, even slightly, the entire cycle can shift."
            )
            patterns_referenced = []
        else:
            # Build fallback from pattern names
            pattern_names = [p.get("pattern_name", "an unnamed pattern") for p in patterns]
            names_text = ", ".join(pattern_names[:-1]) + (
                f", and {pattern_names[-1]}" if len(pattern_names) > 1 else ""
            )

            reframe_text = (
                f"This conversation contains a pattern: {names_text}. "
                f"Neither person is causing this dynamic; instead, both are responding naturally to "
                f"the cycle. The pattern has built up over time through repeated interactions, each "
                f"reinforcing the other. By seeing this as an external dynamic rather than a personal "
                f"failing, you can identify where a different response—even a small one—could break "
                f"the cycle. The intervention point is whenever someone in the pattern chooses to respond differently."
            )
            patterns_referenced = pattern_names

        reframe = {
            "text": reframe_text,
            "patterns_referenced": patterns_referenced,
            "resolution_elements": {
                "externalization": "The dynamic is an external cycle, not a personal failing.",
                "accumulation": "This pattern has built up through repeated interactions, each reinforcing the other.",
                "intervention_point": "Any point in the cycle where someone responds differently could shift the trajectory.",
            },
        }

        # Validate fallback
        validation_errors = validators.validate_reframe(reframe)
        if validation_errors:
            logger.warning(f"Fallback reframe still has validation warnings: {validation_errors}")

        return reframe

    def did_fallback_apply(self) -> bool:
        """
        Check if the last reframe used the fallback approach.

        Returns:
            True if fallback was used, False if LLM succeeded
        """
        return self.fallback_applied
