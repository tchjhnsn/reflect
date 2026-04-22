"""
ThriveSight Trajectory Computer — Deterministic emotion intensity analysis.

This module computes emotional intensity trajectories for each speaker in a conversation
and detects inflection points (significant emotional shifts) across the interaction.

The TrajectoryComputer is fully deterministic and requires no LLM inference.
It uses the Signal Schema as input and produces the Trajectory Schema as output.

Key concepts:
- data_points: intensity values ordered by turn_number for each speaker
- inflection_points: detected emotional shifts exceeding the configured threshold
- direction: "escalation" or "de_escalation" indicating shift direction
- cause: the trigger action from the preceding turn that may have caused the shift
"""

import logging
from typing import Any, Optional

from . import validators

logger = logging.getLogger(__name__)

# Default threshold for detecting inflection points (intensity delta)
DEFAULT_INFLECTION_THRESHOLD = 1.5


class TrajectoryComputer:
    """
    Computes emotional intensity trajectories and inflection points from signals.

    This is a fully deterministic stage that:
    1. Groups signals by speaker
    2. Builds per-speaker intensity arrays ordered by turn_number
    3. Detects inflection points where intensity delta exceeds threshold
    4. Annotates each inflection point with: turn_number, intensity_delta, cause, direction

    Input: List of Signal Schema objects
    Output: Trajectory Schema dict
    """

    def __init__(self, inflection_threshold: float = DEFAULT_INFLECTION_THRESHOLD):
        """
        Initialize the trajectory computer.

        Args:
            inflection_threshold: Minimum intensity delta to register as an inflection point.
                                 Default: 1.5 intensity points.
        """
        self.inflection_threshold = inflection_threshold

    def process(self, signals: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Process signals into a trajectory object.

        Args:
            signals: List of Signal Schema objects with keys:
                    - turn_number: int
                    - speaker: str
                    - emotion: str
                    - intensity: float (1.0-5.0)
                    - trigger_action: dict with 'action_text' and 'category'

        Returns:
            Trajectory Schema dict with structure:
            {
              "speakers": {
                "Speaker1": {
                  "data_points": [...],
                  "inflection_points": [...]
                },
                "Speaker2": {...}
              }
            }
        """
        if not signals:
            logger.warning("No signals to process into trajectory")
            return {"speakers": {}}

        # Step 1: Group signals by speaker
        speaker_signals = self._group_by_speaker(signals)

        # Step 2: Build per-speaker data points and detect inflection points
        trajectory = {"speakers": {}}

        for speaker, speaker_signals_list in speaker_signals.items():
            # Sort by turn number
            sorted_signals = sorted(
                speaker_signals_list, key=lambda s: s.get("turn_number", 0)
            )

            # Build data points
            data_points = self._build_data_points(sorted_signals)

            # Detect inflection points
            inflection_points = self._detect_inflection_points(sorted_signals, data_points)

            trajectory["speakers"][speaker] = {
                "data_points": data_points,
                "inflection_points": inflection_points,
            }

        # Validate output against schema
        validation_errors = self._validate_trajectory(trajectory)
        if validation_errors:
            logger.warning(f"Trajectory validation warnings: {validation_errors}")

        return trajectory

    def _group_by_speaker(
        self, signals: list[dict[str, Any]]
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Group signals by speaker.

        Args:
            signals: List of signal objects

        Returns:
            Dict mapping speaker name to list of their signals
        """
        grouped = {}
        for signal in signals:
            speaker = signal.get("speaker", "Unknown")
            if speaker not in grouped:
                grouped[speaker] = []
            grouped[speaker].append(signal)

        return grouped

    def _build_data_points(self, sorted_signals: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Build the data_points array for a speaker.

        Each data point contains:
        - turn_number: int
        - intensity: float (1.0-5.0)
        - emotion: str

        Args:
            sorted_signals: Signals for one speaker, sorted by turn_number

        Returns:
            List of data point dicts
        """
        data_points = []

        for signal in sorted_signals:
            data_point = {
                "turn_number": signal.get("turn_number"),
                "intensity": signal.get("intensity"),
                "emotion": signal.get("emotion"),
            }
            data_points.append(data_point)

        return data_points

    def _detect_inflection_points(
        self,
        sorted_signals: list[dict[str, Any]],
        data_points: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Detect inflection points where intensity delta exceeds threshold.

        Compares consecutive data points. If the absolute intensity change
        exceeds the threshold, record an inflection point with:
        - turn_number: the turn where the shift is detected
        - intensity_delta: the change in intensity
        - cause: the trigger action from the preceding signal
        - direction: "escalation" (delta > 0) or "de_escalation" (delta < 0)

        Args:
            sorted_signals: Signals sorted by turn_number with trigger_action metadata
            data_points: Already-built data points array

        Returns:
            List of inflection point dicts
        """
        inflection_points = []

        # Need at least 2 data points to detect a change
        if len(data_points) < 2:
            return inflection_points

        for i in range(1, len(data_points)):
            current_intensity = data_points[i].get("intensity", 0)
            previous_intensity = data_points[i - 1].get("intensity", 0)

            intensity_delta = current_intensity - previous_intensity

            # Check if delta exceeds threshold
            if abs(intensity_delta) >= self.inflection_threshold:
                # Determine direction
                if intensity_delta > 0:
                    direction = "escalation"
                else:
                    direction = "de_escalation"

                # Get the trigger action from the preceding signal
                preceding_signal = sorted_signals[i - 1]
                trigger_action = preceding_signal.get("trigger_action", {})
                cause = trigger_action.get("action_text", "unspecified")

                inflection_point = {
                    "turn_number": data_points[i].get("turn_number"),
                    "intensity_delta": round(intensity_delta, 2),
                    "cause": cause,
                    "direction": direction,
                }

                inflection_points.append(inflection_point)

        return inflection_points

    def _validate_trajectory(self, trajectory: dict[str, Any]) -> list[str]:
        """
        Validate trajectory against schema.

        Args:
            trajectory: Computed trajectory object

        Returns:
            List of validation error messages (empty if valid)
        """
        try:
            errors = validators.validate_trajectory(trajectory)
            return errors
        except Exception as e:
            logger.error(f"Trajectory validation raised exception: {e}")
            return [f"Validation exception: {str(e)}"]
