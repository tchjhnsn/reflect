"""
ThriveSight Safety Awareness Module — Coercive control pattern detection.

This module scans analysis results for patterns that may indicate coercive control,
emotional abuse, or situations where professional support would be beneficial.

ThriveSight is NOT a diagnostic tool. This module surfaces awareness, not diagnoses.
It flags dynamics that a counseling professional would want to explore further.

Detection categories:
1. INTENSITY_ESCALATION: Rapid, severe intensity spikes (4.0+ with delta > 2.0)
2. POWER_IMBALANCE: One speaker consistently triggers high-intensity responses in the other
3. ISOLATION_LANGUAGE: Keywords associated with controlling behavior
4. REPEATED_DISMISSAL: Systematic dismissal of one person's concerns
5. THREAT_INDICATORS: Language patterns associated with threats or ultimatums

When ANY flag is raised, the module attaches a safety_awareness object to the response
with: flag type, severity (low/medium/high), and professional resources.
"""

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Keywords and phrases associated with coercive control dynamics
ISOLATION_KEYWORDS = [
    "not allowed", "can't see", "don't talk to", "my permission",
    "i forbid", "you're not going", "stay home", "who were you with",
    "check your phone", "give me your password", "you can't leave",
    "no one else", "only i", "cut off", "not your friends",
]

THREAT_KEYWORDS = [
    "or else", "you'll regret", "i'll make sure", "you'll be sorry",
    "watch what happens", "don't test me", "last warning",
    "i'll take", "you won't see", "consequences",
]

PROFESSIONAL_RESOURCES = {
    "national_hotline": {
        "name": "National Domestic Violence Hotline",
        "contact": "1-800-799-7233",
        "url": "https://www.thehotline.org",
        "description": "24/7 confidential support for anyone experiencing domestic violence.",
    },
    "crisis_text": {
        "name": "Crisis Text Line",
        "contact": "Text HOME to 741741",
        "url": "https://www.crisistextline.org",
        "description": "Free 24/7 crisis support via text message.",
    },
    "counseling_locator": {
        "name": "SAMHSA Treatment Locator",
        "contact": "1-800-662-4357",
        "url": "https://findtreatment.gov",
        "description": "Find counseling and treatment services near you.",
    },
}

# Framing language — ThriveSight is an awareness tool, not a diagnostic
AWARENESS_FRAMING = (
    "ThriveSight is a counseling awareness tool, not a diagnostic instrument. "
    "The patterns detected in this conversation may benefit from professional exploration. "
    "If any of these dynamics feel familiar, consider reaching out to a qualified counselor "
    "or one of the resources below."
)


class SafetyAwareness:
    """
    Scans analysis results for patterns that may indicate dynamics
    benefiting from professional support.

    This is post-processing — it runs after the full pipeline completes
    and attaches a safety_awareness field to the response.
    """

    def __init__(self):
        self.flags = []
        self.isolation_pattern = re.compile(
            "|".join(re.escape(kw) for kw in ISOLATION_KEYWORDS),
            re.IGNORECASE,
        )
        self.threat_pattern = re.compile(
            "|".join(re.escape(kw) for kw in THREAT_KEYWORDS),
            re.IGNORECASE,
        )

    def analyze(
        self,
        signals: list[dict[str, Any]],
        patterns: list[dict[str, Any]],
        conversation: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """
        Run all safety checks on analysis results.

        Args:
            signals: Signal Schema objects from pipeline
            patterns: Pattern Schema objects from pattern engine
            conversation: Conversation metadata (optional)

        Returns:
            safety_awareness dict if flags raised, None if no concerns
        """
        self.flags = []

        self._check_intensity_escalation(signals)
        self._check_power_imbalance(signals)
        self._check_isolation_language(signals)
        self._check_repeated_dismissal(signals, patterns)
        self._check_threat_indicators(signals)

        if not self.flags:
            return None

        # Determine overall severity
        severities = [f["severity"] for f in self.flags]
        if "high" in severities:
            overall_severity = "high"
        elif "medium" in severities:
            overall_severity = "medium"
        else:
            overall_severity = "low"

        return {
            "flags_detected": True,
            "overall_severity": overall_severity,
            "flags": self.flags,
            "framing": AWARENESS_FRAMING,
            "resources": list(PROFESSIONAL_RESOURCES.values()),
        }

    def _check_intensity_escalation(self, signals: list[dict[str, Any]]) -> None:
        """Flag rapid intensity spikes (4.0+ with delta > 2.0 from previous turn)."""
        by_speaker = {}
        for s in signals:
            speaker = s.get("speaker", "unknown")
            by_speaker.setdefault(speaker, []).append(s)

        for speaker, speaker_signals in by_speaker.items():
            sorted_signals = sorted(speaker_signals, key=lambda x: x.get("turn_number", 0))
            for i in range(1, len(sorted_signals)):
                curr_intensity = sorted_signals[i].get("intensity", 0)
                prev_intensity = sorted_signals[i - 1].get("intensity", 0)
                delta = curr_intensity - prev_intensity

                if curr_intensity >= 4.0 and delta >= 2.0:
                    self.flags.append({
                        "type": "INTENSITY_ESCALATION",
                        "severity": "medium",
                        "description": (
                            f"{speaker}'s emotional intensity spiked rapidly "
                            f"(from {prev_intensity} to {curr_intensity} at turn "
                            f"{sorted_signals[i].get('turn_number', '?')}). "
                            f"Rapid escalation patterns may benefit from professional support."
                        ),
                        "turn_number": sorted_signals[i].get("turn_number"),
                    })

    def _check_power_imbalance(self, signals: list[dict[str, Any]]) -> None:
        """Flag when one speaker consistently triggers high-intensity responses."""
        by_speaker = {}
        for s in signals:
            speaker = s.get("speaker", "unknown")
            by_speaker.setdefault(speaker, []).append(s)

        if len(by_speaker) != 2:
            return

        speakers = list(by_speaker.keys())
        avg_intensities = {}
        for sp in speakers:
            intensities = [s.get("intensity", 0) for s in by_speaker[sp]]
            avg_intensities[sp] = sum(intensities) / len(intensities) if intensities else 0

        # Check if one person's average is significantly higher
        diff = abs(avg_intensities[speakers[0]] - avg_intensities[speakers[1]])
        if diff >= 1.5:
            higher = speakers[0] if avg_intensities[speakers[0]] > avg_intensities[speakers[1]] else speakers[1]
            lower = speakers[1] if higher == speakers[0] else speakers[0]

            self.flags.append({
                "type": "POWER_IMBALANCE",
                "severity": "medium",
                "description": (
                    f"There is a notable emotional intensity imbalance: {higher} "
                    f"(avg {avg_intensities[higher]:.1f}) consistently shows higher emotional "
                    f"intensity than {lower} (avg {avg_intensities[lower]:.1f}). "
                    f"This asymmetry may indicate dynamics worth exploring with a counselor."
                ),
            })

    def _check_isolation_language(self, signals: list[dict[str, Any]]) -> None:
        """Flag language associated with controlling or isolating behavior."""
        for s in signals:
            text = s.get("text", "") or ""
            # Also check trigger_action text if present
            trigger_text = ""
            trigger_action = s.get("trigger_action", {})
            if isinstance(trigger_action, dict):
                trigger_text = trigger_action.get("action_text", "") or ""

            combined = f"{text} {trigger_text}"
            matches = self.isolation_pattern.findall(combined)

            if matches:
                self.flags.append({
                    "type": "ISOLATION_LANGUAGE",
                    "severity": "high",
                    "description": (
                        f"Language associated with controlling behavior was detected "
                        f"at turn {s.get('turn_number', '?')}: patterns like "
                        f"'{matches[0]}' may indicate dynamics that benefit from "
                        f"professional counseling support."
                    ),
                    "turn_number": s.get("turn_number"),
                })

    def _check_repeated_dismissal(
        self, signals: list[dict[str, Any]], patterns: list[dict[str, Any]]
    ) -> None:
        """Flag systematic dismissal patterns with high occurrence counts."""
        for p in patterns:
            category = p.get("trigger_category", "").lower()
            count = p.get("occurrence_count", 0)

            if category == "dismissal" and count >= 3:
                self.flags.append({
                    "type": "REPEATED_DISMISSAL",
                    "severity": "medium" if count < 5 else "high",
                    "description": (
                        f"A pattern of repeated dismissal was detected ({count} occurrences). "
                        f"Systematic dismissal of one person's concerns can be emotionally "
                        f"significant and may benefit from professional exploration."
                    ),
                })

    def _check_threat_indicators(self, signals: list[dict[str, Any]]) -> None:
        """Flag language associated with threats or ultimatums."""
        for s in signals:
            text = s.get("text", "") or ""
            trigger_text = ""
            trigger_action = s.get("trigger_action", {})
            if isinstance(trigger_action, dict):
                trigger_text = trigger_action.get("action_text", "") or ""

            combined = f"{text} {trigger_text}"
            matches = self.threat_pattern.findall(combined)

            if matches:
                self.flags.append({
                    "type": "THREAT_INDICATORS",
                    "severity": "high",
                    "description": (
                        f"Language associated with threats or ultimatums was detected "
                        f"at turn {s.get('turn_number', '?')}. These dynamics are important "
                        f"to explore with a qualified professional."
                    ),
                    "turn_number": s.get("turn_number"),
                })
