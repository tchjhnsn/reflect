"""
Serializers for Journey state persistence API.

These validate incoming journey data (phases, soul orderings, provocation
responses) before writing to Neo4j.
"""

from rest_framework import serializers


VALID_JOURNEY_PHASES = [
    "three-plane-intro", "plane-drag-drop", "materialist-fork",
    "soul-onboarding", "transcendental-sub-placement",
    "path-selection", "philosopher-mode", "sovereign-mode",
    "chariot", "sovereign-end-statement", "soul-ordering",
    "socratic-chariot-reveal", "provocation", "provocation-reflecting",
    "values-intro", "value-ordering", "socratic-tier-reveal",
    "tensions-preview", "act-i-complete",
]

VALID_PATH_IDS = ["wanderer", "sovereign", "philosopher"]
VALID_PHILOSOPHER_MODES = ["guided", "socratic"]
VALID_SOVEREIGN_MODES = ["advised", "self-advised"]
VALID_SOUL_PARTS = ["reason", "spirit", "appetite"]
VALID_VALUE_IDS = [
    "liberty", "order", "justice", "dignity", "prosperity", "equality",
    "solidarity", "sovereignty", "authority", "pluralism", "merit", "stewardship",
]
VALID_SOUL_ORDERING_TYPES = ["one-rules", "co-rulers", "equal"]


class JourneyStateSerializer(serializers.Serializer):
    """Validates partial journey state updates."""

    journey_phase = serializers.ChoiceField(
        choices=[(p, p) for p in VALID_JOURNEY_PHASES],
        required=False,
    )
    path_id = serializers.ChoiceField(
        choices=[(p, p) for p in VALID_PATH_IDS],
        required=False,
        allow_null=True,
    )
    philosopher_mode = serializers.ChoiceField(
        choices=[(m, m) for m in VALID_PHILOSOPHER_MODES],
        required=False,
        allow_null=True,
    )
    sovereign_mode = serializers.ChoiceField(
        choices=[(m, m) for m in VALID_SOVEREIGN_MODES],
        required=False,
        allow_null=True,
    )
    sovereign_end_statement = serializers.CharField(
        required=False,
        allow_null=True,
        allow_blank=True,
        max_length=2000,
    )
    soul_ordering = serializers.DictField(required=False, allow_null=True)
    value_ordering = serializers.ListField(
        child=serializers.ChoiceField(choices=[(v, v) for v in VALID_VALUE_IDS]),
        required=False,
        allow_null=True,
    )
    current_provocation_index = serializers.IntegerField(
        required=False,
        min_value=0,
    )
    socratic_chariot_revealed = serializers.BooleanField(required=False)
    socratic_tier_revealed = serializers.BooleanField(required=False)

    def validate_soul_ordering(self, value):
        if value is None:
            return value

        ordering_type = value.get("type")
        if ordering_type not in VALID_SOUL_ORDERING_TYPES:
            raise serializers.ValidationError(
                f"soul_ordering.type must be one of: {VALID_SOUL_ORDERING_TYPES}"
            )

        if ordering_type == "one-rules":
            for field in ("ruler", "second", "third"):
                if value.get(field) not in VALID_SOUL_PARTS:
                    raise serializers.ValidationError(
                        f"soul_ordering.{field} must be one of: {VALID_SOUL_PARTS}"
                    )

        elif ordering_type == "co-rulers":
            rulers = value.get("rulers", [])
            if not isinstance(rulers, list) or len(rulers) != 2:
                raise serializers.ValidationError("soul_ordering.rulers must be a list of 2 soul parts")
            for part in rulers:
                if part not in VALID_SOUL_PARTS:
                    raise serializers.ValidationError(
                        f"soul_ordering.rulers contains invalid soul part: {part}"
                    )
            subordinate = value.get("subordinate")
            if subordinate not in VALID_SOUL_PARTS:
                raise serializers.ValidationError(
                    f"soul_ordering.subordinate must be one of: {VALID_SOUL_PARTS}"
                )

        return value


class ProvocationResponseSerializer(serializers.Serializer):
    """Validates a single provocation response submission."""

    provocationId = serializers.CharField(required=True, max_length=200)
    choiceId = serializers.CharField(required=True, max_length=200)
    servedSoulPart = serializers.ChoiceField(
        choices=[(p, p) for p in VALID_SOUL_PARTS],
        required=True,
    )
    protectedValues = serializers.ListField(
        child=serializers.ChoiceField(choices=[(v, v) for v in VALID_VALUE_IDS]),
        required=False,
        default=list,
    )
    sacrificedValues = serializers.ListField(
        child=serializers.ChoiceField(choices=[(v, v) for v in VALID_VALUE_IDS]),
        required=False,
        default=list,
    )
    deliberationTimeMs = serializers.IntegerField(required=False, default=0, min_value=0)
    wasInstinctive = serializers.BooleanField(required=False, default=False)
