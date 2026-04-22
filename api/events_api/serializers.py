from django.contrib.auth import get_user_model
from django.contrib.auth.password_validation import validate_password
from rest_framework import serializers

from .models import Event, Pattern

User = get_user_model()


class EventSerializer(serializers.ModelSerializer):
    class Meta:
        model = Event
        fields = [
            "id",
            "workspace",
            "created_by",
            "created_at",
            "occurred_at",
            "source",
            "text",
            "context_tags",
            "people",
            "emotion",
            "intensity",
            "reaction",
            "outcome",
        ]
        read_only_fields = ["id", "workspace", "created_by", "created_at"]

    def _validate_string_list(self, value, field_name):
        if value is None:
            return []
        if not isinstance(value, list):
            raise serializers.ValidationError(f"{field_name} must be a list of strings.")
        if any(not isinstance(item, str) for item in value):
            raise serializers.ValidationError(f"{field_name} must be a list of strings.")
        return value

    def validate_context_tags(self, value):
        return self._validate_string_list(value, "context_tags")

    def validate_people(self, value):
        return self._validate_string_list(value, "people")


class PatternSerializer(serializers.ModelSerializer):
    run_id = serializers.UUIDField(source="run.id", read_only=True)

    class Meta:
        model = Pattern
        fields = [
            "id",
            "run_id",
            "key",
            "name",
            "hypothesis",
            "score",
            "evidence",
        ]


class PatternRecomputeRequestSerializer(serializers.Serializer):
    max_patterns = serializers.IntegerField(required=False, min_value=1, default=7)
    evidence_per_pattern = serializers.IntegerField(required=False, min_value=1, default=5)


class AskRequestSerializer(serializers.Serializer):
    question = serializers.CharField(required=True, allow_blank=False)
    focus_event_id = serializers.UUIDField(required=False)


class SignupSerializer(serializers.Serializer):
    username = serializers.CharField(required=True, max_length=150)
    email = serializers.EmailField(required=False, allow_blank=True)
    password = serializers.CharField(required=True, write_only=True, trim_whitespace=False)

    def validate_username(self, value):
        normalized = value.strip()
        if not normalized:
            raise serializers.ValidationError("Username is required.")
        if User.objects.filter(username__iexact=normalized).exists():
            raise serializers.ValidationError("That username is already taken.")
        return normalized

    def validate_password(self, value):
        validate_password(value)
        return value

    def create(self, validated_data):
        return User.objects.create_user(
            username=validated_data["username"],
            email=validated_data.get("email", ""),
            password=validated_data["password"],
        )


class LoginSerializer(serializers.Serializer):
    username = serializers.CharField(required=True)
    password = serializers.CharField(required=True, write_only=True, trim_whitespace=False)


class LogoutSerializer(serializers.Serializer):
    refresh = serializers.CharField(required=True, allow_blank=False, trim_whitespace=True)


class PromoteSerializer(serializers.Serializer):
    """Validate data for promoting an anonymous account to a full account."""

    username = serializers.CharField(required=True, max_length=150)
    email = serializers.EmailField(required=False, allow_blank=True, default="")
    password = serializers.CharField(required=True, write_only=True, trim_whitespace=False)

    def validate_username(self, value):
        normalized = value.strip()
        if not normalized:
            raise serializers.ValidationError("Username is required.")
        if normalized.startswith("anon_"):
            raise serializers.ValidationError("Username cannot start with 'anon_'.")
        if User.objects.filter(username__iexact=normalized).exists():
            raise serializers.ValidationError("That username is already taken.")
        return normalized

    def validate_password(self, value):
        validate_password(value)
        return value
