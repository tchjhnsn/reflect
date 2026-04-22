"""
Journey state persistence API.

Endpoints for reading and writing the user's civic discovery journey
progress (Act I phases, path selection, soul ordering, value ordering,
provocation responses). All data is stored in the Neo4j UserProfile node
and linked ProvocationResponseNode nodes.

These endpoints are workspace-scoped and support anonymous users.
"""

import logging

from rest_framework import status
from rest_framework.response import Response

from .graph_sync import (
    create_provocation_response_in_graph,
    get_journey_state_from_graph,
    list_provocation_responses_from_graph,
    update_journey_state_in_graph,
)
from .journey_scoring import compute_soul_profile, compute_value_profile
from .journey_serializers import JourneyStateSerializer, ProvocationResponseSerializer
from .workspaces import WorkspaceAPIView

logger = logging.getLogger(__name__)


def _normalize_neo4j_value(value):
    """Convert Neo4j native types to JSON-serializable Python types."""
    if value is None:
        return None
    # neo4j.time.DateTime → ISO string
    iso = getattr(value, "iso_format", None)
    if callable(iso):
        return iso()
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        return isoformat()
    return value


def _format_journey_state(raw: dict | None) -> dict:
    """Format a raw Neo4j journey state dict into the API response shape."""
    if raw is None:
        return {
            "journeyPhase": None,
            "pathId": None,
            "philosopherMode": None,
            "sovereignMode": None,
            "sovereignEndStatement": None,
            "soulOrdering": None,
            "valueOrdering": None,
            "currentProvocationIndex": 0,
            "socraticChariotRevealed": False,
            "socraticTierRevealed": False,
        }

    return {
        "journeyPhase": raw.get("journey_phase"),
        "pathId": raw.get("path_id"),
        "philosopherMode": raw.get("philosopher_mode"),
        "sovereignMode": raw.get("sovereign_mode"),
        "sovereignEndStatement": raw.get("sovereign_end_statement"),
        "soulOrdering": raw.get("soul_ordering"),
        "valueOrdering": raw.get("value_ordering"),
        "currentProvocationIndex": raw.get("current_provocation_index") or 0,
        "socraticChariotRevealed": raw.get("socratic_chariot_revealed") or False,
        "socraticTierRevealed": raw.get("socratic_tier_revealed") or False,
    }


def _format_provocation_response(raw: dict) -> dict:
    """Format a raw Neo4j provocation response dict into the API response shape."""
    return {
        "provocationId": raw.get("provocation_id", ""),
        "choiceId": raw.get("choice_id", ""),
        "servedSoulPart": raw.get("served_soul_part", ""),
        "protectedValues": raw.get("protected_values") or [],
        "sacrificedValues": raw.get("sacrificed_values") or [],
        "deliberationTimeMs": raw.get("deliberation_time_ms") or 0,
        "wasInstinctive": raw.get("was_instinctive") or False,
        "timestamp": _normalize_neo4j_value(raw.get("timestamp")),
    }


class JourneyStateView(WorkspaceAPIView):
    """
    GET  /api/journey/state/ — Read current journey state
    POST /api/journey/state/ — Update journey state (partial updates)
    """

    def get(self, request):
        workspace = self.get_workspace()
        state = get_journey_state_from_graph(
            workspace_id=str(workspace.id),
            owner_user_id=request.user.id,
        )
        return Response(_format_journey_state(state))

    def post(self, request):
        serializer = JourneyStateSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        workspace = self.get_workspace()
        updated = update_journey_state_in_graph(
            workspace_id=str(workspace.id),
            owner_user_id=request.user.id,
            updates=serializer.validated_data,
        )

        if updated is None:
            return Response(
                {"detail": "Failed to update journey state. User profile may not exist in graph."},
                status=status.HTTP_404_NOT_FOUND,
            )

        # Re-read full state for consistent response
        state = get_journey_state_from_graph(
            workspace_id=str(workspace.id),
            owner_user_id=request.user.id,
        )
        return Response(_format_journey_state(state))


class ProvocationResponseView(WorkspaceAPIView):
    """
    GET  /api/journey/responses/ — List all provocation responses
    POST /api/journey/responses/ — Submit a new provocation response
    """

    def get(self, request):
        workspace = self.get_workspace()
        responses = list_provocation_responses_from_graph(
            workspace_id=str(workspace.id),
            owner_user_id=request.user.id,
        )
        return Response({
            "responses": [_format_provocation_response(r) for r in responses],
            "count": len(responses),
        })

    def post(self, request):
        serializer = ProvocationResponseSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        workspace = self.get_workspace()
        created = create_provocation_response_in_graph(
            workspace_id=str(workspace.id),
            owner_user_id=request.user.id,
            response_data=serializer.validated_data,
        )

        if created is None:
            return Response(
                {"detail": "Failed to create provocation response. User profile may not exist in graph."},
                status=status.HTTP_404_NOT_FOUND,
            )

        return Response(
            _format_provocation_response(created),
            status=status.HTTP_201_CREATED,
        )


class JourneyProfileView(WorkspaceAPIView):
    """
    GET /api/journey/profile/ — Compute and return both ValueProfile
    and SoulProfile for the authenticated user.
    """

    def get(self, request):
        workspace = self.get_workspace()
        ws_id = str(workspace.id)
        uid = request.user.id

        value_profile = compute_value_profile(workspace_id=ws_id, owner_user_id=uid)
        soul_profile = compute_soul_profile(workspace_id=ws_id, owner_user_id=uid)

        return Response({
            "valueProfile": value_profile,
            "soulProfile": soul_profile,
        })


class ValueProfileView(WorkspaceAPIView):
    """
    GET /api/journey/profile/values/ — Compute and return ValueProfile only.
    """

    def get(self, request):
        workspace = self.get_workspace()
        profile = compute_value_profile(
            workspace_id=str(workspace.id),
            owner_user_id=request.user.id,
        )
        return Response({"valueProfile": profile})


class SoulProfileView(WorkspaceAPIView):
    """
    GET /api/journey/profile/soul/ — Compute and return SoulProfile only.
    """

    def get(self, request):
        workspace = self.get_workspace()
        profile = compute_soul_profile(
            workspace_id=str(workspace.id),
            owner_user_id=request.user.id,
        )
        return Response({"soulProfile": profile})
