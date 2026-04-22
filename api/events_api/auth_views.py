import json
import logging
import math
import uuid
from datetime import datetime, timezone as dt_timezone
from xml.sax.saxutils import escape

from django.contrib.auth import authenticate, get_user_model
from django.db import transaction
from django.http import HttpResponse
from django.middleware.csrf import get_token
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import ensure_csrf_cookie
from rest_framework import status
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework_simplejwt.tokens import RefreshToken

from .graph_sync import delete_user_graph_data, delete_workspace_graph_data, full_user_graph_sync
from .neo4j_client import cypher_query
from .models import Event, Pattern, PatternRun, Workspace, ensure_personal_workspace
from .serializers import LoginSerializer, LogoutSerializer, PromoteSerializer, SignupSerializer
from .workspaces import WorkspaceAPIView, get_request_workspace

User = get_user_model()

logger = logging.getLogger(__name__)


def _serialize_workspace(workspace, *, role=None, user_id=None):
    return {
        "id": str(workspace.id),
        "name": workspace.name,
        "is_personal": workspace.is_personal,
        "role": role or Workspace.ROLE_OWNER,
        "is_owner": (role == Workspace.ROLE_OWNER) or (user_id is not None and workspace.owner_id == user_id),
    }


def _serialize_user_state(user, *, workspace=None):
    if not getattr(user, "is_authenticated", False):
        return {"authenticated": False, "user": None, "workspace": None}

    default_workspace = ensure_personal_workspace(user)
    active_workspace = workspace or default_workspace
    memberships = list(
        user.workspace_memberships.select_related("workspace")
        .order_by("-workspace__is_personal", "workspace__created_at")
    )
    active_role = next(
        (membership.role for membership in memberships if membership.workspace_id == active_workspace.id),
        Workspace.ROLE_OWNER,
    )
    return {
        "authenticated": True,
        "user": {
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "is_anonymous": user.username.startswith("anon_"),
        },
        "workspace": _serialize_workspace(active_workspace, role=active_role, user_id=user.id),
        "workspaces": [
            _serialize_workspace(membership.workspace, role=membership.role, user_id=user.id)
            for membership in memberships
        ],
    }


def _issue_tokens(user):
    refresh = RefreshToken.for_user(user)
    return {
        "accessToken": str(refresh.access_token),
        "refreshToken": str(refresh),
    }


def _serialize_auth_response(user, *, status_code=status.HTTP_200_OK):
    payload = _serialize_user_state(user)
    payload.update(_issue_tokens(user))
    return Response(payload, status=status_code)


GRAPH_KIND_COLORS = {
    "Conversation": "#b14f22",
    "Topic": "#2d8659",
    "Pattern": "#916b3b",
    "UserProfile": "#5b6abf",
    "DataSource": "#8b5e3c",
    "UserTurn": "#c47a3b",
    "AssistantTurn": "#7d9157",
    "Signal": "#d4763b",
    "Cluster": "#8b3b7d",
    "PipelineTrace": "#4a8fa6",
    "Person": "#6b8f3b",
    "Insight": "#bf9f3b",
    "Reflection": "#3b6fbf",
    "ContextNode": "#7a6b8f",
    "ActionNode": "#8f6b5e",
    "TemporalNode": "#5e8f7a",
    "default": "#7a8a85",
}


def _normalize_export_value(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {key: _normalize_export_value(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_normalize_export_value(item) for item in value]

    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        return isoformat()

    iso_format = getattr(value, "iso_format", None)
    if callable(iso_format):
        return iso_format()

    return str(value)


def _format_export_timestamp(value):
    normalized = _normalize_export_value(value)
    if normalized in (None, ""):
        return None

    if isinstance(normalized, (int, float)):
        return datetime.fromtimestamp(float(normalized), tz=dt_timezone.utc).isoformat()

    if isinstance(normalized, str):
        try:
            return datetime.fromtimestamp(float(normalized), tz=dt_timezone.utc).isoformat()
        except (TypeError, ValueError):
            return normalized

    return str(normalized)


def _graph_rows(query, params):
    rows, columns = cypher_query(query, params)
    return [_normalize_export_value(dict(zip(columns, row))) for row in rows]


def build_conversations_export(*, user, workspace):
    workspace_id = str(workspace.id)
    conversations = _graph_rows(
        """
            MATCH (c:Conversation {workspace_id: $workspace_id})
            RETURN c.conversation_id AS conversation_id,
                   coalesce(c.title, 'Untitled conversation') AS title,
                   coalesce(c.turn_count, 0) AS turn_count,
                   c.last_active AS last_active,
                   c.create_time AS create_time
            ORDER BY c.last_active DESC, c.create_time DESC
        """,
        {"workspace_id": workspace_id},
    )

    exported_conversations = []
    for conversation in conversations:
        history = _graph_rows(
            """
                MATCH (c:Conversation {conversation_id: $conversation_id, workspace_id: $workspace_id})
                      -[:CONTAINS]->(turn)
                WHERE turn:UserTurn OR turn:AssistantTurn
                WITH turn,
                     CASE WHEN 'UserTurn' IN labels(turn) THEN 'user' ELSE 'assistant' END AS role
                RETURN role,
                       coalesce(turn.content, turn.content_preview) AS content,
                       turn.create_time AS created_at
                ORDER BY turn.create_time ASC
            """,
            {
                "workspace_id": workspace_id,
                "conversation_id": conversation["conversation_id"],
            },
        )

        exported_conversations.append(
            {
                "conversation_id": conversation["conversation_id"],
                "title": conversation["title"],
                "turn_count": conversation["turn_count"],
                "last_active": _format_export_timestamp(conversation.get("last_active")),
                "created_at": _format_export_timestamp(conversation.get("create_time")),
                "messages": [
                    {
                        "role": turn.get("role"),
                        "content": turn.get("content") or "",
                        "created_at": _format_export_timestamp(turn.get("created_at")),
                    }
                    for turn in history
                ],
            }
        )

    return {
        "export_type": "conversations",
        "exported_at": datetime.now(dt_timezone.utc).isoformat(),
        "user": {
            "id": user.id,
            "username": user.username,
            "email": user.email,
        },
        "workspace": {
            "id": workspace_id,
            "name": workspace.name,
        },
        "conversation_count": len(exported_conversations),
        "conversations": exported_conversations,
    }


def _layout_memory_graph(nodes):
    return _layout_memory_graph_force(nodes, [])


def _stable_offset(value):
    return sum(ord(char) for char in str(value)) % 997


def _kind_anchor(kind):
    anchors = {
        "UserProfile": (900.0, 620.0),
        "Conversation": (900.0, 620.0),
        "Topic": (1450.0, 620.0),
        "Person": (1230.0, 220.0),
        "ContextNode": (760.0, 180.0),
        "ActionNode": (760.0, 1060.0),
        "TemporalNode": (1230.0, 1020.0),
        "Emotion": (1130.0, 620.0),
        "UserTurn": (320.0, 360.0),
        "AssistantTurn": (320.0, 860.0),
        "Signal": (1560.0, 280.0),
        "Cluster": (1560.0, 960.0),
        "Pattern": (1490.0, 120.0),
        "Insight": (1490.0, 1120.0),
        "PipelineTrace": (150.0, 1080.0),
        "DataSource": (150.0, 160.0),
        "Reflection": (520.0, 620.0),
        "default": (900.0, 620.0),
    }
    return anchors.get(kind, anchors["default"])


def _layout_memory_graph_force(nodes, edges):
    if not nodes:
        return {}

    width = 1800.0
    height = 1240.0
    positions = {}
    grouped = {}
    for node in nodes:
        grouped.setdefault(node.get("kind") or "default", []).append(node)

    # Seed nodes around kind-specific anchor regions so the force pass starts structured.
    for kind, group in grouped.items():
        anchor_x, anchor_y = _kind_anchor(kind)
        columns = max(1, math.ceil(math.sqrt(len(group))))
        spacing_x = 46.0
        spacing_y = 42.0
        for index, node in enumerate(group):
            col = index % columns
            row = index // columns
            offset_seed = _stable_offset(node.get("uid"))
            jitter_x = ((offset_seed % 11) - 5) * 3.0
            jitter_y = (((offset_seed // 11) % 11) - 5) * 3.0
            positions[node["uid"]] = (
                anchor_x + (col - (columns - 1) / 2.0) * spacing_x + jitter_x,
                anchor_y + (row - max(0, (len(group) / columns) - 1) / 2.0) * spacing_y + jitter_y,
            )

    node_lookup = {node["uid"]: node for node in nodes}
    adjacency = {node["uid"]: set() for node in nodes}
    for edge in edges:
        source = edge.get("source")
        target = edge.get("target")
        if source in adjacency and target in adjacency:
            adjacency[source].add(target)
            adjacency[target].add(source)

    temperature = 18.0
    repulsion = 320000.0
    spring_strength = 0.018
    anchor_strength = 0.014
    min_distance = 24.0

    node_ids = list(positions.keys())
    for _ in range(42):
        displacement = {node_id: [0.0, 0.0] for node_id in node_ids}

        for index, source_id in enumerate(node_ids):
            sx, sy = positions[source_id]
            for target_id in node_ids[index + 1:]:
                tx, ty = positions[target_id]
                dx = sx - tx
                dy = sy - ty
                distance_sq = max(dx * dx + dy * dy, 0.01)
                distance = math.sqrt(distance_sq)
                repel = repulsion / distance_sq
                nx = dx / distance
                ny = dy / distance
                displacement[source_id][0] += nx * repel
                displacement[source_id][1] += ny * repel
                displacement[target_id][0] -= nx * repel
                displacement[target_id][1] -= ny * repel

                if distance < min_distance:
                    overlap_force = (min_distance - distance) * 1.6
                    displacement[source_id][0] += nx * overlap_force
                    displacement[source_id][1] += ny * overlap_force
                    displacement[target_id][0] -= nx * overlap_force
                    displacement[target_id][1] -= ny * overlap_force

        for source_id, neighbors in adjacency.items():
            sx, sy = positions[source_id]
            for target_id in neighbors:
                if source_id >= target_id:
                    continue
                tx, ty = positions[target_id]
                dx = tx - sx
                dy = ty - sy
                distance = max(math.sqrt(dx * dx + dy * dy), 0.01)
                desired = 95.0
                if node_lookup[source_id].get("kind") == "Conversation" or node_lookup[target_id].get("kind") == "Conversation":
                    desired = 110.0
                spring_force = (distance - desired) * spring_strength
                nx = dx / distance
                ny = dy / distance
                displacement[source_id][0] += nx * spring_force
                displacement[source_id][1] += ny * spring_force
                displacement[target_id][0] -= nx * spring_force
                displacement[target_id][1] -= ny * spring_force

        for node_id in node_ids:
            node = node_lookup[node_id]
            px, py = positions[node_id]
            anchor_x, anchor_y = _kind_anchor(node.get("kind") or "default")
            displacement[node_id][0] += (anchor_x - px) * anchor_strength
            displacement[node_id][1] += (anchor_y - py) * anchor_strength

        for node_id in node_ids:
            px, py = positions[node_id]
            dx, dy = displacement[node_id]
            magnitude = max(math.sqrt(dx * dx + dy * dy), 0.01)
            limited = min(temperature, magnitude)
            px += (dx / magnitude) * limited
            py += (dy / magnitude) * limited
            positions[node_id] = (
                min(width - 50.0, max(50.0, px)),
                min(height - 50.0, max(50.0, py)),
            )

        temperature *= 0.93

    return positions


def _node_label_priority(node, degree):
    kind = node.get("kind") or "default"
    if kind == "UserProfile":
        return 1000
    if kind == "Conversation":
        return 900 + degree
    if kind in {"Person", "ContextNode", "ActionNode", "TemporalNode", "Pattern", "Cluster", "Insight"}:
        return 700 + degree
    if kind == "Topic":
        return 500 + degree
    if kind in {"UserTurn", "AssistantTurn"}:
        return 250 + degree
    return 100 + degree


def _select_labeled_nodes(nodes, edges):
    degree = {node["uid"]: 0 for node in nodes}
    for edge in edges:
        if edge.get("source") in degree:
            degree[edge["source"]] += 1
        if edge.get("target") in degree:
            degree[edge["target"]] += 1

    by_kind = {}
    for node in nodes:
        by_kind.setdefault(node.get("kind") or "default", []).append(node)

    labeled = set()
    always_label = {"UserProfile", "Conversation", "Person", "ContextNode", "ActionNode", "TemporalNode"}
    caps = {
        "Conversation": 18,
        "Topic": 28,
        "UserTurn": 6,
        "AssistantTurn": 6,
        "Signal": 0,
        "PipelineTrace": 0,
        "DataSource": 3,
        "Pattern": 8,
        "Cluster": 8,
        "Insight": 8,
        "Emotion": 10,
    }

    for kind, group in by_kind.items():
        ranked = sorted(
            group,
            key=lambda node: (
                -_node_label_priority(node, degree.get(node["uid"], 0)),
                str(node.get("label") or ""),
            ),
        )
        limit = caps.get(kind, len(group) if kind in always_label else 5)
        if kind in always_label:
            limit = max(limit, min(len(group), 10))
        for node in ranked[:limit]:
            labeled.add(node["uid"])

    return labeled, degree


def build_memory_graph_svg(*, workspace):
    workspace_id = str(workspace.id)
    nodes = _graph_rows(
        """
            MATCH (n)
            WHERE n.workspace_id = $workspace_id AND n.uid IS NOT NULL
            WITH n, labels(n) AS lbls
            RETURN n.uid AS uid,
                   lbls[0] AS kind,
                   COALESCE(n.title, n.name, n.word, n.content_preview, n.username, n.description, '') AS label
            ORDER BY kind ASC, label ASC
        """,
        {"workspace_id": workspace_id},
    )
    edges = _graph_rows(
        """
            MATCH (a)-[r]->(b)
            WHERE a.workspace_id = $workspace_id
              AND b.workspace_id = $workspace_id
              AND a.uid IS NOT NULL
              AND b.uid IS NOT NULL
            RETURN DISTINCT a.uid AS source, b.uid AS target, type(r) AS rel_type
        """,
        {"workspace_id": workspace_id},
    )

    positions = _layout_memory_graph_force(nodes, edges)
    labeled_nodes, _degree = _select_labeled_nodes(nodes, edges)
    width = 1800
    height = 1240

    legend_counts = {}
    for node in nodes:
        kind = node.get("kind") or "default"
        legend_counts[kind] = legend_counts.get(kind, 0) + 1

    edge_markup = []
    for edge in edges:
        source = positions.get(edge.get("source"))
        target = positions.get(edge.get("target"))
        if not source or not target:
            continue
        dx = target[0] - source[0]
        dy = target[1] - source[1]
        distance = max(math.sqrt(dx * dx + dy * dy), 1.0)
        nx = -dy / distance
        ny = dx / distance
        curve_seed = (_stable_offset(f"{edge.get('source')}->{edge.get('target')}") % 9) - 4
        offset = curve_seed * 10.0
        control_x = (source[0] + target[0]) / 2.0 + nx * offset
        control_y = (source[1] + target[1]) / 2.0 + ny * offset
        edge_markup.append(
            f'<path d="M {source[0]:.1f} {source[1]:.1f} Q {control_x:.1f} {control_y:.1f} {target[0]:.1f} {target[1]:.1f}" '
            'fill="none" stroke="rgba(110,110,110,0.16)" stroke-width="0.9" />'
        )

    node_markup = []
    for node in nodes:
        x, y = positions.get(node["uid"], (540.0, 420.0))
        kind = node.get("kind") or "default"
        color = GRAPH_KIND_COLORS.get(kind, GRAPH_KIND_COLORS["default"])
        label = escape(str(node.get("label") or kind))
        short_label = f"{label[:28]}…" if len(label) > 29 else label
        radius = 7 if kind == "Conversation" else 6
        radius = 9 if kind == "UserProfile" else radius
        radius = 5 if kind in {"Signal", "PipelineTrace", "UserTurn", "AssistantTurn"} else radius
        node_markup.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}" fill="{color}" stroke="#ffffff" stroke-width="1.6" />'
        )
        if node["uid"] in labeled_nodes:
            anchor = "start" if x < width / 2 else "end"
            label_x = x + 12 if anchor == "start" else x - 12
            node_markup.append(
                f'<text x="{label_x:.1f}" y="{y + 4:.1f}" font-size="12" text-anchor="{anchor}" fill="#1b2523">{short_label}</text>'
            )

    legend_markup = []
    for index, (kind, count) in enumerate(sorted(legend_counts.items())):
        y = 96 + (index * 20)
        color = GRAPH_KIND_COLORS.get(kind, GRAPH_KIND_COLORS["default"])
        legend_markup.append(
            f'<circle cx="38" cy="{y}" r="5" fill="{color}" />'
            f'<text x="52" y="{y + 4}" font-size="12" fill="#1b2523">{escape(kind)} ({count})</text>'
        )

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#f7f4ee" />
  <text x="32" y="42" font-size="24" font-weight="700" fill="#1b2523">ThriveSight Memory Graph</text>
  <text x="32" y="66" font-size="13" fill="#5e6a67">Workspace: {escape(workspace.name)} • Exported {escape(datetime.now(dt_timezone.utc).strftime("%B %d, %Y"))}</text>
  <text x="32" y="86" font-size="12" fill="#5e6a67">Layout: force-directed with type anchors and label thinning for readability</text>
  <g>{''.join(edge_markup)}</g>
  <g>{''.join(node_markup)}</g>
  <g>{''.join(legend_markup)}</g>
</svg>"""


@method_decorator(ensure_csrf_cookie, name="dispatch")
class CsrfTokenView(APIView):
    permission_classes = [AllowAny]

    def get(self, request):
        return Response({"csrfToken": get_token(request)})


@method_decorator(ensure_csrf_cookie, name="dispatch")
class SessionView(APIView):
    permission_classes = [AllowAny]

    def get(self, request):
        workspace = None
        if getattr(request.user, "is_authenticated", False):
            workspace = get_request_workspace(request)
        return Response(_serialize_user_state(request.user, workspace=workspace))


class SignupView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        serializer = SignupSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        user = serializer.save()

        # Sync new user to Neo4j knowledge graph (best-effort, never blocks auth)
        workspace = ensure_personal_workspace(user)
        full_user_graph_sync(user, workspace)

        return _serialize_auth_response(user, status_code=status.HTTP_201_CREATED)


class LoginView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        serializer = LoginSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        user = authenticate(
            request,
            username=serializer.validated_data["username"],
            password=serializer.validated_data["password"],
        )
        if user is None:
            return Response(
                {"detail": "Invalid username or password."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        workspace = ensure_personal_workspace(user)
        try:
            full_user_graph_sync(user, workspace)
        except Exception as exc:
            logger.warning("Graph sync failed on login for user %s: %s", user.username, exc)

        return _serialize_auth_response(user)


class LogoutView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        serializer = LogoutSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        refresh_token = serializer.validated_data["refresh"]

        try:
            token = RefreshToken(refresh_token)
            if token.get("user_id") != request.user.id:
                return Response(
                    {"detail": "Refresh token does not belong to the authenticated user."},
                    status=status.HTTP_400_BAD_REQUEST,
                )
            token.blacklist()
        except Exception:
            return Response(
                {"detail": "Refresh token is invalid or already revoked."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        return Response({"authenticated": False, "user": None, "workspace": None})


class AnonymousSessionView(APIView):
    """Create an anonymous user with a real workspace for immediate interaction."""

    permission_classes = [AllowAny]

    def post(self, request):
        username = f"anon_{uuid.uuid4().hex[:12]}"
        user = User.objects.create_user(username=username, email="", password=None)
        user.set_unusable_password()
        user.save(update_fields=["password"])

        workspace = ensure_personal_workspace(user)
        try:
            full_user_graph_sync(user, workspace)
        except Exception as exc:
            logger.warning("Graph sync failed for anonymous user %s: %s", username, exc)

        return _serialize_auth_response(user, status_code=status.HTTP_201_CREATED)


class PromoteAnonymousView(APIView):
    """Promote an anonymous user to a full account, preserving all workspace data."""

    permission_classes = [IsAuthenticated]

    def post(self, request):
        user = request.user

        if not user.username.startswith("anon_"):
            return Response(
                {"detail": "Only anonymous accounts can be promoted."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        serializer = PromoteSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        user.username = serializer.validated_data["username"]
        user.email = serializer.validated_data.get("email", "")
        user.set_password(serializer.validated_data["password"])
        user.save()

        # Update workspace name to reflect new username
        workspace = ensure_personal_workspace(user)
        workspace.name = f"{user.username}'s Workspace"
        workspace.save(update_fields=["name"])

        return _serialize_auth_response(user)


class DemoUsersView(APIView):
    """List available demo user profiles for exploration."""

    permission_classes = [AllowAny]

    DEMO_PROFILES = [
        {
            "username": "demo_alex",
            "display_name": "Alex",
            "description": "Processing a recurring conflict with their manager — see how emotional patterns emerge across workplace interactions.",
            "theme": "Workplace Conflict",
        },
        {
            "username": "demo_jordan",
            "display_name": "Jordan",
            "description": "Working through a relationship decision — explore how the system maps emotional tension across contexts.",
            "theme": "Relationship Decision",
        },
        {
            "username": "demo_sam",
            "display_name": "Sam",
            "description": "Journaling vague feelings — see how ThriveSight handles ambiguity and gradually resolves emotional patterns.",
            "theme": "Exploratory Journaling",
        },
    ]

    def get(self, request):
        available = []
        for profile in self.DEMO_PROFILES:
            user = User.objects.filter(username=profile["username"]).first()
            if user:
                workspace = Workspace.objects.filter(owner=user, is_personal=True).first()
                stats = {"signal_count": 0, "conversation_count": 0}
                if workspace:
                    try:
                        rows, _ = cypher_query(
                            "MATCH (s:Signal {workspace_id: $ws}) RETURN count(s) AS sc",
                            {"ws": str(workspace.id)},
                        )
                        stats["signal_count"] = rows[0][0] if rows else 0
                        rows2, _ = cypher_query(
                            "MATCH (c:Conversation {workspace_id: $ws}) RETURN count(c) AS cc",
                            {"ws": str(workspace.id)},
                        )
                        stats["conversation_count"] = rows2[0][0] if rows2 else 0
                    except Exception:
                        pass

                available.append({**profile, **stats, "workspace_id": str(workspace.id) if workspace else None})

        return Response({"demo_users": available})


class DeleteDataView(WorkspaceAPIView):
    """Delete persisted data for the current personal workspace while preserving the account."""

    def delete(self, request):
        workspace = self.get_workspace()
        workspace_id = str(workspace.id)

        deleted_events = Event.objects.filter(workspace=workspace).count()
        deleted_pattern_runs = PatternRun.objects.filter(workspace=workspace).count()
        deleted_patterns = Pattern.objects.filter(run__workspace=workspace).count()

        try:
            with transaction.atomic():
                PatternRun.objects.filter(workspace=workspace).delete()
                Event.objects.filter(workspace=workspace).delete()
                graph_summary = delete_workspace_graph_data(workspace_id=workspace_id)
        except Exception as exc:
            return Response(
                {"detail": f"Failed to delete your stored data: {exc}"},
                status=status.HTTP_502_BAD_GATEWAY,
            )

        return Response(
            {
                "deleted": True,
                "deleted_events": deleted_events,
                "deleted_pattern_runs": deleted_pattern_runs,
                "deleted_patterns": deleted_patterns,
                "deleted_graph_nodes": graph_summary.get("deleted_nodes", 0),
                "workspace": {
                    "id": workspace_id,
                    "name": workspace.name,
                },
            }
        )


class ExportConversationsView(WorkspaceAPIView):
    """Download stored conversations as a readable JSON file."""

    def get(self, request):
        workspace = self.get_workspace()
        try:
            payload = build_conversations_export(user=request.user, workspace=workspace)
        except Exception as exc:
            return Response(
                {"detail": f"Failed to export conversations: {exc}"},
                status=status.HTTP_502_BAD_GATEWAY,
            )
        response = HttpResponse(
            json.dumps(payload, indent=2),
            content_type="application/json",
        )
        response["Content-Disposition"] = (
            f'attachment; filename="thrivesight-conversations-{request.user.username}.json"'
        )
        return response


class ExportMemoryGraphView(WorkspaceAPIView):
    """Download the current workspace memory graph as an SVG image."""

    def get(self, request):
        workspace = self.get_workspace()
        try:
            svg = build_memory_graph_svg(workspace=workspace)
        except Exception as exc:
            return Response(
                {"detail": f"Failed to export memory graph: {exc}"},
                status=status.HTTP_502_BAD_GATEWAY,
            )
        response = HttpResponse(svg, content_type="image/svg+xml")
        response["Content-Disposition"] = (
            f'attachment; filename="thrivesight-memory-graph-{request.user.username}.svg"'
        )
        return response


class DeleteAccountView(WorkspaceAPIView):
    """Delete the signed-in account and all owned workspace data."""

    def delete(self, request):
        user = request.user
        owned_workspaces = list(Workspace.objects.filter(owner=user).values_list("id", flat=True))
        workspace_ids = [str(workspace_id) for workspace_id in owned_workspaces]

        deleted_events = Event.objects.filter(workspace_id__in=owned_workspaces).count()
        deleted_pattern_runs = PatternRun.objects.filter(workspace_id__in=owned_workspaces).count()
        deleted_patterns = Pattern.objects.filter(run__workspace_id__in=owned_workspaces).count()
        deleted_workspaces = len(workspace_ids)
        deleted_user_id = user.id
        deleted_username = user.username

        try:
            with transaction.atomic():
                graph_summary = delete_user_graph_data(
                    workspace_ids=workspace_ids,
                    owner_user_id=user.id,
                )
                user.delete()
        except Exception as exc:
            return Response(
                {"detail": f"Failed to delete your account: {exc}"},
                status=status.HTTP_502_BAD_GATEWAY,
            )

        return Response(
            {
                "deleted": True,
                "deleted_user_id": deleted_user_id,
                "deleted_username": deleted_username,
                "deleted_workspaces": deleted_workspaces,
                "deleted_events": deleted_events,
                "deleted_pattern_runs": deleted_pattern_runs,
                "deleted_patterns": deleted_patterns,
                "deleted_graph_nodes": graph_summary.get("deleted_nodes", 0),
            }
        )
