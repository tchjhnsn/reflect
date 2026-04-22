from rest_framework.exceptions import NotFound
from rest_framework.permissions import IsAuthenticated
from rest_framework.views import APIView

from .models import Workspace, WorkspaceMembership, ensure_personal_workspace


WORKSPACE_ID_HEADER = "X-Workspace-Id"


def _extract_requested_workspace_id(request) -> str | None:
    header_value = (request.headers.get(WORKSPACE_ID_HEADER) or "").strip()
    if header_value:
        return header_value

    query_value = (request.query_params.get("workspace_id") or "").strip()
    if query_value:
        return query_value

    data = getattr(request, "data", None)
    if isinstance(data, dict):
        raw_value = data.get("workspace_id")
        if raw_value is not None:
            body_value = str(raw_value).strip()
            if body_value:
                return body_value

    return None


def get_request_workspace(request):
    user = request.user
    workspace = ensure_personal_workspace(user)
    WorkspaceMembership.objects.get_or_create(
        workspace=workspace,
        user=user,
        defaults={"role": workspace.ROLE_OWNER},
    )

    requested_workspace_id = _extract_requested_workspace_id(request)
    if not requested_workspace_id or requested_workspace_id == str(workspace.id):
        return workspace

    membership = (
        WorkspaceMembership.objects.select_related("workspace")
        .filter(user=user, workspace_id=requested_workspace_id)
        .first()
    )
    if membership is not None:
        return membership.workspace

    owned_workspace = Workspace.objects.filter(owner=user, id=requested_workspace_id).first()
    if owned_workspace is not None:
        WorkspaceMembership.objects.get_or_create(
            workspace=owned_workspace,
            user=user,
            defaults={"role": Workspace.ROLE_OWNER},
        )
        return owned_workspace

    raise NotFound("Workspace not found or not accessible to this user.")


class WorkspaceAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def get_workspace(self):
        return get_request_workspace(self.request)
