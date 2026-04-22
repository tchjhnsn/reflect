import uuid

from django.conf import settings
from django.core.validators import MaxValueValidator, MinValueValidator
from django.db import models
from django.db.models import Q


class Workspace(models.Model):
    ROLE_OWNER = "owner"
    ROLE_MEMBER = "member"

    ROLE_CHOICES = [
        (ROLE_OWNER, "Owner"),
        (ROLE_MEMBER, "Member"),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=255)
    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="owned_workspaces",
    )
    is_personal = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["created_at"]
        constraints = [
            models.UniqueConstraint(
                fields=["owner"],
                condition=Q(is_personal=True),
                name="unique_personal_workspace_per_owner",
            )
        ]

    def __str__(self) -> str:
        return self.name


class WorkspaceMembership(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    workspace = models.ForeignKey(Workspace, on_delete=models.CASCADE, related_name="memberships")
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="workspace_memberships")
    role = models.CharField(max_length=20, choices=Workspace.ROLE_CHOICES, default=Workspace.ROLE_MEMBER)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["created_at"]
        constraints = [
            models.UniqueConstraint(fields=["workspace", "user"], name="unique_workspace_membership")
        ]

    def __str__(self) -> str:
        return f"{self.user_id}:{self.workspace_id}:{self.role}"


class Event(models.Model):
    SOURCE_MANUAL = "manual"
    SOURCE_CHAT = "chat"
    SOURCE_IMPORT = "import"

    SOURCE_CHOICES = [
        (SOURCE_MANUAL, "Manual"),
        (SOURCE_CHAT, "Chat"),
        (SOURCE_IMPORT, "Import"),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    workspace = models.ForeignKey(
        Workspace,
        on_delete=models.CASCADE,
        related_name="events",
        null=True,
        blank=True,
    )
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        related_name="events",
        null=True,
        blank=True,
    )
    created_at = models.DateTimeField(auto_now_add=True)
    occurred_at = models.DateTimeField()
    source = models.CharField(max_length=20, choices=SOURCE_CHOICES, default=SOURCE_MANUAL)
    text = models.TextField()
    context_tags = models.JSONField(default=list, blank=True)
    people = models.JSONField(default=list, blank=True)
    emotion = models.CharField(max_length=100, null=True, blank=True)
    intensity = models.PositiveSmallIntegerField(
        null=True,
        blank=True,
        validators=[MinValueValidator(1), MaxValueValidator(5)],
    )
    reaction = models.TextField(null=True, blank=True)
    outcome = models.TextField(null=True, blank=True)

    class Meta:
        ordering = ["-occurred_at", "-created_at"]

    def __str__(self) -> str:
        return f"Event {self.id} ({self.occurred_at.isoformat()})"


class PatternRun(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    workspace = models.ForeignKey(
        Workspace,
        on_delete=models.CASCADE,
        related_name="pattern_runs",
        null=True,
        blank=True,
    )
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        related_name="pattern_runs",
        null=True,
        blank=True,
    )
    created_at = models.DateTimeField(auto_now_add=True)
    params = models.JSONField(default=dict, blank=True)
    event_count = models.PositiveIntegerField(default=0)
    notes = models.TextField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"PatternRun {self.id}"


class Pattern(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    run = models.ForeignKey(PatternRun, on_delete=models.CASCADE, related_name="patterns")
    key = models.CharField(max_length=255)
    name = models.CharField(max_length=255)
    hypothesis = models.TextField()
    score = models.FloatField()
    evidence = models.JSONField(default=list, blank=True)

    class Meta:
        ordering = ["-score", "name"]

    def __str__(self) -> str:
        return self.name


def ensure_personal_workspace(user) -> Workspace:
    workspace, _ = Workspace.objects.get_or_create(
        owner=user,
        is_personal=True,
        defaults={"name": f"{user.username}'s Workspace"},
    )
    WorkspaceMembership.objects.get_or_create(
        workspace=workspace,
        user=user,
        defaults={"role": Workspace.ROLE_OWNER},
    )
    return workspace
