from django.contrib import admin

from .models import Event, Pattern, PatternRun, Workspace, WorkspaceMembership


@admin.register(Workspace)
class WorkspaceAdmin(admin.ModelAdmin):
    list_display = ("id", "name", "owner", "is_personal", "created_at")
    search_fields = ("name", "owner__username", "owner__email")


@admin.register(WorkspaceMembership)
class WorkspaceMembershipAdmin(admin.ModelAdmin):
    list_display = ("workspace", "user", "role", "created_at")
    list_filter = ("role",)
    search_fields = ("workspace__name", "user__username", "user__email")


@admin.register(Event)
class EventAdmin(admin.ModelAdmin):
    list_display = ("id", "workspace", "created_by", "occurred_at", "source", "emotion", "intensity")
    list_filter = ("source", "emotion")
    search_fields = ("text", "reaction", "outcome")


@admin.register(PatternRun)
class PatternRunAdmin(admin.ModelAdmin):
    list_display = ("id", "workspace", "created_by", "created_at", "event_count")
    readonly_fields = ("created_at",)


@admin.register(Pattern)
class PatternAdmin(admin.ModelAdmin):
    list_display = ("id", "run", "key", "name", "score")
    list_filter = ("run",)
    search_fields = ("key", "name", "hypothesis")
