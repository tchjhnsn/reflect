"""
Management command to backfill existing Django users into Neo4j.

Usage:
    python manage.py sync_users_to_graph          # sync all users
    python manage.py sync_users_to_graph --user 1  # sync user with id=1
    python manage.py sync_users_to_graph --dry-run  # preview without writing

Zone 1: Proprietary — internal tool.
"""

from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand

from events_api.models import ensure_personal_workspace
from events_api.graph_sync import full_user_graph_sync

User = get_user_model()


class Command(BaseCommand):
    help = "Sync Django users to Neo4j UserProfile nodes and back-link owned graph data."

    def add_arguments(self, parser):
        parser.add_argument(
            "--user",
            type=int,
            help="Sync a single user by Django user ID",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="List users that would be synced without writing to Neo4j",
        )

    def handle(self, *args, **options):
        user_id = options.get("user")
        dry_run = options.get("dry_run", False)

        if user_id:
            users = User.objects.filter(id=user_id)
            if not users.exists():
                self.stderr.write(self.style.ERROR(f"User with id={user_id} not found."))
                return
        else:
            users = User.objects.all()

        total = users.count()
        self.stdout.write(f"Found {total} user(s) to sync.")

        if dry_run:
            self.stdout.write(self.style.WARNING("DRY RUN — no Neo4j writes."))
            for user in users:
                workspace = ensure_personal_workspace(user)
                self.stdout.write(
                    f"  Would sync: {user.username} (id={user.id}, "
                    f"workspace={workspace.id})"
                )
            return

        synced = 0
        failed = 0

        for user in users:
            workspace = ensure_personal_workspace(user)
            profile = full_user_graph_sync(user, workspace)

            if profile is not None:
                synced += 1
                self.stdout.write(
                    self.style.SUCCESS(
                        f"  Synced: {user.username} (id={user.id})"
                    )
                )
            else:
                failed += 1
                self.stderr.write(
                    self.style.ERROR(
                        f"  FAILED: {user.username} (id={user.id})"
                    )
                )

        self.stdout.write("")
        self.stdout.write(
            self.style.SUCCESS(f"Done. Synced: {synced}, Failed: {failed}, Total: {total}")
        )
