# Generated manually for MVP schema.
import uuid

import django.core.validators
from django.db import migrations, models


class Migration(migrations.Migration):
    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="Event",
            fields=[
                ("id", models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("occurred_at", models.DateTimeField()),
                (
                    "source",
                    models.CharField(
                        choices=[("manual", "Manual"), ("chat", "Chat"), ("import", "Import")],
                        default="manual",
                        max_length=20,
                    ),
                ),
                ("text", models.TextField()),
                ("context_tags", models.JSONField(blank=True, default=list)),
                ("people", models.JSONField(blank=True, default=list)),
                ("emotion", models.CharField(blank=True, max_length=100, null=True)),
                (
                    "intensity",
                    models.PositiveSmallIntegerField(
                        blank=True,
                        null=True,
                        validators=[
                            django.core.validators.MinValueValidator(1),
                            django.core.validators.MaxValueValidator(5),
                        ],
                    ),
                ),
                ("reaction", models.TextField(blank=True, null=True)),
                ("outcome", models.TextField(blank=True, null=True)),
            ],
            options={
                "ordering": ["-occurred_at", "-created_at"],
            },
        ),
        migrations.CreateModel(
            name="PatternRun",
            fields=[
                ("id", models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("params", models.JSONField(blank=True, default=dict)),
                ("event_count", models.PositiveIntegerField(default=0)),
                ("notes", models.TextField(blank=True, null=True)),
            ],
            options={
                "ordering": ["-created_at"],
            },
        ),
        migrations.CreateModel(
            name="Pattern",
            fields=[
                ("id", models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ("key", models.CharField(max_length=255)),
                ("name", models.CharField(max_length=255)),
                ("hypothesis", models.TextField()),
                ("score", models.FloatField()),
                ("evidence", models.JSONField(blank=True, default=list)),
                (
                    "run",
                    models.ForeignKey(on_delete=models.deletion.CASCADE, related_name="patterns", to="events_api.patternrun"),
                ),
            ],
            options={
                "ordering": ["-score", "name"],
            },
        ),
    ]
