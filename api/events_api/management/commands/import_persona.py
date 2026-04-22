import json
import re
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError
from django.db import transaction
from django.utils.dateparse import parse_datetime

from events_api.models import Event, PatternRun

JSONL_FILES = (
    "lifelog.jsonl",
    "conversations.jsonl",
    "emails.jsonl",
    "calendar.jsonl",
    "social_posts.jsonl",
    "transactions.jsonl",
    "files_index.jsonl",
)

CHAT_SOURCES = {"ai_chat"}

EMOTION_LEXICON = {
    "anger": {
        "anger",
        "angry",
        "frustration",
        "frustrated",
        "conflict",
        "fight",
        "argument",
        "resentment",
    },
    "anxiety": {
        "anxiety",
        "anxious",
        "stress",
        "stressed",
        "burnout",
        "overwhelmed",
        "worry",
        "worried",
        "panic",
        "imposter",
    },
    "sadness": {
        "sad",
        "sadness",
        "lonely",
        "isolation",
        "disappointed",
        "disappointment",
        "grief",
    },
    "joy": {
        "joy",
        "happy",
        "gratitude",
        "grateful",
        "proud",
        "win",
        "milestone",
        "celebrate",
        "celebration",
        "success",
    },
    "fear": {
        "fear",
        "afraid",
        "scared",
        "uncertain",
        "uncertainty",
        "risk",
    },
    "guilt": {
        "guilt",
        "guilty",
        "shame",
        "ashamed",
    },
}

HIGH_INTENSITY_TERMS = {
    "urgent",
    "panic",
    "furious",
    "exhausted",
    "overwhelmed",
    "not",
    "sustainable",
    "fight",
    "argument",
    "crisis",
}

LOW_INTENSITY_TERMS = {
    "maybe",
    "might",
    "somewhat",
    "slightly",
    "kind",
    "brief",
}

REACTION_CUES = (
    "i ",
    "i'm ",
    "im ",
    "i felt",
    "i feel",
    "i kept",
    "i ended up",
    "i reacted",
    "i responded",
    "i said",
    "i told",
    "i avoided",
    "i ignored",
    "i snapped",
)

OUTCOME_CUES = (
    "result",
    "led to",
    "ended",
    "outcome",
    "so that",
    "which meant",
    "therefore",
    "afterward",
    "afterwards",
)


def _normalize_tags(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    normalized: list[str] = []
    for item in value:
        if isinstance(item, str):
            cleaned = item.strip()
            if cleaned:
                normalized.append(cleaned)
    return normalized


def _map_source(raw_source: str, filename: str) -> str:
    source = (raw_source or "").strip().lower()
    if source in CHAT_SOURCES or filename == "conversations.jsonl":
        return Event.SOURCE_CHAT
    return Event.SOURCE_IMPORT


def _tokenize(value: str) -> list[str]:
    return [token.lower() for token in re.findall(r"[a-zA-Z0-9]+", value or "")]


def _infer_emotion(tags: list[str], text: str) -> str | None:
    tokens = set(_tokenize(" ".join(tags) + " " + text))
    if not tokens:
        return None

    scores = {emotion: len(tokens.intersection(words)) for emotion, words in EMOTION_LEXICON.items()}
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    if not ranked or ranked[0][1] == 0:
        return None
    return ranked[0][0]


def _infer_intensity(text: str, emotion: str | None) -> int | None:
    if emotion is None:
        return None

    baseline = 3
    if emotion in {"joy"}:
        baseline = 2
    elif emotion in {"anger", "anxiety", "fear"}:
        baseline = 3
    elif emotion in {"sadness", "guilt"}:
        baseline = 2
    elif emotion in {"neutral"}:
        baseline = 2

    tokens = set(_tokenize(text))
    intensity = baseline
    if tokens.intersection(HIGH_INTENSITY_TERMS):
        intensity += 1
    if "!" in text:
        intensity += 1
    if tokens.intersection(LOW_INTENSITY_TERMS):
        intensity -= 1

    return max(1, min(5, intensity))


def _sentences(text: str) -> list[str]:
    chunks = [chunk.strip() for chunk in re.split(r"(?<=[.!?])\s+", text or "") if chunk.strip()]
    return chunks if chunks else ([text.strip()] if text and text.strip() else [])


def _infer_reaction(text: str) -> str | None:
    for sentence in _sentences(text):
        lowered = sentence.lower()
        if any(cue in lowered for cue in REACTION_CUES):
            return sentence[:240]
    return None


def _infer_outcome(text: str) -> str | None:
    for sentence in _sentences(text):
        lowered = sentence.lower()
        if any(cue in lowered for cue in OUTCOME_CUES):
            return sentence[:240]
    return None


class Command(BaseCommand):
    help = "Import a hackathon persona folder (JSON + JSONL) into events."

    def add_arguments(self, parser):
        parser.add_argument(
            "--persona-dir",
            required=True,
            help="Absolute or relative path to persona_pXX folder.",
        )
        parser.add_argument(
            "--replace",
            action="store_true",
            help="Delete all existing events/pattern runs before import.",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Parse and validate files without writing to the database.",
        )
        parser.add_argument(
            "--limit-per-file",
            type=int,
            default=0,
            help="Optional cap of imported lines per file (0 = no cap).",
        )
        parser.add_argument(
            "--enrich",
            action="store_true",
            help="Derive emotion/intensity/reaction/outcome from tags/text using deterministic rules.",
        )

    def handle(self, *args, **options):
        persona_dir = Path(options["persona_dir"]).expanduser().resolve()
        replace = options["replace"]
        dry_run = options["dry_run"]
        limit_per_file = options["limit_per_file"]
        enrich = options["enrich"]

        if limit_per_file < 0:
            raise CommandError("--limit-per-file must be >= 0.")
        if not persona_dir.exists() or not persona_dir.is_dir():
            raise CommandError(f"Persona directory not found: {persona_dir}")

        profile_path = persona_dir / "persona_profile.json"
        consent_path = persona_dir / "consent.json"
        if not profile_path.exists():
            raise CommandError(f"Missing required file: {profile_path}")
        if not consent_path.exists():
            raise CommandError(f"Missing required file: {consent_path}")

        missing_files = [str(persona_dir / name) for name in JSONL_FILES if not (persona_dir / name).exists()]
        if missing_files:
            joined = "\n".join(missing_files)
            raise CommandError(f"Missing required JSONL files:\n{joined}")

        all_events: list[Event] = []
        imported_per_file: dict[str, int] = {}

        for filename in JSONL_FILES:
            file_path = persona_dir / filename
            imported_for_file = 0
            with file_path.open("r", encoding="utf-8") as handle:
                for line_number, raw_line in enumerate(handle, start=1):
                    if limit_per_file and imported_for_file >= limit_per_file:
                        break

                    line = raw_line.strip()
                    if not line:
                        continue

                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise CommandError(f"Invalid JSON in {file_path}:{line_number}: {exc}") from exc

                    ts = record.get("ts")
                    text = record.get("text")
                    if not isinstance(ts, str) or not ts.strip():
                        raise CommandError(f"Missing/invalid 'ts' in {file_path}:{line_number}")
                    if not isinstance(text, str) or not text.strip():
                        raise CommandError(f"Missing/invalid 'text' in {file_path}:{line_number}")

                    occurred_at = parse_datetime(ts)
                    if occurred_at is None:
                        raise CommandError(f"Unparseable timestamp in {file_path}:{line_number}: {ts}")

                    source = _map_source(str(record.get("source", "")), filename)
                    tags = _normalize_tags(record.get("tags"))

                    emotion = None
                    intensity = None
                    reaction = None
                    outcome = None
                    if enrich:
                        emotion = _infer_emotion(tags, text) or "neutral"
                        intensity = _infer_intensity(text, emotion)
                        reaction = _infer_reaction(text)
                        outcome = _infer_outcome(text)

                    all_events.append(
                        Event(
                            occurred_at=occurred_at,
                            source=source,
                            text=text.strip(),
                            context_tags=tags,
                            people=[],
                            emotion=emotion,
                            intensity=intensity,
                            reaction=reaction,
                            outcome=outcome,
                        )
                    )
                    imported_for_file += 1

            imported_per_file[filename] = imported_for_file

        total = sum(imported_per_file.values())
        if dry_run:
            self.stdout.write(self.style.WARNING("Dry run only: no database changes were made."))
            self._print_summary(persona_dir, imported_per_file, total, replaced=False, enriched=enrich)
            return

        with transaction.atomic():
            if replace:
                PatternRun.objects.all().delete()
                Event.objects.all().delete()
            Event.objects.bulk_create(all_events, batch_size=500)

        self._print_summary(persona_dir, imported_per_file, total, replaced=replace, enriched=enrich)

    def _print_summary(
        self, persona_dir: Path, imported_per_file: dict[str, int], total: int, *, replaced: bool, enriched: bool
    ):
        self.stdout.write(self.style.SUCCESS(f"Imported {total} events from {persona_dir}"))
        if replaced:
            self.stdout.write("Replaced existing events and derived pattern runs.")
        if enriched:
            self.stdout.write("Applied deterministic enrichment for emotion/intensity/reaction/outcome.")
        for filename in JSONL_FILES:
            count = imported_per_file.get(filename, 0)
            self.stdout.write(f"  - {filename}: {count}")
