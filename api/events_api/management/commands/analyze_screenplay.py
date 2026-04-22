"""
Management command to parse and analyze a screenplay file.

Usage:
    python manage.py analyze_screenplay \
        --file scripts/spider_man.txt \
        --title "Spider-Man" \
        --character "PETER" \
        --description "A shy high school student who gains spider powers" \
        --known-chars "PETER,MARY JANE,HARRY,NORMAN,FLASH,AUNT MAY,UNCLE BEN" \
        --output datasets/peter_parker.json

    # Parse only (no LLM analysis):
    python manage.py analyze_screenplay \
        --file scripts/spider_man.txt \
        --title "Spider-Man" \
        --parse-only

    # Specific lenses only:
    python manage.py analyze_screenplay \
        --file scripts/spider_man.txt \
        --title "Spider-Man" \
        --character "PETER" \
        --lenses dialogue,plot,context
"""

import json
import os
import sys
import time

from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = "Parse and analyze a screenplay file through the multi-lens character analysis pipeline."

    def add_arguments(self, parser):
        parser.add_argument(
            "--file", "-f",
            required=True,
            help="Path to the screenplay text file",
        )
        parser.add_argument(
            "--title", "-t",
            default="Untitled",
            help="Film title (default: Untitled)",
        )
        parser.add_argument(
            "--character", "-c",
            default="",
            help="Focal character name (required unless --parse-only)",
        )
        parser.add_argument(
            "--description", "-d",
            default="",
            help="Brief character description",
        )
        parser.add_argument(
            "--known-chars",
            default="",
            help="Comma-separated list of known character names",
        )
        parser.add_argument(
            "--lenses",
            default="",
            help="Comma-separated list of lenses (dialogue,plot,psychology,relational,context). Default: all",
        )
        parser.add_argument(
            "--output", "-o",
            default="",
            help="Output JSON file path. Default: stdout",
        )
        parser.add_argument(
            "--parse-only",
            action="store_true",
            help="Only parse the screenplay (no LLM analysis)",
        )

    def handle(self, *args, **options):
        filepath = options["file"]
        title = options["title"]
        character = options["character"]
        description = options["description"]
        known_chars_str = options["known_chars"]
        lenses_str = options["lenses"]
        output_path = options["output"]
        parse_only = options["parse_only"]

        # Validate inputs
        if not os.path.exists(filepath):
            self.stderr.write(self.style.ERROR(f"File not found: {filepath}"))
            sys.exit(1)

        if not parse_only and not character:
            self.stderr.write(self.style.ERROR("--character is required unless --parse-only is set"))
            sys.exit(1)

        # Read the screenplay
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            raw_text = f.read()

        self.stdout.write(f"Read {len(raw_text)} characters from {filepath}")

        known_characters = [c.strip() for c in known_chars_str.split(",") if c.strip()]
        lenses = [l.strip() for l in lenses_str.split(",") if l.strip()] or None

        start_time = time.time()

        if parse_only:
            # Parse only
            from events_api.screenplay_parser import ScreenplayParser

            parser = ScreenplayParser(known_characters=known_characters)
            result = parser.parse(raw_text, film_title=title)

            elapsed = time.time() - start_time
            self.stdout.write(self.style.SUCCESS(
                f"\nParsed in {elapsed:.1f}s:\n"
                f"  Scenes: {result['total_scenes']}\n"
                f"  Dialogue turns: {result['total_dialogue_turns']}\n"
                f"  Characters: {', '.join(result['characters'][:15])}\n"
                f"  Scenes with dialogue: {result['parse_metadata']['scenes_with_dialogue']}"
            ))

            # Print scene roster
            self.stdout.write("\nScene Roster:")
            for scene in result["scenes"]:
                chars = ", ".join(scene["characters_present"][:5])
                witnesses = ", ".join(scene["witnesses"][:3])
                turns = len(scene["dialogue_turns"])
                line = f"  #{scene['scene_number']:3d} | {scene['slugline'][:60]:60s} | {turns:3d} turns | {chars}"
                if witnesses:
                    line += f" | witnesses: {witnesses}"
                self.stdout.write(line)

        else:
            # Full multi-lens analysis
            from events_api.character_analyzer import build_character_dataset

            self.stdout.write(f"Running multi-lens analysis for {character}...")
            if lenses:
                self.stdout.write(f"  Lenses: {', '.join(lenses)}")

            result = build_character_dataset(
                raw_screenplay=raw_text,
                film_title=title,
                focal_character=character,
                character_description=description,
                known_characters=known_characters,
                lenses=lenses,
            )

            elapsed = time.time() - start_time
            meta = result.get("analysis_metadata", {})
            self.stdout.write(self.style.SUCCESS(
                f"\nAnalysis complete in {elapsed:.1f}s:\n"
                f"  Scenes analyzed: {meta.get('total_scenes_analyzed', 0)}\n"
                f"  Dialogue turns: {meta.get('total_dialogue_turns', 0)}\n"
                f"  Lenses applied: {', '.join(meta.get('lenses_applied', []))}\n"
                f"  Psychology motivations: {len(result.get('psychology', {}).get('core_motivations', []))}\n"
                f"  Relationships tracked: {len(result.get('relationships', {}))}\n"
                f"  Arc turning points: {len(result.get('arc', {}).get('turning_points', []))}\n"
                f"  Build-up threads: {len(result.get('buildup_threads', []))}"
            ))

        # Output
        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, default=str)
            self.stdout.write(self.style.SUCCESS(f"\nSaved to {output_path}"))
        else:
            self.stdout.write("\n" + json.dumps(result, indent=2, default=str))
