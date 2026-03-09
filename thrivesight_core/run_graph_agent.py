"""
Django management command for running the ThriveSight Graph Agent.

Usage:
    # Single detection cycle
    python manage.py run_graph_agent --once

    # Continuous mode (default interval: 1 hour)
    python manage.py run_graph_agent

    # Custom interval (seconds)
    python manage.py run_graph_agent --interval 1800

    # Verbose output
    python manage.py run_graph_agent --once --verbose
"""

from django.core.management.base import BaseCommand

from events_api.graph_agent import GraphAgent, DEFAULT_INTERVAL


class Command(BaseCommand):
    help = "Run the ThriveSight Graph Agent for background cluster detection and maintenance."

    def add_arguments(self, parser):
        parser.add_argument(
            "--once",
            action="store_true",
            help="Run a single detection cycle and exit",
        )
        parser.add_argument(
            "--interval",
            type=int,
            default=DEFAULT_INTERVAL,
            help=f"Seconds between cycles in continuous mode (default: {DEFAULT_INTERVAL})",
        )
        parser.add_argument(
            "--verbose",
            action="store_true",
            help="Enable verbose output",
        )

    def handle(self, *args, **options):
        verbose = options["verbose"]
        agent = GraphAgent(verbose=verbose)

        if options["once"]:
            self.stdout.write("Running single detection cycle...")
            results = agent.run_detection_cycle()

            self.stdout.write(self.style.SUCCESS(
                f"\nCycle complete in {results['duration_seconds']}s:"
            ))
            self.stdout.write(f"  New clusters: {results['new_clusters']}")
            self.stdout.write(f"  Trajectories updated: {results['trajectories_updated']}")
            self.stdout.write(f"  Clusters dissolved: {results['clusters_dissolved']}")
            self.stdout.write(f"  Pending insights: {results['pending_insights_created']}")
            self.stdout.write(f"  Insights pruned: {results['insights_pruned']}")
            self.stdout.write(f"  Embeddings computed: {results['embeddings_computed']}")

            if results.get("errors"):
                self.stdout.write(self.style.WARNING(
                    f"  Errors: {len(results['errors'])}"
                ))
                for err in results["errors"]:
                    self.stdout.write(f"    - {err}")
        else:
            interval = options["interval"]
            self.stdout.write(
                f"Starting graph agent in continuous mode "
                f"(interval: {interval}s). Press Ctrl+C to stop."
            )
            try:
                agent.run_continuous(interval=interval)
            except KeyboardInterrupt:
                self.stdout.write(self.style.SUCCESS("\nGraph agent stopped."))
