import logging
from django.apps import AppConfig

logger = logging.getLogger(__name__)


class EventsApiConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "events_api"

    def ready(self):
        """Initialize graph models and Neomodel connection at app startup."""
        # Configure Neomodel connection (Neo4j AuraDB)
        try:
            from django.conf import settings
            from neomodel import config as neo_config

            bolt_url = settings.NEOMODEL_NEO4J_BOLT_URL
            neo_config.DATABASE_URL = bolt_url
            # Aura home-database routing works reliably when we let the driver
            # choose the database instead of forcing DATABASE_NAME.
            if hasattr(neo_config, "DATABASE_NAME"):
                neo_config.DATABASE_NAME = None
            logger.info(f"Neomodel configured with {bolt_url}")
        except Exception as e:
            logger.warning(f"Neomodel configuration failed (app will continue): {e}")

        # Import graph models to register them
        try:
            from .graph_models import initialize_indexes
            logger.info("Graph models imported successfully")
            index_results = initialize_indexes()
            created_indexes = sum(1 for result in index_results if result["status"] == "created")
            logger.info(f"Neo4j indexes initialized ({created_indexes} statements applied)")
        except ImportError as e:
            logger.warning(f"Graph models import failed (optional): {e}")
        except Exception as e:
            logger.warning(f"Neo4j index initialization failed (app will continue): {e}")

        # Initialize seed categories
        try:
            from .conversation import SEED_CATEGORIES
            logger.info(f"Loaded {len(SEED_CATEGORIES)} seed trigger categories")
        except Exception as e:
            logger.warning(f"Seed categories initialization failed (optional): {e}")

        # Initialize seed contexts (V3 SA coordinate hierarchy)
        try:
            from .graph_models import SEED_CONTEXTS
            logger.info(f"Loaded {len(SEED_CONTEXTS)} seed context categories")
        except Exception as e:
            logger.warning(f"Seed contexts initialization failed (optional): {e}")
