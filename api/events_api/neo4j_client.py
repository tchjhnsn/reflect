import os
from functools import lru_cache
from urllib.parse import urlparse

from django.conf import settings
from neo4j import GraphDatabase


def _build_driver_uri(raw_url: str) -> tuple[str, tuple[str | None, str | None]]:
    parsed = urlparse(raw_url)
    uri = f"{parsed.scheme}://{parsed.hostname}"
    if parsed.port:
        uri = f"{uri}:{parsed.port}"
    return uri, (parsed.username, parsed.password)


def _get_connection_settings() -> tuple[str, tuple[str | None, str | None], str | None]:
    aura_uri = os.getenv("NEO4J_URI", "").strip()
    aura_username = os.getenv("NEO4J_USERNAME", "").strip()
    aura_password = os.getenv("NEO4J_PASSWORD", "").strip()
    aura_database = os.getenv("NEO4J_DATABASE", "").strip() or None

    if aura_uri and aura_username and aura_password:
        return aura_uri, (aura_username, aura_password), aura_database

    uri, auth = _build_driver_uri(settings.NEOMODEL_NEO4J_BOLT_URL)
    fallback_database = getattr(settings, "NEO4J_DATABASE", None) or None
    return uri, auth, fallback_database


@lru_cache(maxsize=1)
def get_driver():
    uri, auth, _ = _get_connection_settings()
    return GraphDatabase.driver(uri, auth=auth)


def cypher_query(query: str, params: dict | None = None) -> tuple[list[list], list[str]]:
    _, _, database = _get_connection_settings()
    session_kwargs = {"database": database} if database else {}
    with get_driver().session(**session_kwargs) as session:
        result = session.run(query, parameters=params or {})
        columns = list(result.keys())
        rows = [list(record.values()) for record in result]
    return rows, columns
