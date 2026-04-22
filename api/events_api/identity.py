"""
Helpers for canonicalizing user identity in the graph layer.

UserProfile is the only canonical representation of the authenticated user.
Self-like aliases should never create Person nodes.
"""


SELF_ALIASES = {
    "self",
    "user",
    "me",
    "myself",
    "i",
}


def build_self_alias_names(*, current_username: str | None = None) -> list[str]:
    """Return normalized self-like aliases to collapse onto UserProfile."""
    aliases = set(SELF_ALIASES)
    username = " ".join(str(current_username or "").strip().split()).lower()
    if username:
        aliases.add(username)
    return sorted(aliases)


def normalize_graph_person_name(name: str | None, *, current_username: str | None = None) -> str | None:
    """
    Normalize a person label before writing it to the graph.

    Returns:
        - ``None`` for self-references that should map to UserProfile
        - normalized non-empty person name otherwise
    """
    normalized = " ".join(str(name or "").strip().split())
    if not normalized:
        return None

    lowered = normalized.lower()
    if lowered in build_self_alias_names(current_username=current_username):
        return None

    return normalized
