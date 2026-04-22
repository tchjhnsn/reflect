"""
ThriveSight Coordinate System — Signal Address Parsing and Resolution.

The Signal Address System uses four-dimensional coordinates to locate
emotional moments in behavioral space: SA(context, person, action, temporal).

This module provides:
- SA string parsing and construction
- Wildcard detection and resolution completeness scoring
- Coordinate overlap comparison between two signal addresses
- Hierarchy traversal utilities for coordinate nodes
"""

import re
from typing import Optional


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

COORDINATE_NAMES = ["context", "person", "action", "temporal"]
WILDCARD = "*"

# Regex: SA(val1, val2, val3, val4) optionally followed by |emotion
_SA_PATTERN = re.compile(
    r"^\s*SA\s*\(\s*"
    r"([^,]+?)\s*,\s*"
    r"([^,]+?)\s*,\s*"
    r"([^,]+?)\s*,\s*"
    r"([^)]+?)\s*"
    r"\)\s*"
    r"(?:\|\s*(.+?)\s*)?$"
)


# ──────────────────────────────────────────────────────────────────────────────
# Parsing
# ──────────────────────────────────────────────────────────────────────────────


def parse_signal_address(sa_string: str) -> dict:
    """
    Parse a signal address string into its coordinate components.

    Args:
        sa_string: A string like "SA(work, manager, dismissal, monday)"
                   or "SA(work, manager, dismissal, monday)|anger"
                   or "SA(work, *, dismissal, *)"

    Returns:
        dict with keys: context, person, action, temporal, emotion, wildcards

    Raises:
        ValueError: If the string doesn't match the SA format.
    """
    if not sa_string or not sa_string.strip():
        raise ValueError("Signal address string cannot be empty")

    match = _SA_PATTERN.match(sa_string)
    if not match:
        raise ValueError(
            f"Invalid signal address format: '{sa_string}'. "
            f"Expected: SA(context, person, action, temporal)[|emotion]"
        )

    context = match.group(1).strip()
    person = match.group(2).strip()
    action = match.group(3).strip()
    temporal = match.group(4).strip()
    emotion = match.group(5).strip() if match.group(5) else None

    coords = {
        "context": context,
        "person": person,
        "action": action,
        "temporal": temporal,
    }

    wildcards = [name for name in COORDINATE_NAMES if coords[name] == WILDCARD]

    return {
        "context": context,
        "person": person,
        "action": action,
        "temporal": temporal,
        "emotion": emotion,
        "wildcards": wildcards,
    }


def build_signal_address(
    context: str = WILDCARD,
    person: str = WILDCARD,
    action: str = WILDCARD,
    temporal: str = WILDCARD,
    emotion: Optional[str] = None,
) -> str:
    """
    Build a signal address string from coordinate components.

    Args:
        context: Context coordinate value (default: wildcard).
        person: Person coordinate value (default: wildcard).
        action: Action coordinate value (default: wildcard).
        temporal: Temporal coordinate value (default: wildcard).
        emotion: Optional emotion suffix.

    Returns:
        Signal address string like "SA(work, manager, dismissal, monday)|anger".
    """
    sa = f"SA({context}, {person}, {action}, {temporal})"
    if emotion:
        sa += f"|{emotion}"
    return sa


# ──────────────────────────────────────────────────────────────────────────────
# Wildcard Detection
# ──────────────────────────────────────────────────────────────────────────────


def detect_wildcards(parsed_address: dict) -> list:
    """
    Identify which coordinates are wildcards in a parsed signal address.

    Args:
        parsed_address: Dict from parse_signal_address().

    Returns:
        List of coordinate names that are wildcards.
    """
    return [
        name
        for name in COORDINATE_NAMES
        if parsed_address.get(name) == WILDCARD
    ]


def is_fully_resolved(parsed_address: dict) -> bool:
    """
    Check if a parsed signal address has no wildcards.

    Args:
        parsed_address: Dict from parse_signal_address().

    Returns:
        True if all four coordinates are resolved (non-wildcard).
    """
    return len(detect_wildcards(parsed_address)) == 0


def resolution_completeness(parsed_address: dict) -> float:
    """
    Compute what fraction of coordinates are resolved (non-wildcard).

    Args:
        parsed_address: Dict from parse_signal_address().

    Returns:
        Float between 0.0 (all wildcards) and 1.0 (fully resolved).
    """
    wildcard_count = len(detect_wildcards(parsed_address))
    return (len(COORDINATE_NAMES) - wildcard_count) / len(COORDINATE_NAMES)


# ──────────────────────────────────────────────────────────────────────────────
# Coordinate Overlap
# ──────────────────────────────────────────────────────────────────────────────


def coordinate_overlap(parsed_a: dict, parsed_b: dict) -> dict:
    """
    Compare two parsed signal addresses and determine coordinate overlap.

    Wildcards in either address are excluded from both shared and divergent
    counts — they are tracked separately.

    Args:
        parsed_a: Dict from parse_signal_address().
        parsed_b: Dict from parse_signal_address().

    Returns:
        dict with keys:
            - shared: list of coordinate names with matching values
            - divergent: list of coordinate names with different values
            - wildcards: list of coordinate names where either side is wildcard
            - overlap_score: float between 0.0 and 1.0
    """
    shared = []
    divergent = []
    wildcards = []

    for name in COORDINATE_NAMES:
        val_a = parsed_a.get(name, WILDCARD)
        val_b = parsed_b.get(name, WILDCARD)

        if val_a == WILDCARD or val_b == WILDCARD:
            wildcards.append(name)
        elif val_a.lower() == val_b.lower():
            shared.append(name)
        else:
            divergent.append(name)

    # Overlap score is based on non-wildcard coordinates only
    comparable = len(shared) + len(divergent)
    if comparable == 0:
        overlap_score = 0.0
    else:
        overlap_score = len(shared) / comparable

    return {
        "shared": shared,
        "divergent": divergent,
        "wildcards": wildcards,
        "overlap_score": overlap_score,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Hierarchy Traversal (graph-dependent utilities)
# ──────────────────────────────────────────────────────────────────────────────


def resolve_coordinate(coordinate_name: str, value: str):
    """
    Resolve a coordinate string value to a graph node.

    This creates or finds the appropriate ContextNode, ActionNode,
    TemporalNode, or Person node. If the node doesn't exist, it is
    created at level 0 (root).

    Args:
        coordinate_name: One of "context", "person", "action", "temporal".
        value: The coordinate value string (e.g., "work", "manager").

    Returns:
        The resolved graph node, or None if the value is a wildcard.
    """
    if value == WILDCARD:
        return None

    # Import here to avoid circular imports with graph_models
    from events_api.graph_models import (
        ActionNode,
        ContextNode,
        Person,
        TemporalNode,
    )

    node_class_map = {
        "context": ContextNode,
        "action": ActionNode,
        "temporal": TemporalNode,
        "person": Person,
    }

    node_class = node_class_map.get(coordinate_name)
    if not node_class:
        raise ValueError(f"Unknown coordinate name: {coordinate_name}")

    # Find existing or create new
    normalized = value.lower().strip()
    existing = node_class.nodes.filter(name=normalized)
    if existing:
        return existing[0]

    # Create new root-level node
    node = node_class(name=normalized)
    if hasattr(node, "level"):
        node.level = 0
    node.save()
    return node


def resolve_all_coordinates(parsed_address: dict) -> dict:
    """
    Resolve all non-wildcard coordinates to graph nodes.

    Args:
        parsed_address: Dict from parse_signal_address().

    Returns:
        Dict mapping coordinate names to resolved graph nodes (or None for wildcards).
    """
    resolved = {}
    for name in COORDINATE_NAMES:
        value = parsed_address.get(name, WILDCARD)
        resolved[name] = resolve_coordinate(name, value)
    return resolved


def find_hierarchy_ancestors(node) -> list:
    """
    Walk up the SUBCATEGORY_OF hierarchy from a coordinate node.

    Args:
        node: A ContextNode, ActionNode, TemporalNode, or Person node.

    Returns:
        List of ancestor nodes from immediate parent to root.
    """
    ancestors = []
    current = node
    visited = set()

    while current:
        # Guard against cycles
        if current.uid in visited:
            break
        visited.add(current.uid)

        # Check for parent relationship
        parent_rel = getattr(current, "parent", None) or getattr(
            current, "parent_role", None
        )
        if parent_rel:
            parents = list(parent_rel.all())
            if parents:
                current = parents[0]
                ancestors.append(current)
            else:
                break
        else:
            break

    return ancestors


def find_hierarchy_descendants(node, max_depth: int = 5) -> list:
    """
    Walk down the SUBCATEGORY_OF hierarchy from a coordinate node.

    Args:
        node: A ContextNode, ActionNode, TemporalNode, or Person node.
        max_depth: Maximum depth to traverse (default: 5).

    Returns:
        List of descendant nodes (breadth-first).
    """
    descendants = []
    queue = [(node, 0)]
    visited = {node.uid}

    while queue:
        current, depth = queue.pop(0)
        if depth >= max_depth:
            continue

        children_rel = getattr(current, "children", None) or getattr(
            current, "child_roles", None
        )
        if children_rel:
            for child in children_rel.all():
                if child.uid not in visited:
                    visited.add(child.uid)
                    descendants.append(child)
                    queue.append((child, depth + 1))

    return descendants


# ──────────────────────────────────────────────────────────────────────────────
# CoordinateSystem class (high-level API)
# ──────────────────────────────────────────────────────────────────────────────


class CoordinateSystem:
    """
    High-level interface for Signal Address System coordinate operations.

    Wraps the module-level functions into a class for use in the signal
    engine and context assembly layer.
    """

    def parse(self, sa_string: str) -> dict:
        """Parse a signal address string."""
        return parse_signal_address(sa_string)

    def build(self, **kwargs) -> str:
        """Build a signal address string from components."""
        return build_signal_address(**kwargs)

    def wildcards(self, parsed: dict) -> list:
        """Get wildcard coordinates from a parsed address."""
        return detect_wildcards(parsed)

    def overlap(self, parsed_a: dict, parsed_b: dict) -> dict:
        """Compare two parsed addresses."""
        return coordinate_overlap(parsed_a, parsed_b)

    def resolve(self, coordinate_name: str, value: str):
        """Resolve a coordinate value to a graph node."""
        return resolve_coordinate(coordinate_name, value)

    def resolve_all(self, parsed: dict) -> dict:
        """Resolve all coordinates to graph nodes."""
        return resolve_all_coordinates(parsed)

    def ancestors(self, node) -> list:
        """Get hierarchy ancestors of a node."""
        return find_hierarchy_ancestors(node)

    def descendants(self, node, max_depth: int = 5) -> list:
        """Get hierarchy descendants of a node."""
        return find_hierarchy_descendants(node, max_depth)


# Module-level convenience alias
parse_sa = parse_signal_address
