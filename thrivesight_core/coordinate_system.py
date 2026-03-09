"""
ThriveSight Coordinate System — SA parsing, wildcard detection, hierarchy traversal.

The Signal Address System uses four coordinates to locate emotional moments:
    SA(context, person, action, temporal)

This module handles:
- Parsing SA string representations into structured dicts
- Detecting wildcard (*) coordinates that need exploration
- Resolving string coordinate values to graph nodes
- Traversing and building coordinate hierarchies
"""

import re
from typing import Optional

from neomodel import db

from .graph_models import ActionNode, ContextNode, Person, TemporalNode

# ──────────────────────────────────────────────────────────────────────────────
# SA PARSING
# ──────────────────────────────────────────────────────────────────────────────

# Matches SA(context, person, action, temporal) with optional emotion suffix |emotion
SA_PATTERN = re.compile(
    r"^SA\(\s*"
    r"(?P<context>[^,]+?)\s*,\s*"
    r"(?P<person>[^,]+?)\s*,\s*"
    r"(?P<action>[^,]+?)\s*,\s*"
    r"(?P<temporal>[^)]+?)\s*"
    r"\)"
    r"(?:\|(?P<emotion>\w+))?"
    r"$"
)

WILDCARD = "*"

COORDINATE_NAMES = ["context", "person", "action", "temporal"]

# Coordinate type → neomodel class mapping
COORDINATE_NODE_CLASSES = {
    "context": ContextNode,
    "action": ActionNode,
    "temporal": TemporalNode,
    "person": Person,
}


def parse_signal_address(sa_string: str) -> dict:
    """
    Parse a signal address string into a structured dict.

    Args:
        sa_string: Signal address string, e.g., "SA(work, manager, dismissal, monday)|anger"

    Returns:
        dict with keys: context, person, action, temporal, emotion (optional),
        wildcards (list of coordinate names with *), raw (original string)

    Raises:
        ValueError: If the string doesn't match the SA pattern.

    Examples:
        >>> parse_signal_address("SA(work, manager, dismissal, monday)|anger")
        {
            "context": "work",
            "person": "manager",
            "action": "dismissal",
            "temporal": "monday",
            "emotion": "anger",
            "wildcards": [],
            "raw": "SA(work, manager, dismissal, monday)|anger"
        }

        >>> parse_signal_address("SA(work, *, dismissal, *)")
        {
            "context": "work",
            "person": "*",
            "action": "dismissal",
            "temporal": "*",
            "emotion": None,
            "wildcards": ["person", "temporal"],
            "raw": "SA(work, *, dismissal, *)"
        }
    """
    match = SA_PATTERN.match(sa_string.strip())
    if not match:
        raise ValueError(f"Invalid signal address format: {sa_string}")

    groups = match.groupdict()
    wildcards = [
        coord for coord in COORDINATE_NAMES
        if groups.get(coord, "").strip() == WILDCARD
    ]

    return {
        "context": groups["context"].strip(),
        "person": groups["person"].strip(),
        "action": groups["action"].strip(),
        "temporal": groups["temporal"].strip(),
        "emotion": groups.get("emotion"),
        "wildcards": wildcards,
        "raw": sa_string.strip(),
    }


def build_signal_address(
    context: str = WILDCARD,
    person: str = WILDCARD,
    action: str = WILDCARD,
    temporal: str = WILDCARD,
    emotion: str = None,
) -> str:
    """
    Build a signal address string from components.

    Args:
        context: Context coordinate value (default: "*")
        person: Person coordinate value (default: "*")
        action: Action coordinate value (default: "*")
        temporal: Temporal coordinate value (default: "*")
        emotion: Optional emotion suffix

    Returns:
        Formatted SA string, e.g., "SA(work, manager, dismissal, monday)|anger"
    """
    sa = f"SA({context}, {person}, {action}, {temporal})"
    if emotion:
        sa += f"|{emotion}"
    return sa


# ──────────────────────────────────────────────────────────────────────────────
# WILDCARD DETECTION
# ──────────────────────────────────────────────────────────────────────────────


def detect_wildcards(parsed_sa: dict) -> list:
    """
    Identify which coordinates in a parsed SA are wildcards.

    Wildcards represent exploration targets — areas where the system
    needs more context or where the user has intentionally left a
    dimension broad.

    Args:
        parsed_sa: Output from parse_signal_address()

    Returns:
        List of coordinate names that are wildcards.
    """
    return [
        coord for coord in COORDINATE_NAMES
        if parsed_sa.get(coord, "").strip() == WILDCARD
    ]


def is_fully_resolved(parsed_sa: dict) -> bool:
    """
    Check if all coordinates in a parsed SA have non-wildcard values.

    A fully resolved SA has no exploration targets remaining.
    """
    return len(detect_wildcards(parsed_sa)) == 0


def resolution_completeness(parsed_sa: dict) -> float:
    """
    Calculate how complete the signal address resolution is.

    Returns:
        Float from 0.0 (all wildcards) to 1.0 (fully resolved).
    """
    wildcards = detect_wildcards(parsed_sa)
    return 1.0 - (len(wildcards) / len(COORDINATE_NAMES))


# ──────────────────────────────────────────────────────────────────────────────
# COORDINATE RESOLUTION — Match strings to graph nodes
# ──────────────────────────────────────────────────────────────────────────────


def resolve_context(value: str) -> Optional[ContextNode]:
    """
    Find or create a ContextNode matching the given value.

    Searches existing nodes first (case-insensitive). If not found,
    creates a new node at level 0 (root).
    """
    if value == WILDCARD:
        return None

    normalized = value.strip().lower()
    existing = ContextNode.nodes.filter(name=normalized)
    if existing:
        return existing[0]

    # Create new context node
    node = ContextNode(name=normalized, level=0)
    node.save()
    return node


def resolve_action(value: str) -> Optional[ActionNode]:
    """
    Find or create an ActionNode matching the given value.
    """
    if value == WILDCARD:
        return None

    normalized = value.strip().lower()
    existing = ActionNode.nodes.filter(name=normalized)
    if existing:
        return existing[0]

    node = ActionNode(name=normalized, level=0)
    node.save()
    return node


def resolve_temporal(value: str) -> Optional[TemporalNode]:
    """
    Find or create a TemporalNode matching the given value.

    Attempts to detect temporal type from the value:
    - Specific dates → "specific"
    - Day names, recurring patterns → "cyclical"
    - Ranges → "period"
    """
    if value == WILDCARD:
        return None

    normalized = value.strip().lower()
    existing = TemporalNode.nodes.filter(name=normalized)
    if existing:
        return existing[0]

    # Detect temporal type
    cyclical_patterns = [
        "monday", "tuesday", "wednesday", "thursday", "friday",
        "saturday", "sunday", "morning", "afternoon", "evening",
        "night", "weekly", "monthly", "quarterly", "annually",
        "daily", "weekend", "weekday",
    ]
    temporal_type = "cyclical" if any(p in normalized for p in cyclical_patterns) else "specific"

    node = TemporalNode(name=normalized, temporal_type=temporal_type)
    node.save()
    return node


def resolve_person(value: str) -> Optional[Person]:
    """
    Find or create a Person node matching the given value.

    For role-based references (e.g., "manager", "colleague"), creates
    the person with role_type="role_category". For specific names,
    uses role_type="specific_person".
    """
    if value == WILDCARD:
        return None

    normalized = value.strip().lower()
    existing = Person.nodes.filter(name=normalized)
    if existing:
        return existing[0]

    # Detect if this is a role reference or specific person
    role_keywords = [
        "manager", "boss", "supervisor", "colleague", "coworker",
        "co-worker", "friend", "partner", "spouse", "parent",
        "mother", "father", "sibling", "brother", "sister",
        "therapist", "doctor", "teacher", "mentor", "team_lead",
    ]
    is_role = any(kw in normalized for kw in role_keywords)

    node = Person(
        name=normalized,
        role_type="role_category" if is_role else "specific_person",
        role=normalized if is_role else None,
    )
    node.save()
    return node


def resolve_coordinates(parsed_sa: dict) -> dict:
    """
    Resolve all non-wildcard coordinates in a parsed SA to graph nodes.

    Args:
        parsed_sa: Output from parse_signal_address()

    Returns:
        dict mapping coordinate names to their resolved graph nodes (or None for wildcards).
    """
    return {
        "context": resolve_context(parsed_sa["context"]),
        "person": resolve_person(parsed_sa["person"]),
        "action": resolve_action(parsed_sa["action"]),
        "temporal": resolve_temporal(parsed_sa["temporal"]),
    }


# ──────────────────────────────────────────────────────────────────────────────
# HIERARCHY TRAVERSAL
# ──────────────────────────────────────────────────────────────────────────────


def get_hierarchy_path(node) -> list:
    """
    Get the full hierarchy path from a node to its root.

    Traverses SUBCATEGORY_OF relationships upward until no parent is found.

    Args:
        node: A ContextNode, ActionNode, TemporalNode, or Person node.

    Returns:
        List of nodes from root to the given node, e.g.,
        [authority_figures, managers, my_current_manager, sarah]
    """
    path = [node]
    current = node

    # Traverse up through parents (max depth 10 to prevent infinite loops)
    for _ in range(10):
        parents = current.parent.all() if hasattr(current, "parent") else []
        if hasattr(current, "parent_role"):
            parents = current.parent_role.all()

        if not parents:
            break
        current = parents[0]
        path.append(current)

    path.reverse()  # Root first
    return path


def get_hierarchy_children(node) -> list:
    """
    Get all direct children of a hierarchy node.

    Args:
        node: A ContextNode, ActionNode, TemporalNode, or Person node.

    Returns:
        List of child nodes.
    """
    if hasattr(node, "children"):
        return list(node.children.all())
    if hasattr(node, "child_roles"):
        return list(node.child_roles.all())
    return []


def add_to_hierarchy(child_node, parent_node):
    """
    Establish a parent-child hierarchy relationship.

    Sets the child's level to parent_level + 1 and creates the
    SUBCATEGORY_OF edge.

    Args:
        child_node: The node to be nested under the parent.
        parent_node: The parent node in the hierarchy.
    """
    # Set hierarchy level
    parent_level = getattr(parent_node, "level", 0) or 0
    if hasattr(child_node, "level"):
        child_node.level = parent_level + 1
        child_node.save()

    # Create relationship
    if hasattr(child_node, "parent"):
        child_node.parent.connect(parent_node)
    elif hasattr(child_node, "parent_role"):
        child_node.parent_role.connect(parent_node)


def find_or_create_in_hierarchy(
    name: str,
    coordinate_type: str,
    parent_name: str = None,
) -> object:
    """
    Find a node in the coordinate hierarchy, or create it.

    If parent_name is provided, the node is created as a child of the
    specified parent. If not, it's created as a root node.

    Args:
        name: The coordinate value (e.g., "standup")
        coordinate_type: "context" | "action" | "temporal" | "person"
        parent_name: Optional parent node name for hierarchy placement

    Returns:
        The found or created node.
    """
    node_class = COORDINATE_NODE_CLASSES.get(coordinate_type)
    if not node_class:
        raise ValueError(f"Unknown coordinate type: {coordinate_type}")

    normalized = name.strip().lower()

    # Check if exists
    if coordinate_type == "person":
        existing = Person.nodes.filter(name=normalized)
    else:
        existing = node_class.nodes.filter(name=normalized)

    if existing:
        return existing[0]

    # Create new node
    if coordinate_type == "person":
        node = Person(name=normalized, role_type="specific_person")
    else:
        node = node_class(name=normalized)

    node.save()

    # Link to parent if specified
    if parent_name:
        parent_normalized = parent_name.strip().lower()
        if coordinate_type == "person":
            parents = Person.nodes.filter(name=parent_normalized)
        else:
            parents = node_class.nodes.filter(name=parent_normalized)

        if parents:
            add_to_hierarchy(node, parents[0])

    return node


# ──────────────────────────────────────────────────────────────────────────────
# SIGNAL ADDRESS COMPARISON
# ──────────────────────────────────────────────────────────────────────────────


def coordinate_overlap(sa1: dict, sa2: dict) -> dict:
    """
    Compare two parsed signal addresses and identify overlapping coordinates.

    Args:
        sa1: First parsed SA (from parse_signal_address)
        sa2: Second parsed SA

    Returns:
        dict with:
        - shared: list of coordinate names that match (non-wildcard)
        - divergent: list of coordinate names that differ (non-wildcard)
        - wildcards: list of coordinate names where either has a wildcard
        - overlap_score: float 0.0-1.0 indicating similarity
    """
    shared = []
    divergent = []
    wildcards = []

    for coord in COORDINATE_NAMES:
        v1 = sa1.get(coord, WILDCARD)
        v2 = sa2.get(coord, WILDCARD)

        if v1 == WILDCARD or v2 == WILDCARD:
            wildcards.append(coord)
        elif v1.lower() == v2.lower():
            shared.append(coord)
        else:
            divergent.append(coord)

    resolved_count = len(shared) + len(divergent)
    overlap_score = len(shared) / resolved_count if resolved_count > 0 else 0.0

    return {
        "shared": shared,
        "divergent": divergent,
        "wildcards": wildcards,
        "overlap_score": overlap_score,
    }
