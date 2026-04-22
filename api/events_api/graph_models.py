"""
ThriveSight Graph Schema V3 — Signal Address System Architecture.

V3 is a complete overhaul that removes screenplay-focused nodes (SourceWork,
MasterDocument, CharacterProfile, Scene, SignificantWord, Reframe) and rebuilds
the schema around the Signal Address System as an emotional epistemology engine.

Architecture Layers:
- Signal Address: SA(c, p, a, t) with multi-emotion payloads
- Coordinate Hierarchy: ContextNode, ActionNode, TemporalNode (first-class nodes)
- Participant Graph: Person nodes with roles (PARTICIPANT_IN relationships)
- Signal Clusters: First-class cluster nodes with trajectory tracking
- Reasoning Traces: Insight (LLM) and Reflection (user) nodes
- Observation Bias: Confidence/provenance on signals, bias flags
- Background Detection: PendingInsight nodes from graph agent

Key V3 changes over V2:
- Removed: SourceWork, MasterDocument, CharacterProfile, Scene, SignificantWord,
  Reframe, RelatesToRel, DirectedAtRel, AppearsInRel
- Signal node: multi-emotion (JSONProperty), confidence, provenance, embedding,
  observation bias flags, exploration geometry
- Person node: evolved with role hierarchy (SUBCATEGORY_OF)
- New: ContextNode, ActionNode, TemporalNode (coordinate hierarchy)
- New: Cluster with trajectory_history, confidence, user dispute support
- New: Insight, Reflection, PendingInsight (reasoning traces)
- New relationships: MEMBER_OF, PARTICIPANT_IN, IN_CONTEXT, INVOLVES_ACTION,
  AT_TIME, DERIVED_FROM, INFORMED_BY, RESPONDS_TO, AWAITING_REVIEW
"""

from neomodel import (
    ArrayProperty,
    BooleanProperty,
    DateTimeProperty,
    FloatProperty,
    IntegerProperty,
    JSONProperty,
    RelationshipFrom,
    RelationshipTo,
    StringProperty,
    StructuredNode,
    StructuredRel,
    UniqueIdProperty,
)


# ──────────────────────────────────────────────────────────────────────────────
# STRUCTURED RELATIONSHIPS (Enriched Edges)
# ──────────────────────────────────────────────────────────────────────────────


class ParticipantInRel(StructuredRel):
    """
    PARTICIPANT_IN edge from Person to Signal.

    Each person in a signal has a role describing their contribution to the
    emotional outcome. Group dynamics are modeled through multiple
    PARTICIPANT_IN edges on a single signal.

    Roles:
    - subject: The user themselves (the one experiencing the emotion)
    - primary_actor: The person whose action is the main trigger
    - amplifier: Someone whose reaction intensified the emotion (e.g., co-worker laughing)
    - witness: Present but not directly involved; their presence/silence may matter
    - mentioned: Referenced in the signal but not physically present
    """

    role = StringProperty(required=True)  # "subject" | "primary_actor" | "amplifier" | "witness" | "mentioned"
    confidence = FloatProperty(default=1.0)  # How sure we are of this attribution (0.0-1.0)
    attributed_by = StringProperty(default="llm")  # "user" | "llm" | "system"


class MemberOfRel(StructuredRel):
    """
    MEMBER_OF edge from Signal to Cluster.

    Tracks whether membership is currently active and when it was established.
    Preserved even after dissolution for historical trajectory analysis.
    """

    active = BooleanProperty(default=True)
    joined_at = DateTimeProperty(default_now=True)
    deactivated_at = DateTimeProperty()


class DerivedFromRel(StructuredRel):
    """
    DERIVED_FROM edge between Signals (child → parent).

    Tracks signal derivation — when exploring a signal produces a new,
    more specific signal. The derivation_type describes how the child
    was generated.
    """

    derivation_type = StringProperty(default="exploration")  # "exploration" | "decomposition" | "resolution" | "reframe"
    derived_at = DateTimeProperty(default_now=True)


class InformedByRel(StructuredRel):
    """
    INFORMED_BY edge from Insight to Signal or Cluster.

    Tracks which signals/clusters an LLM insight drew upon.
    """

    relevance = FloatProperty(default=1.0)  # How central this signal was to the insight


# ──────────────────────────────────────────────────────────────────────────────
# COORDINATE HIERARCHY NODES
# ──────────────────────────────────────────────────────────────────────────────
# Coordinates in the SA system are first-class graph nodes, not just strings.
# Each coordinate type has a hierarchy discoverable through exploration:
# e.g., authority_figures → managers → my_current_manager → Sarah


class ContextNode(StructuredNode):
    """
    First-class context coordinate in the Signal Address System.

    Represents the setting or environment where a signal occurs: work, home,
    social, health, etc. Context nodes form hierarchies:
    work → work/meetings → work/meetings/standup

    Relationships:
        - parent: SUBCATEGORY_OF → ContextNode (hierarchy)
        - children: SUBCATEGORY_OF ← ContextNode
        - signals: IN_CONTEXT ← Signal
    """

    uid = UniqueIdProperty()
    workspace_id = StringProperty()
    owner_user_id = IntegerProperty()
    name = StringProperty(required=True)
    description = StringProperty()
    level = IntegerProperty(default=0)  # Depth in hierarchy (0 = root)
    properties_json = JSONProperty()  # Accumulated properties from conversations
    created_at = DateTimeProperty(default_now=True)

    parent = RelationshipTo("ContextNode", "SUBCATEGORY_OF")
    children = RelationshipFrom("ContextNode", "SUBCATEGORY_OF")
    signals = RelationshipFrom("Signal", "IN_CONTEXT")


class ActionNode(StructuredNode):
    """
    First-class action coordinate in the Signal Address System.

    Represents what happened or was done: dismissal, praise, interruption, etc.
    Action nodes form hierarchies:
    communication → negative_communication → dismissal → public_dismissal

    Relationships:
        - parent: SUBCATEGORY_OF → ActionNode (hierarchy)
        - children: SUBCATEGORY_OF ← ActionNode
        - signals: INVOLVES_ACTION ← Signal
    """

    uid = UniqueIdProperty()
    workspace_id = StringProperty()
    owner_user_id = IntegerProperty()
    name = StringProperty(required=True)
    description = StringProperty()
    level = IntegerProperty(default=0)
    properties_json = JSONProperty()
    created_at = DateTimeProperty(default_now=True)

    parent = RelationshipTo("ActionNode", "SUBCATEGORY_OF")
    children = RelationshipFrom("ActionNode", "SUBCATEGORY_OF")
    signals = RelationshipFrom("Signal", "INVOLVES_ACTION")


class TemporalNode(StructuredNode):
    """
    First-class temporal coordinate in the Signal Address System.

    Represents when something occurred. Can be specific (March 8, 2026) or
    cyclical (Monday mornings, quarterly reviews, holidays). Temporal nodes
    capture recurring patterns as well as specific moments.

    Relationships:
        - parent: SUBCATEGORY_OF → TemporalNode (hierarchy)
        - children: SUBCATEGORY_OF ← TemporalNode
        - signals: AT_TIME ← Signal
    """

    uid = UniqueIdProperty()
    workspace_id = StringProperty()
    owner_user_id = IntegerProperty()
    name = StringProperty(required=True)  # "monday_morning", "march_2026", "quarterly_review"
    temporal_type = StringProperty(default="specific")  # "specific" | "cyclical" | "period" | "recurring"
    description = StringProperty()
    properties_json = JSONProperty()
    created_at = DateTimeProperty(default_now=True)

    parent = RelationshipTo("TemporalNode", "SUBCATEGORY_OF")
    children = RelationshipFrom("TemporalNode", "SUBCATEGORY_OF")
    signals = RelationshipFrom("Signal", "AT_TIME")


# ──────────────────────────────────────────────────────────────────────────────
# CORE NODES — Conversation, Person, Turn
# ──────────────────────────────────────────────────────────────────────────────


class Conversation(StructuredNode):
    """
    Container for a complete dialogue session.

    The primary unit of user interaction. Each conversation generates
    multiple turns, each of which may produce one or more signals.

    Relationships:
        - turns: CONTAINS → Turn
        - participants: INVOLVES → Person
        - data_source: IMPORTED_FROM → DataSource
    """

    uid = UniqueIdProperty()
    workspace_id = StringProperty()
    owner_user_id = IntegerProperty()
    conversation_id = StringProperty()
    title = StringProperty(required=True)
    created_at = DateTimeProperty(default_now=True)
    source_type = StringProperty(default="live")  # "live" | "import_chatgpt" | "import_claude" | "text"
    platform = StringProperty()  # "thrivesight" | "chatgpt" | "claude"
    turn_count = IntegerProperty(default=0)
    create_time = FloatProperty()  # Epoch timestamp for imports
    last_active = FloatProperty()

    turns = RelationshipTo("Turn", "CONTAINS")
    participants = RelationshipTo("Person", "INVOLVES")


class Person(StructuredNode):
    """
    A person mentioned in conversations — first-class coordinate node.

    Person nodes accumulate properties from every conversation in which
    they appear. They form hierarchies through roles:
    authority_figures → managers → my_current_manager → Sarah

    Relationships:
        - parent_role: SUBCATEGORY_OF → Person (role hierarchy)
        - child_roles: SUBCATEGORY_OF ← Person
        - signals: PARTICIPANT_IN → Signal (with ParticipantInRel)
        - turns: SPOKEN_BY ← Turn
        - conversations: INVOLVES ← Conversation
    """

    uid = UniqueIdProperty()
    workspace_id = StringProperty()
    owner_user_id = IntegerProperty()
    name = StringProperty(required=True)
    role = StringProperty()  # "manager" | "colleague" | "friend" | "family" | "self" | etc.
    role_type = StringProperty()  # "specific_person" | "role_category" | "abstract_category"
    description = StringProperty()
    properties_json = JSONProperty()  # Accumulated properties from all conversations
    created_at = DateTimeProperty(default_now=True)

    # Role hierarchy
    parent_role = RelationshipTo("Person", "SUBCATEGORY_OF")
    child_roles = RelationshipFrom("Person", "SUBCATEGORY_OF")

    # Signal participation with role attribution
    signals = RelationshipTo("Signal", "PARTICIPANT_IN", model=ParticipantInRel)

    # Turns spoken by this person
    turns = RelationshipFrom("Turn", "SPOKEN_BY")

    # Conversations involving this person
    conversations = RelationshipFrom("Conversation", "INVOLVES")


class Turn(StructuredNode):
    """
    An individual utterance within a conversation.

    The atomic unit of dialogue. Every signal traces back to one or more
    turns. In live conversations, each user message and AI response is a turn.

    Relationships:
        - next_turn: NEXT → Turn
        - speaker: SPOKEN_BY → Person
        - signals: PRODUCES → Signal
        - conversation: CONTAINS ← Conversation
    """

    uid = UniqueIdProperty()
    workspace_id = StringProperty()
    owner_user_id = IntegerProperty()
    text = StringProperty(required=True)
    turn_number = IntegerProperty(required=True)
    speaker_name = StringProperty()  # Denormalized for quick display
    role = StringProperty()  # "user" | "assistant" | "system"
    word_count = IntegerProperty()
    timestamp = DateTimeProperty()
    content_preview = StringProperty()  # Truncated version for list views
    has_question = BooleanProperty(default=False)
    create_time = FloatProperty()

    next_turn = RelationshipTo("Turn", "NEXT")
    speaker = RelationshipTo("Person", "SPOKEN_BY")
    signals = RelationshipTo("Signal", "PRODUCES")
    conversation = RelationshipFrom("Conversation", "CONTAINS")


# ──────────────────────────────────────────────────────────────────────────────
# SIGNAL ADDRESS SYSTEM — Core Signal Node
# ──────────────────────────────────────────────────────────────────────────────


class Signal(StructuredNode):
    """
    The core node of the Signal Address System.

    A signal represents an emotional moment located in four-dimensional
    behavioral space: SA(context, person, action, temporal). The emotional
    payload is structured — multiple emotions, each traceable to specific
    coordinates or participants.

    V3 changes from V2:
    - emotions: JSONProperty array (was single emotion + intensity)
    - confidence_score: observation bias tracking
    - provenance: how this signal was created
    - observation_bias_flags: detected bias types
    - embedding: semantic vector for lightweight similarity
    - exploration_geometry: geometric pattern of the exploration tree
    - signal_address: still present as a string representation for display

    Emotion array structure:
    [
        {
            "emotion": "anger",
            "intensity": 7.5,
            "source_coordinate": "person",  # Which SA dimension this emotion arises from
            "source_description": "manager's public dismissal",
            "confidence": 0.9
        },
        {
            "emotion": "shame",
            "intensity": 6.0,
            "source_coordinate": "context",  # The group setting amplified this
            "source_description": "happened in front of the whole team",
            "confidence": 0.7
        }
    ]

    Relationships:
        - context: IN_CONTEXT → ContextNode
        - action: INVOLVES_ACTION → ActionNode
        - temporal: AT_TIME → TemporalNode
        - participants: PARTICIPANT_IN ← Person (with ParticipantInRel)
        - parent_signal: DERIVED_FROM → Signal (exploration tree)
        - child_signals: DERIVED_FROM ← Signal
        - trigger: TRIGGERED_BY → TriggerAction
        - source_turns: PRODUCES ← Turn
        - clusters: MEMBER_OF → Cluster (with MemberOfRel)
        - insights: INFORMED_BY ← Insight
    """

    uid = UniqueIdProperty()
    workspace_id = StringProperty()
    owner_user_id = IntegerProperty()

    # Signal Address (display representation)
    signal_address = StringProperty()  # "SA(work, manager, dismissal, monday)" for display/search

    # Multi-emotion payload
    emotions = JSONProperty(default=[])  # Array of {emotion, intensity, source_coordinate, source_description, confidence}

    # V2 backward compatibility (single emotion — deprecated, use emotions[] instead)
    emotion = StringProperty()
    intensity = FloatProperty()
    reaction = StringProperty()

    # Observation Bias Layer
    confidence_score = FloatProperty(default=1.0)  # 0.0 (pure projection) to 1.0 (verified fact)
    provenance = StringProperty(default="user_stated")  # "user_stated" | "llm_inferred" | "derived" | "system_detected"
    observation_bias_flags = JSONProperty(default=[])  # ["projection", "rumination_amplification", "confirmation_bias", "narrative_construction"]

    # Exploration State
    is_resolved = BooleanProperty(default=False)  # Exploration reached resolution
    exploration_geometry = StringProperty()  # "circle" | "spiral_inward" | "spiral_outward" | "starburst" | "line" | "figure_eight"
    wildcard_coordinates = JSONProperty(default=[])  # ["context", "temporal"] — which dimensions are incomplete

    # Semantic Embedding
    embedding = JSONProperty()  # Vector for lightweight similarity (graph agent)

    # Metadata
    created_at = DateTimeProperty(default_now=True)
    raw_offset = FloatProperty()  # Position in original text (legacy support)

    # Coordinate relationships (first-class graph edges)
    context = RelationshipTo("ContextNode", "IN_CONTEXT")
    action = RelationshipTo("ActionNode", "INVOLVES_ACTION")
    temporal = RelationshipTo("TemporalNode", "AT_TIME")

    # Person relationships come through PARTICIPANT_IN (Person → Signal)
    participants = RelationshipFrom("Person", "PARTICIPANT_IN")

    # Signal derivation tree
    parent_signal = RelationshipTo("Signal", "DERIVED_FROM", model=DerivedFromRel)
    child_signals = RelationshipFrom("Signal", "DERIVED_FROM")

    # Trigger attribution (retained from V2)
    trigger = RelationshipTo("TriggerAction", "TRIGGERED_BY")

    # Source turns
    source_turns = RelationshipFrom("Turn", "PRODUCES")

    # Cluster membership
    clusters = RelationshipTo("Cluster", "MEMBER_OF", model=MemberOfRel)

    # Insights that reference this signal
    insights = RelationshipFrom("Insight", "INFORMED_BY")

    # Contract gap relationships — Emotion, Behavior, Outcome
    emotions_expressed = RelationshipTo("Emotion", "EXPRESSES_EMOTION")
    behaviors = RelationshipTo("Behavior", "SHOWS_BEHAVIOR")
    outcomes = RelationshipTo("Outcome", "LED_TO")


# ──────────────────────────────────────────────────────────────────────────────
# TRIGGER SYSTEM (retained from V2, evolved)
# ──────────────────────────────────────────────────────────────────────────────


class TriggerAction(StructuredNode):
    """
    What provoked a signal.

    Retained from V2 with minor evolution. Now linked to signals that may
    have multi-emotion payloads.
    """

    uid = UniqueIdProperty()
    workspace_id = StringProperty()
    owner_user_id = IntegerProperty()
    action_text = StringProperty(required=True)
    description = StringProperty()
    is_new_category = BooleanProperty(default=False)
    inferred_by = StringProperty(
        choices={"user": "user", "llm": "llm", "system": "system"}
    )

    category = RelationshipTo("TriggerCategory", "HAS_CATEGORY")


class TriggerCategory(StructuredNode):
    """
    First-class vocabulary node for trigger categories.

    Retained from V2. Categories transition from proposed to validated
    when usage_count >= 3.
    """

    uid = UniqueIdProperty()
    name = StringProperty(unique_index=True, required=True)
    description = StringProperty()
    is_system = BooleanProperty(default=False)
    is_proposed = BooleanProperty(default=True)
    usage_count = IntegerProperty(default=0)
    proposed_by = StringProperty(default="system")
    first_seen = DateTimeProperty(default_now=True)


class Pattern(StructuredNode):
    """
    Recurring emotional pattern detected across signals.

    V3: now uses multi-dimensional SA coordinate grouping instead of
    flat (trigger_category, emotion) grouping. Patterns can be detected
    at variable resolution (role-level vs person-level vs action-level).
    """

    uid = UniqueIdProperty()
    workspace_id = StringProperty()
    owner_user_id = IntegerProperty()
    name = StringProperty(required=True)
    hypothesis = StringProperty()
    score = FloatProperty()
    evidence_count = IntegerProperty(default=0)
    resolution_level = StringProperty()  # "role" | "person" | "action" | "context" | "temporal"
    shared_dimensions = JSONProperty()  # Which SA dimensions this pattern groups on
    created_at = DateTimeProperty(default_now=True)

    evidence_signals = RelationshipTo("Signal", "SOURCED_FROM")
    evidence_clusters = RelationshipTo("Cluster", "SOURCED_FROM")


# ──────────────────────────────────────────────────────────────────────────────
# SIGNAL CLUSTERS
# ──────────────────────────────────────────────────────────────────────────────


class Cluster(StructuredNode):
    """
    A recognized grouping of signals with coordinate density.

    Clusters are first-class graph nodes that emerge when multiple signals
    share coordinate values but may diverge along other dimensions. Clusters
    have their own temporal trajectory — their formation, strengthening,
    weakening, dissolution, and potential re-emergence are all tracked.

    Cluster types (based on shared vs divergent dimensions):
    - same_time_diff_emotion: Emotional range within a relationship
    - same_person_diff_time: Pattern detection (recurring emotion with person)
    - same_context_diff_person: Emotion attached to context, not individual
    - same_action_diff_everything: Structural relationship to the action itself

    Relationships:
        - member_signals: MEMBER_OF ← Signal (with MemberOfRel)
        - insights: INFORMED_BY ← Insight
        - pending_insights: AWAITING_REVIEW ← PendingInsight
        - patterns: SOURCED_FROM ← Pattern
    """

    uid = UniqueIdProperty()
    workspace_id = StringProperty()
    owner_user_id = IntegerProperty()
    cluster_id = StringProperty(required=True)  # Human-readable ID: "CLU-{type}-{hash}"
    cluster_type = StringProperty()  # "same_time_diff_emotion" | "same_person_diff_time" | etc.

    # Defining characteristics
    shared_coordinates = JSONProperty(default={})  # {"person": ["Sarah", "manager"], "context": ["work"]}
    divergent_dimensions = JSONProperty(default={})  # {"emotion": ["anger", "warmth"]}

    # Strength and trajectory
    strength = FloatProperty(default=0.0)  # Current strength (0.0-1.0)
    confidence_score = FloatProperty(default=0.5)  # Influenced by system detection + user validation
    trajectory_history = JSONProperty(default=[])  # [{timestamp, strength, member_count, event}]
    member_count = IntegerProperty(default=0)

    # Lifecycle
    status = StringProperty(default="active")  # "active" | "weakening" | "dissolved" | "disputed"
    created_at = DateTimeProperty(default_now=True)
    dissolved_at = DateTimeProperty()
    last_updated = DateTimeProperty(default_now=True)

    # Relationships
    member_signals = RelationshipFrom("Signal", "MEMBER_OF")
    insights = RelationshipFrom("Insight", "INFORMED_BY")
    pending_insights = RelationshipFrom("PendingInsight", "AWAITING_REVIEW")
    patterns = RelationshipFrom("Pattern", "SOURCED_FROM")


# ──────────────────────────────────────────────────────────────────────────────
# EMOTION, BEHAVIOR, OUTCOME — Contract Gap Nodes
# ──────────────────────────────────────────────────────────────────────────────
# These nodes close the gap identified in Reflect_Graph_Contract_v1.
# Emotion nodes provide graph-navigable aggregation across signals.
# Behavior and Outcome nodes support the full trigger → response → consequence
# chain once the system matures beyond MVP.


class Emotion(StructuredNode):
    """
    First-class Emotion node in the Signal Address System.

    Represents a distinct emotion that appears across signals: anger, shame,
    warmth, frustration, etc. While each Signal stores its own emotions as a
    JSON array (with intensity, source_coordinate, confidence per emotion),
    Emotion nodes aggregate across signals for graph navigation.

    A user can ask "show me everything connected to anger" — the Emotion node
    is the anchor for that query.

    Relationships:
        - signals: EXPRESSES_EMOTION ← Signal
    """

    uid = UniqueIdProperty()
    workspace_id = StringProperty()
    owner_user_id = IntegerProperty()
    name = StringProperty(required=True, unique_index=True)
    description = StringProperty()  # Optional human description
    valence = StringProperty()  # "positive" | "negative" | "neutral" | "mixed"
    created_at = DateTimeProperty(default_now=True)

    # Signals expressing this emotion
    signals = RelationshipFrom("Signal", "EXPRESSES_EMOTION")


class Behavior(StructuredNode):
    """
    Aggregated trigger-response pattern node (tentative).

    Represents a recurring behavioral pattern: how someone typically responds
    in a given context. Example: "avoidance when confronted", "defensiveness
    under criticism". Behaviors bridge the gap between raw signals and
    higher-level pattern recognition.

    This node is tentative for MVP — it may be deferred to post-hackathon
    if the Pattern + Cluster system provides sufficient coverage.

    Relationships:
        - signals: SHOWS_BEHAVIOR ← Signal
    """

    uid = UniqueIdProperty()
    workspace_id = StringProperty()
    owner_user_id = IntegerProperty()
    name = StringProperty(required=True)
    description = StringProperty()
    behavior_type = StringProperty()  # "response" | "avoidance" | "escalation" | "de-escalation"
    frequency = IntegerProperty(default=0)  # How many signals exhibit this behavior
    created_at = DateTimeProperty(default_now=True)

    # Signals showing this behavior
    signals = RelationshipFrom("Signal", "SHOWS_BEHAVIOR")


class Outcome(StructuredNode):
    """
    Downstream consequence of a reaction or behavioral pattern.

    Represents what happened as a result: reconciliation, escalation,
    withdrawal, resolution, etc. Outcomes close the loop on the
    trigger → response → consequence chain.

    Relationships:
        - signals: LED_TO ← Signal
    """

    uid = UniqueIdProperty()
    workspace_id = StringProperty()
    owner_user_id = IntegerProperty()
    name = StringProperty(required=True)
    description = StringProperty()
    valence = StringProperty()  # "positive" | "negative" | "neutral"
    created_at = DateTimeProperty(default_now=True)

    # Signals that led to this outcome
    signals = RelationshipFrom("Signal", "LED_TO")


# ──────────────────────────────────────────────────────────────────────────────
# REASONING TRACES — Insight and Reflection Nodes
# ──────────────────────────────────────────────────────────────────────────────


class Insight(StructuredNode):
    """
    LLM-generated reasoning trace stored in the graph.

    When the AI generates a meaningful observation — a connection, a pattern
    hypothesis, a cluster interpretation — it gets stored as an Insight node.
    Users can validate, dispute, or ignore insights. The insight's validation
    status is itself a signal (all feedback is signal).

    Relationships:
        - informed_by_signals: INFORMED_BY → Signal (with InformedByRel)
        - informed_by_clusters: INFORMED_BY → Cluster
        - reflections: RESPONDS_TO ← Reflection
        - conversation: GENERATED_DURING → Conversation
    """

    uid = UniqueIdProperty()
    workspace_id = StringProperty()
    owner_user_id = IntegerProperty()
    reasoning_text = StringProperty(required=True)
    persona = StringProperty()  # Which AI persona generated this
    confidence = FloatProperty(default=0.5)  # LLM's self-assessed confidence
    validation_status = StringProperty(default="pending")  # "pending" | "validated" | "disputed" | "ignored"
    generated_at = DateTimeProperty(default_now=True)
    validated_at = DateTimeProperty()
    validation_feedback = StringProperty()  # User's reason for validating/disputing

    # What this insight is about
    informed_by_signals = RelationshipTo("Signal", "INFORMED_BY", model=InformedByRel)
    informed_by_clusters = RelationshipTo("Cluster", "INFORMED_BY")

    # User responses
    reflections = RelationshipFrom("Reflection", "RESPONDS_TO")

    # Conversation context
    conversation = RelationshipTo("Conversation", "GENERATED_DURING")


class Reflection(StructuredNode):
    """
    User-authored reasoning trace stored in the graph.

    When the user responds to an AI observation — agreeing, pushing back,
    adding context, having an insight of their own — it gets stored as a
    Reflection node. Reflections interleave with Insights to form a full
    reasoning trace that is graph-navigable.

    Relationships:
        - responds_to: RESPONDS_TO → Insight
        - refines: REFINES → Signal
    """

    uid = UniqueIdProperty()
    workspace_id = StringProperty()
    owner_user_id = IntegerProperty()
    text = StringProperty(required=True)
    reflection_type = StringProperty()  # "agreement" | "dispute" | "elaboration" | "realization" | "question"
    created_at = DateTimeProperty(default_now=True)

    responds_to = RelationshipTo("Insight", "RESPONDS_TO")
    refines = RelationshipTo("Signal", "REFINES")


class PendingInsight(StructuredNode):
    """
    Background detection result awaiting surfacing in a conversation.

    Generated by the graph agent (Mode 3) during background processing.
    Not shown to the user directly — instead picked up by the context
    assembly layer during the next Mode 2 conversation and included in
    the context packet for the LLM to decide whether to mention.

    Relationships:
        - cluster: AWAITING_REVIEW → Cluster
    """

    uid = UniqueIdProperty()
    workspace_id = StringProperty()
    owner_user_id = IntegerProperty()
    detection_type = StringProperty(required=True)  # "new_cluster" | "cluster_strengthened" | "cluster_dissolving" | "pattern_detected" | "trajectory_shift"
    description = StringProperty()  # Brief description for context packet
    confidence = FloatProperty(default=0.5)
    status = StringProperty(default="pending")  # "pending" | "surfaced" | "expired"
    created_at = DateTimeProperty(default_now=True)
    surfaced_at = DateTimeProperty()
    expires_at = DateTimeProperty()  # Stale insights get pruned

    cluster = RelationshipTo("Cluster", "AWAITING_REVIEW")


# ──────────────────────────────────────────────────────────────────────────────
# PIPELINE TRACE (context assembly observability)
# ──────────────────────────────────────────────────────────────────────────────


class PipelineTrace(StructuredNode):
    """
    Captures what happened during context assembly for a user message.

    Persisted to Neo4j so the Insights tab can show "How this was understood" —
    the entities extracted, which persona was used, and a summary of the context
    packet sent to the LLM.

    Relationships:
        - for_turn: TRACES → UserTurn (the message this trace describes)
    """

    uid = UniqueIdProperty()
    workspace_id = StringProperty(required=True, index=True)
    entities_extracted = JSONProperty(default={})  # {persons: [], contexts: [], actions: [], temporal: []}
    persona_used = StringProperty()  # "gentle_explorer" | "direct_challenger" | "neutral_observer"
    context_packet_summary = StringProperty()  # Compressed text summary of what was injected
    signals_referenced = IntegerProperty(default=0)
    clusters_referenced = IntegerProperty(default=0)
    token_count = IntegerProperty(default=0)
    created_at = FloatProperty(default_or_none=None)

    for_turn = RelationshipTo("UserTurn", "TRACES")


# ──────────────────────────────────────────────────────────────────────────────
# DATA SOURCE (for imports)
# ──────────────────────────────────────────────────────────────────────────────


class DataSource(StructuredNode):
    """
    Source of imported conversation data.

    Used by the import pipeline to track where conversations came from.
    """

    uid = UniqueIdProperty()
    platform = StringProperty(required=True)  # "chatgpt" | "claude" | "thrivesight"
    workspace_id = StringProperty()
    owner_user_id = IntegerProperty()
    import_date = DateTimeProperty(default_now=True)
    total_conversations = IntegerProperty(default=0)
    total_turns = IntegerProperty(default=0)
    date_range_start = FloatProperty()
    date_range_end = FloatProperty()
    imported_at = DateTimeProperty()
    last_updated = DateTimeProperty()

    conversations = RelationshipTo("Conversation", "IMPORTED_FROM")


class UserProfile(StructuredNode):
    """
    User identity node in the knowledge graph.

    Created on signup/login to anchor all user-owned graph data.
    Every Signal, Conversation, Cluster, and Insight links back to
    the owning user via workspace_id / owner_user_id properties.
    The UserProfile node makes users queryable in Neo4j and serves
    as the root for per-user graph traversals.

    Journey properties store the user's progress through the Polity
    civic discovery journey (Act I phases, path selection, soul ordering,
    value ordering). Provocation responses are stored as separate
    ProvocationResponseNode nodes linked via HAS_PROVOCATION_RESPONSE.
    """

    uid = UniqueIdProperty()
    workspace_id = StringProperty(index=True)
    owner_user_id = IntegerProperty(index=True)
    username = StringProperty(index=True)
    email = StringProperty()
    total_messages = IntegerProperty(default=0)
    total_conversations = IntegerProperty(default=0)
    total_signals = IntegerProperty(default=0)
    avg_message_length = FloatProperty()
    question_ratio = FloatProperty()
    top_topics = JSONProperty()
    created_at = DateTimeProperty(default_now=True)
    last_active = DateTimeProperty()
    last_import = DateTimeProperty()

    # Journey state — civic discovery progress
    journey_phase = StringProperty()          # One of the 18 JourneyPhase values
    path_id = StringProperty()                # "wanderer" | "sovereign" | "philosopher"
    philosopher_mode = StringProperty()       # "guided" | "socratic"
    sovereign_mode = StringProperty()         # "advised" | "self-advised"
    sovereign_end_statement = StringProperty()
    soul_ordering = JSONProperty()            # Serialized SoulOrdering object
    value_ordering = JSONProperty()           # List of ValueId strings
    current_provocation_index = IntegerProperty(default=0)
    socratic_chariot_revealed = BooleanProperty(default=False)
    socratic_tier_revealed = BooleanProperty(default=False)

    # Relationships to user-owned data
    conversations = RelationshipTo("Conversation", "OWNS_CONVERSATION")
    provocation_responses = RelationshipTo("ProvocationResponseNode", "HAS_PROVOCATION_RESPONSE")


class ProvocationResponseNode(StructuredNode):
    """
    A user's response to a single provocation in the civic journey.

    Stored as separate nodes (not a JSON blob) so they can be queried
    individually and linked to value/soul analysis in the graph.
    """

    uid = UniqueIdProperty()
    workspace_id = StringProperty(index=True)
    owner_user_id = IntegerProperty(index=True)
    provocation_id = StringProperty(required=True)
    choice_id = StringProperty(required=True)
    served_soul_part = StringProperty()       # "reason" | "spirit" | "appetite"
    protected_values = JSONProperty()         # List of ValueId strings
    sacrificed_values = JSONProperty()        # List of ValueId strings
    deliberation_time_ms = IntegerProperty()
    was_instinctive = BooleanProperty(default=False)
    timestamp = DateTimeProperty(default_now=True)

    # Ontology links (created when ontology is seeded)
    to_provocation = RelationshipTo("ProvocationOntologyNode", "TO_PROVOCATION")
    chose = RelationshipTo("ProvocationChoiceNode", "CHOSE")


class Topic(StructuredNode):
    """
    Extracted topic from conversations.
    """

    uid = UniqueIdProperty()
    workspace_id = StringProperty()
    owner_user_id = IntegerProperty()
    word = StringProperty()
    name = StringProperty(required=True)
    frequency = IntegerProperty(default=0)
    is_bigram = BooleanProperty(default=False)
    total_count = IntegerProperty(default=0)
    conversation_count = IntegerProperty(default=0)

    conversations = RelationshipTo("Conversation", "DISCUSSED_IN")


# ──────────────────────────────────────────────────────────────────────────────
# CROSS-SOURCE DATA (STRETCH — retained from V1)
# ──────────────────────────────────────────────────────────────────────────────


class Transaction(StructuredNode):
    """Financial transaction from imported external data."""

    uid = UniqueIdProperty()
    amount = FloatProperty()
    merchant = StringProperty()
    category = StringProperty()
    occurred_at = DateTimeProperty()


class JournalEntry(StructuredNode):
    """Journal entry or AI conversation excerpt from imported data."""

    uid = UniqueIdProperty()
    text = StringProperty()
    emotion = StringProperty()
    occurred_at = DateTimeProperty()


class CalendarEvent(StructuredNode):
    """Calendar event from imported external data."""

    uid = UniqueIdProperty()
    title = StringProperty()
    start_time = DateTimeProperty()
    end_time = DateTimeProperty()
    stress_level = IntegerProperty()


class SocialPost(StructuredNode):
    """Social media post from imported external data."""

    uid = UniqueIdProperty()
    text = StringProperty()
    platform = StringProperty()
    posted_at = DateTimeProperty()
    sentiment = StringProperty()


# ──────────────────────────────────────────────────────────────────────────────
# SEED VOCABULARY AND INITIALIZATION
# ──────────────────────────────────────────────────────────────────────────────


SEED_CATEGORIES = [
    {
        "name": "dismissal",
        "description": "Minimizing or ignoring the other person's concern, feeling, or statement.",
    },
    {
        "name": "accusation",
        "description": "Directly blaming the other person, often with absolutist language like 'you always' or 'you never.'",
    },
    {
        "name": "deflection",
        "description": "Changing the subject or redirecting blame to avoid addressing the issue.",
    },
    {
        "name": "withdrawal",
        "description": "Emotionally or physically disengaging from the conversation.",
    },
    {
        "name": "demand",
        "description": "Issuing an ultimatum or insisting on a specific action.",
    },
    {
        "name": "questioning",
        "description": "Asking for clarification or information, which may be genuine or interrogative.",
    },
    {
        "name": "validation",
        "description": "Acknowledging the other person's perspective or feeling as legitimate.",
    },
    {
        "name": "acknowledgment",
        "description": "Recognizing the situation without necessarily agreeing.",
    },
    {
        "name": "concession",
        "description": "Yielding ground, accepting partial responsibility, or offering compromise.",
    },
    {
        "name": "sarcasm",
        "description": "Using irony or mocking tone to undermine the other person's position.",
    },
    {
        "name": "initiation",
        "description": "Opening a conversation topic. Used for the first turn or when a new topic is introduced.",
    },
]


SEED_CONTEXTS = [
    {"name": "work", "description": "Professional or employment-related settings"},
    {"name": "home", "description": "Domestic or household settings"},
    {"name": "social", "description": "Social gatherings, friendships, community"},
    {"name": "family", "description": "Family relationships and dynamics"},
    {"name": "health", "description": "Physical or mental health contexts"},
    {"name": "self", "description": "Internal reflection, self-relationship"},
]


def initialize_seed_categories():
    """
    Create seed TriggerCategory nodes if they don't already exist.

    Returns:
        dict: Summary of categories created and skipped.
    """
    created = []
    skipped = []

    for cat_data in SEED_CATEGORIES:
        existing = TriggerCategory.nodes.filter(name=cat_data["name"])
        if not existing:
            cat = TriggerCategory(
                name=cat_data["name"],
                description=cat_data["description"],
                is_system=True,
                is_proposed=False,
                proposed_by="system",
                usage_count=0,
            )
            cat.save()
            created.append(cat_data["name"])
        else:
            skipped.append(cat_data["name"])

    return {"created": created, "skipped": skipped}


def initialize_seed_contexts():
    """
    Create seed ContextNode nodes if they don't already exist.

    Returns:
        dict: Summary of contexts created and skipped.
    """
    created = []
    skipped = []

    for ctx_data in SEED_CONTEXTS:
        existing = ContextNode.nodes.filter(name=ctx_data["name"])
        if not existing:
            ctx = ContextNode(
                name=ctx_data["name"],
                description=ctx_data["description"],
                level=0,
            )
            ctx.save()
            created.append(ctx_data["name"])
        else:
            skipped.append(ctx_data["name"])

    return {"created": created, "skipped": skipped}


def get_active_categories():
    """
    Return categories to include in LLM prompts for TriggerActionInference.

    Active categories consist of:
    1. All system categories (is_system=True)
    2. All validated proposed categories (is_proposed=True AND usage_count >= 3)
    """
    system_categories = TriggerCategory.nodes.filter(is_system=True)
    validated_proposed = TriggerCategory.nodes.filter(
        is_proposed=True, usage_count__gte=3
    )

    return list(system_categories) + list(validated_proposed)


# ──────────────────────────────────────────────────────────────────────────────
# CIVIC ONTOLOGY NODES (Seeded reference data for Journey's value framework)
# ──────────────────────────────────────────────────────────────────────────────


class ValueNode(StructuredNode):
    """
    One of 12 civic values in the Polity ontology.

    Seeded once via ``manage.py seed_ontology``.  Not workspace-scoped —
    the ontology is shared across all users.
    """

    value_id = StringProperty(unique_index=True, required=True)
    name = StringProperty(required=True)
    definition = StringProperty()
    tradeoff = StringProperty()  # Formerly sacrifice_test
    tier = StringProperty()  # "foundational" | "structural" | "aspirational"
    corrupt_name = StringProperty()
    corrupt_form = StringProperty()

    # Relationships
    tensions = RelationshipTo("ValueNode", "TENSIONS")
    soul_affinity = RelationshipTo("SoulPartNode", "SOUL_AFFINITY")


class SoulPartNode(StructuredNode):
    """
    One of the three Platonic soul parts: reason, spirit, appetite.

    Seeded once via ``manage.py seed_ontology``.
    """

    part_id = StringProperty(unique_index=True, required=True)
    name = StringProperty(required=True)
    description = StringProperty()
    tier_affiliation = StringProperty()  # "foundational" | "structural" | "aspirational"


class ProvocationOntologyNode(StructuredNode):
    """
    One of 12 provocations in the civic journey.

    Named ProvocationOntologyNode to avoid collision with user-scoped
    ProvocationResponseNode.  Seeded once via ``manage.py seed_ontology``.
    """

    provocation_id = StringProperty(unique_index=True, required=True)
    form = StringProperty()  # "moment" | "tension"
    title = StringProperty()
    text = StringProperty()
    tension = StringProperty()  # "reason-spirit" | "reason-appetite" | "spirit-appetite"
    sequence_order = IntegerProperty()

    # Relationships
    choices = RelationshipTo("ProvocationChoiceNode", "HAS_CHOICE")


class ProvocationChoiceNode(StructuredNode):
    """
    A single choice within a provocation (2 per provocation, 24 total).

    Seeded once via ``manage.py seed_ontology``.
    """

    choice_id = StringProperty(unique_index=True, required=True)
    text = StringProperty()
    parent_provocation_id = StringProperty(index=True)

    # Relationships
    serves = RelationshipTo("SoulPartNode", "SERVES")
    protects = RelationshipTo("ValueNode", "PROTECTS")
    sacrifices = RelationshipTo("ValueNode", "SACRIFICES")


# ──────────────────────────────────────────────────────────────────────────────
# INDEX DEFINITIONS
# ──────────────────────────────────────────────────────────────────────────────

STARTUP_INDEXES = [
    # Core nodes
    "CREATE INDEX IF NOT EXISTS FOR (n:Turn) ON (n.turn_number)",
    "CREATE INDEX IF NOT EXISTS FOR (n:Signal) ON (n.emotion)",
    "CREATE INDEX IF NOT EXISTS FOR (n:Signal) ON (n.signal_address)",
    "CREATE INDEX IF NOT EXISTS FOR (n:Signal) ON (n.confidence_score)",
    "CREATE INDEX IF NOT EXISTS FOR (n:TriggerCategory) ON (n.name)",
    "CREATE INDEX IF NOT EXISTS FOR (n:Person) ON (n.name)",
    "CREATE INDEX IF NOT EXISTS FOR (n:Conversation) ON (n.conversation_id)",
    "CREATE INDEX IF NOT EXISTS FOR (n:Conversation) ON (n.workspace_id)",
    "CREATE INDEX IF NOT EXISTS FOR (n:Topic) ON (n.workspace_id)",
    "CREATE INDEX IF NOT EXISTS FOR (n:Topic) ON (n.word)",
    "CREATE INDEX IF NOT EXISTS FOR (n:UserProfile) ON (n.workspace_id)",
    "CREATE INDEX IF NOT EXISTS FOR (n:UserProfile) ON (n.owner_user_id)",
    "CREATE INDEX IF NOT EXISTS FOR (n:DataSource) ON (n.workspace_id)",
    "CREATE INDEX IF NOT EXISTS FOR (n:UserTurn) ON (n.workspace_id)",
    "CREATE INDEX IF NOT EXISTS FOR (n:AssistantTurn) ON (n.workspace_id)",
    # Coordinate hierarchy
    "CREATE INDEX IF NOT EXISTS FOR (n:ContextNode) ON (n.name)",
    "CREATE INDEX IF NOT EXISTS FOR (n:ActionNode) ON (n.name)",
    "CREATE INDEX IF NOT EXISTS FOR (n:TemporalNode) ON (n.name)",
    # Clusters
    "CREATE INDEX IF NOT EXISTS FOR (n:Cluster) ON (n.cluster_id)",
    "CREATE INDEX IF NOT EXISTS FOR (n:Cluster) ON (n.status)",
    # Reasoning traces
    "CREATE INDEX IF NOT EXISTS FOR (n:Insight) ON (n.validation_status)",
    "CREATE INDEX IF NOT EXISTS FOR (n:PendingInsight) ON (n.status)",
    # Pipeline observability
    "CREATE INDEX IF NOT EXISTS FOR (n:PipelineTrace) ON (n.workspace_id)",
    # Contract gap nodes
    "CREATE INDEX IF NOT EXISTS FOR (n:Emotion) ON (n.name)",
    "CREATE INDEX IF NOT EXISTS FOR (n:Behavior) ON (n.name)",
    "CREATE INDEX IF NOT EXISTS FOR (n:Outcome) ON (n.name)",
    # Workspace scoping (enables multi-tenant queries)
    "CREATE INDEX IF NOT EXISTS FOR (n:Signal) ON (n.workspace_id)",
    "CREATE INDEX IF NOT EXISTS FOR (n:Conversation) ON (n.workspace_id)",
    "CREATE INDEX IF NOT EXISTS FOR (n:Cluster) ON (n.workspace_id)",
    "CREATE INDEX IF NOT EXISTS FOR (n:Pattern) ON (n.workspace_id)",
    # Civic ontology
    "CREATE INDEX IF NOT EXISTS FOR (n:ValueNode) ON (n.value_id)",
    "CREATE INDEX IF NOT EXISTS FOR (n:SoulPartNode) ON (n.part_id)",
    "CREATE INDEX IF NOT EXISTS FOR (n:ProvocationOntologyNode) ON (n.provocation_id)",
    "CREATE INDEX IF NOT EXISTS FOR (n:ProvocationChoiceNode) ON (n.choice_id)",
    "CREATE INDEX IF NOT EXISTS FOR (n:ProvocationResponseNode) ON (n.provocation_id)",
]


# ──────────────────────────────────────────────────────────────────────────────
# V2 COMPATIBILITY STUBS — Deprecated screenplay nodes
# ──────────────────────────────────────────────────────────────────────────────
# These classes are retained ONLY for backward compatibility with existing
# views (CharacterProfileListCreateView, ScreenplayAnalyzeView, etc.) and
# character_network.py. They will be removed in a future cleanup phase once
# the screenplay pipeline is fully deprecated.
#
# DO NOT USE THESE FOR NEW CODE. Use the V3 SA architecture nodes above.
# ──────────────────────────────────────────────────────────────────────────────


class RelatesToRel(StructuredRel):
    """DEPRECATED V2 — enriched RELATES_TO edge between CharacterProfile nodes."""

    relationship_type = StringProperty()
    co_occurrence_count = IntegerProperty(default=0)
    shared_dialogue_scenes = IntegerProperty(default=0)
    witness_scenes = IntegerProperty(default=0)
    first_observed_scene = IntegerProperty()
    last_observed_scene = IntegerProperty()
    significance_score = FloatProperty()
    qualitative_weight = FloatProperty()
    observation_source = StringProperty(default="screenplay_parse")
    observation_completeness = StringProperty(default="full_work")


class DirectedAtRel(StructuredRel):
    """DEPRECATED V2 — DIRECTED_AT edge from Turn to CharacterProfile."""

    inference_method = StringProperty(required=True)
    confidence = StringProperty(default="inferred")
    audit_corrected = BooleanProperty(default=False)
    original_inference_method = StringProperty()
    audit_confidence = FloatProperty()


class AppearsInRel(StructuredRel):
    """DEPRECATED V2 — APPEARS_IN edge from CharacterProfile to Scene."""

    role = StringProperty(default="present")
    scene_number = IntegerProperty()


class SourceWork(StructuredNode):
    """DEPRECATED V2 — ontological root node for screenplay analysis."""

    uid = UniqueIdProperty()
    title = StringProperty(required=True, unique_index=True)
    year = IntegerProperty()
    medium = StringProperty()
    writers = ArrayProperty(StringProperty())
    source_material = StringProperty()
    universe = StringProperty()
    ontology_type = StringProperty(required=True)

    master_documents = RelationshipFrom("MasterDocument", "PARSED_FROM")
    characters = RelationshipFrom("CharacterProfile", "BELONGS_TO")


class MasterDocument(StructuredNode):
    """DEPRECATED V2 — canonical parse output for screenplay analysis."""

    uid = UniqueIdProperty()
    total_scenes = IntegerProperty(default=0)
    total_dialogue_turns = IntegerProperty(default=0)
    total_characters = IntegerProperty(default=0)
    parse_date = DateTimeProperty(default_now=True)
    parser_version = StringProperty(default="1.0")
    raw_text_hash = StringProperty()

    source_work = RelationshipTo("SourceWork", "PARSED_FROM")
    scenes = RelationshipTo("Scene", "CONTAINS")


class CharacterProfile(StructuredNode):
    """DEPRECATED V2 — identity node for screenplay character analysis."""

    uid = UniqueIdProperty()
    name = StringProperty(required=True, unique_index=True)
    source_work = StringProperty(required=True)
    description = StringProperty()
    profile_image_url = StringProperty()
    created_at = DateTimeProperty(default_now=True)
    vocabulary_json = JSONProperty()

    belongs_to = RelationshipTo("SourceWork", "BELONGS_TO")
    scenes = RelationshipTo("Scene", "APPEARS_IN", model=AppearsInRel)
    relationships = RelationshipTo("CharacterProfile", "RELATES_TO", model=RelatesToRel)
    significant_words = RelationshipFrom("SignificantWord", "USED_BY")
    turns = RelationshipFrom("Turn", "SPOKEN_BY")

    @property
    def conversations(self):
        return list(self.scenes.all())

    @property
    def scene_count(self):
        return len(self.scenes.all()) if self.scenes else 0


class Scene(StructuredNode):
    """DEPRECATED V2 — discrete dramatic unit for screenplay analysis."""

    uid = UniqueIdProperty()
    scene_number = IntegerProperty(required=True)
    slugline = StringProperty()
    location = StringProperty()
    time_of_day = StringProperty()
    action_summary = StringProperty()
    narrative_order = IntegerProperty()
    title = StringProperty()
    created_at = DateTimeProperty(default_now=True)
    source_type = StringProperty(default="screenplay")
    episode_reference = StringProperty()

    turns = RelationshipTo("Turn", "CONTAINS")
    characters = RelationshipFrom("CharacterProfile", "APPEARS_IN")
    master_document = RelationshipFrom("MasterDocument", "CONTAINS")
    signals = RelationshipFrom("Signal", "SOURCED_FROM")


class SignificantWord(StructuredNode):
    """DEPRECATED V2 — breathing graph promoted vocabulary item."""

    uid = UniqueIdProperty()
    word = StringProperty(required=True, unique_index=True)
    total_frequency = IntegerProperty(default=0)
    promoted_by = StringProperty(required=True)
    promoted_at = DateTimeProperty(default_now=True)
    promotion_reason = StringProperty()
    scene_distribution = JSONProperty()
    emotional_context = JSONProperty()
    demotion_eligible = BooleanProperty(default=True)
    traversal_count = IntegerProperty(default=0)
    last_traversed = DateTimeProperty()

    character = RelationshipTo("CharacterProfile", "USED_BY")
    scenes = RelationshipTo("Scene", "APPEARS_IN")
    co_occurs_with = RelationshipTo("SignificantWord", "CO_OCCURS_WITH")


class Reframe(StructuredNode):
    """DEPRECATED V2 — alternative perspective on key moments."""

    uid = UniqueIdProperty()
    text = StringProperty(required=True)
    original_turn_text = StringProperty()
    perspective_type = StringProperty()
    generated_at = DateTimeProperty(default_now=True)

    turn = RelationshipTo("Turn", "REFRAMES")


def initialize_indexes(db_connection=None):
    """
    Create performance indexes in Neo4j.

    Args:
        db_connection: Optional neomodel db connection. If None, uses default.

    Returns:
        dict: Summary of indexes created.
    """
    from neomodel import db as default_db

    connection = db_connection or default_db
    results = []

    for index_cypher in STARTUP_INDEXES:
        try:
            connection.cypher_query(index_cypher)
            results.append({"query": index_cypher, "status": "created"})
        except Exception as e:
            results.append({"query": index_cypher, "status": f"skipped ({e})"})

    return results
