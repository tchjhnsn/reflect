import jsonschema
from typing import List, Dict, Any

# ============================================================================
# SCHEMA DEFINITIONS
# ============================================================================

TURN_SCHEMA = {
    "type": "object",
    "required": ["turn_number", "speaker", "text"],
    "properties": {
        "turn_number": {
            "type": "integer",
            "minimum": 1,
            "description": "Sequential position in the conversation"
        },
        "speaker": {
            "type": "string",
            "description": "Identified speaker name or label"
        },
        "text": {
            "type": "string",
            "minLength": 1,
            "description": "The content of the turn"
        },
        "timestamp": {
            "type": ["string", "null"],
            "description": "ISO 8601 timestamp if available"
        },
        "raw_offset": {
            "type": "integer",
            "description": "Character offset in the original input"
        }
    }
}

SIGNAL_SCHEMA = {
    "type": "object",
    "required": ["turn_number", "speaker", "emotion", "intensity"],
    "properties": {
        "turn_number": {
            "type": "integer"
        },
        "speaker": {
            "type": "string"
        },
        "emotion": {
            "type": "string",
            "description": "From known + proposed emotion set"
        },
        "intensity": {
            "type": "number",
            "minimum": 1.0,
            "maximum": 5.0,
            "description": "Emotional intensity on 1-5 scale"
        },
        "reaction": {
            "type": "string",
            "enum": ["defended", "counter_attacked", "withdrew", "de_escalated", "acknowledged", "escalated", "deflected", "conceded"],
            "description": "How this speaker responded to the trigger"
        },
        "trigger_action": {
            "type": "object",
            "required": ["action_text", "category"],
            "properties": {
                "action_text": {
                    "type": "string",
                    "description": "What the other person did in the preceding turn"
                },
                "category": {
                    "type": "string",
                    "description": "Existing or proposed trigger category name"
                },
                "is_new_category": {
                    "type": "boolean",
                    "default": False
                },
                "category_description": {
                    "type": ["string", "null"],
                    "description": "Required if is_new_category is true"
                }
            }
        },
        "signal_address": {
            "type": "string",
            "pattern": "^SA\\(.*\\)$",
            "description": "SA(context, person, action, temporal) address"
        }
    }
}

PATTERN_SCHEMA = {
    "type": "object",
    "required": ["pattern_name", "hypothesis", "score", "evidence", "trigger_category", "response_emotion", "occurrence_count"],
    "properties": {
        "pattern_name": {
            "type": "string",
            "description": "Human-readable label for the dynamic"
        },
        "hypothesis": {
            "type": "string",
            "description": "Testable statement about the pattern's mechanism"
        },
        "score": {
            "type": "number",
            "description": "occurrence_count × average_intensity"
        },
        "evidence": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "turn_number": {"type": "integer"},
                    "signal_address": {"type": "string"},
                    "text_excerpt": {"type": "string"}
                }
            }
        },
        "trigger_category": {
            "type": "string"
        },
        "response_emotion": {
            "type": "string"
        },
        "occurrence_count": {
            "type": "integer",
            "minimum": 2,
            "description": "Minimum 2 for single-conversation analysis"
        }
    }
}

TRAJECTORY_SCHEMA = {
    "type": "object",
    "required": ["speakers"],
    "properties": {
        "speakers": {
            "type": "object",
            "additionalProperties": {
                "type": "object",
                "properties": {
                    "data_points": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "turn_number": {"type": "integer"},
                                "intensity": {"type": "number"},
                                "emotion": {"type": "string"}
                            }
                        }
                    },
                    "inflection_points": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "turn_number": {"type": "integer"},
                                "intensity_delta": {"type": "number"},
                                "cause": {
                                    "type": "string",
                                    "description": "The trigger action that caused this shift"
                                },
                                "direction": {
                                    "type": "string",
                                    "enum": ["escalation", "de_escalation"]
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

REFRAME_SCHEMA = {
    "type": "object",
    "required": ["text", "patterns_referenced"],
    "properties": {
        "text": {
            "type": "string",
            "description": "The externalized, plain-language reframe"
        },
        "patterns_referenced": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Pattern names that informed this reframe"
        },
        "resolution_elements": {
            "type": "object",
            "properties": {
                "externalization": {
                    "type": "string",
                    "description": "The dynamic described as a thing, not a person"
                },
                "accumulation": {
                    "type": "string",
                    "description": "How this pattern has built over time"
                },
                "intervention_point": {
                    "type": "string",
                    "description": "Where in the cycle a different response could break it"
                }
            }
        }
    }
}

# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def validate_turns(turns: List[Dict]) -> List[str]:
    """
    Validate an array of turn objects.

    Args:
        turns: List of turn dictionaries

    Returns:
        List of error messages (empty if valid)
    """
    errors = []
    for i, turn in enumerate(turns):
        try:
            jsonschema.validate(turn, TURN_SCHEMA)
        except jsonschema.ValidationError as e:
            errors.append(f"Turn {i}: {e.message}")
    return errors


def validate_signal(signal: Dict) -> List[str]:
    """
    Validate a single signal object with business rules.

    Business rules:
    - Intensity must be in range [1.0, 5.0]
    - New categories must include a description

    Args:
        signal: Signal dictionary

    Returns:
        List of error messages (empty if valid)
    """
    errors = []

    # Structural validation
    try:
        jsonschema.validate(signal, SIGNAL_SCHEMA)
    except jsonschema.ValidationError as e:
        errors.append(e.message)

    # Business rule: intensity range
    intensity = signal.get('intensity')
    if intensity is not None and (intensity < 1.0 or intensity > 5.0):
        errors.append(f"Intensity {intensity} out of range [1.0, 5.0]")

    # Business rule: new category requires description
    trigger_action = signal.get('trigger_action', {})
    if trigger_action.get('is_new_category') and not trigger_action.get('category_description'):
        errors.append("New category proposed without description")

    return errors


def validate_signals(signals: List[Dict]) -> List[str]:
    """
    Validate an array of signal objects.

    Args:
        signals: List of signal dictionaries

    Returns:
        List of error messages (empty if valid)
    """
    errors = []
    for i, signal in enumerate(signals):
        signal_errors = validate_signal(signal)
        for error in signal_errors:
            errors.append(f"Signal {i}: {error}")
    return errors


def validate_pattern(pattern: Dict) -> List[str]:
    """
    Validate a single pattern object with business rules.

    Business rules:
    - Evidence must contain at least 2 items (minimum threshold)
    - Occurrence count must be >= 2

    Args:
        pattern: Pattern dictionary

    Returns:
        List of error messages (empty if valid)
    """
    errors = []

    # Structural validation
    try:
        jsonschema.validate(pattern, PATTERN_SCHEMA)
    except jsonschema.ValidationError as e:
        errors.append(e.message)

    # Business rule: minimum evidence threshold
    evidence = pattern.get('evidence', [])
    if len(evidence) < 2:
        errors.append(f"Pattern must have at least 2 evidence items, found {len(evidence)}")

    # Business rule: occurrence count minimum
    occurrence_count = pattern.get('occurrence_count')
    if occurrence_count is not None and occurrence_count < 2:
        errors.append(f"Occurrence count must be >= 2, found {occurrence_count}")

    return errors


def validate_patterns(patterns: List[Dict]) -> List[str]:
    """
    Validate an array of pattern objects.

    Args:
        patterns: List of pattern dictionaries

    Returns:
        List of error messages (empty if valid)
    """
    errors = []
    for i, pattern in enumerate(patterns):
        pattern_errors = validate_pattern(pattern)
        for error in pattern_errors:
            errors.append(f"Pattern {i}: {error}")
    return errors


def validate_trajectory(trajectory: Dict) -> List[str]:
    """
    Validate a trajectory object.

    Args:
        trajectory: Trajectory dictionary

    Returns:
        List of error messages (empty if valid)
    """
    errors = []

    # Structural validation
    try:
        jsonschema.validate(trajectory, TRAJECTORY_SCHEMA)
    except jsonschema.ValidationError as e:
        errors.append(e.message)

    return errors


def validate_reframe(reframe: Dict) -> List[str]:
    """
    Validate a reframe object with business rules.

    Business rules:
    - Must reference at least one pattern
    - Must not assign blame to either person (checked via text content)

    Args:
        reframe: Reframe dictionary

    Returns:
        List of error messages (empty if valid)
    """
    errors = []

    # Structural validation
    try:
        jsonschema.validate(reframe, REFRAME_SCHEMA)
    except jsonschema.ValidationError as e:
        errors.append(e.message)

    # Business rule: must reference at least one pattern
    patterns_referenced = reframe.get('patterns_referenced', [])
    if not patterns_referenced or len(patterns_referenced) == 0:
        errors.append("Reframe must reference at least one pattern")

    # Business rule: check for blame assignment language
    reframe_text = reframe.get('text', '').lower()
    blame_keywords = ["you always", "you never", "you're the problem", "your fault", "you caused"]

    for keyword in blame_keywords:
        if keyword in reframe_text:
            errors.append(f"Reframe appears to assign blame with language: '{keyword}'")

    return errors


def validate_analysis_response(response: Dict) -> List[str]:
    """
    Validate the complete analysis response object.

    Validates:
    - conversation metadata
    - signals array
    - patterns array
    - trajectory object
    - reframe object
    - metadata object

    Args:
        response: Complete analysis response dictionary

    Returns:
        List of error messages (empty if valid)
    """
    errors = []

    # Validate conversation metadata
    conversation = response.get('conversation', {})
    if not conversation:
        errors.append("Response missing 'conversation' object")
    else:
        if 'title' not in conversation:
            errors.append("Conversation missing 'title'")
        if 'speakers' not in conversation:
            errors.append("Conversation missing 'speakers'")
        if 'total_turns' not in conversation:
            errors.append("Conversation missing 'total_turns'")

    # Validate signals
    signals = response.get('signals', [])
    signal_errors = validate_signals(signals)
    errors.extend(signal_errors)

    # Validate patterns
    patterns = response.get('patterns', [])
    pattern_errors = validate_patterns(patterns)
    errors.extend(pattern_errors)

    # Validate trajectory
    trajectory = response.get('trajectory')
    if trajectory:
        trajectory_errors = validate_trajectory(trajectory)
        errors.extend(trajectory_errors)
    else:
        errors.append("Response missing 'trajectory' object")

    # Validate reframe
    reframe = response.get('reframe')
    if reframe:
        reframe_errors = validate_reframe(reframe)
        errors.extend(reframe_errors)
    else:
        errors.append("Response missing 'reframe' object")

    # Validate metadata
    metadata = response.get('metadata', {})
    if not metadata:
        errors.append("Response missing 'metadata' object")
    else:
        if 'processing_time_ms' not in metadata:
            errors.append("Metadata missing 'processing_time_ms'")
        if 'llm_calls' not in metadata:
            errors.append("Metadata missing 'llm_calls'")

    return errors
