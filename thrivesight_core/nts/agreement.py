"""Agreement between a model's readings and the person's own labels.

This is what replaces training: the person's corrections are the test set. For each
prime, it counts entries where both, only the model, only the person, or neither said
the prime was operating, and reports Cohen's kappa. Only the top-level element of each
scope path is compared.
"""
from __future__ import annotations

from .table import load_table


def kappa(both: int, model_only: int, person_only: int, neither: int) -> float | None:
    n = both + model_only + person_only + neither
    if n == 0:
        return None
    observed = (both + neither) / n
    p_model = (both + model_only) / n
    p_person = (both + person_only) / n
    expected = p_model * p_person + (1 - p_model) * (1 - p_person)
    return None if expected == 1 else (observed - expected) / (1 - expected)


def agreement(model: dict[str, set[str]], person: dict[str, set[str]]) -> dict[str, dict]:
    """model and person map entry id -> set of prime ids. Entries the person has not labelled are skipped."""
    entries = [e for e in person if e in model]
    report = {}
    for prime in (p["id"] for p in load_table().primes):
        cells = {"both": 0, "model_only": 0, "person_only": 0, "neither": 0}
        for e in entries:
            m, p = prime in model[e], prime in person[e]
            cells["both" if m and p else "model_only" if m else "person_only" if p else "neither"] += 1
        report[prime] = {**cells, "kappa": kappa(**cells)}
    return report
