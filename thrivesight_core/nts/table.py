"""Load and check the periodic table of the soul (elements.json, composites.json).

The JSON files are the source of truth. This module only reads them and refuses a
table that breaks the structure v7 states.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from math import prod
from pathlib import Path

HERE = Path(__file__).parent


def _is_prime(n: int) -> bool:
    return n >= 2 and all(n % d for d in range(2, int(n**0.5) + 1))


@dataclass(frozen=True)
class Table:
    data: dict
    composites: dict

    @property
    def elements(self) -> list[dict]:
        return self.data["elements"]

    @property
    def primes(self) -> list[dict]:
        return [e for e in self.elements if e["kind"] == "prime"]

    def element(self, key: str | int) -> dict:
        for e in self.elements:
            if e["id"] == key or e["number"] == key:
                return e
        raise KeyError(f"no element {key!r}")

    def category(self, cid: str) -> dict:
        return next(c for c in self.data["categories"] if c["id"] == cid)

    def composite_id(self, element_ids: list[str]) -> int:
        """The product of the primes: a name for an unordered multiset of primes.

        Identities are refused: Memory (0) would zero the product and Image (1)
        changes nothing, so neither can be a factor.
        """
        numbers = [self.element(i)["number"] for i in element_ids]
        if any(n < 2 for n in numbers):
            raise ValueError("Memory and Image are identities, not factors")
        return prod(numbers)

    def factor(self, n: int) -> list[str]:
        """Inverse of composite_id: the primes of n, smallest first."""
        out, rest = [], n
        for p in sorted(e["number"] for e in self.primes):
            while rest % p == 0:
                out.append(self.element(p)["id"])
                rest //= p
        if rest != 1:
            raise ValueError(f"{n} is not a product of the soul's primes")
        return out

    def check(self) -> None:
        els = self.elements
        ids = [e["id"] for e in els]
        nums = [e["number"] for e in els]
        assert len(set(ids)) == len(ids) and len(set(nums)) == len(nums), "duplicate element"
        identities = sorted(e["number"] for e in els if e["kind"] != "prime")
        assert identities == [0, 1], "the identities are Memory (0) and Image (1)"
        primes = sorted(e["number"] for e in self.primes)
        assert all(_is_prime(n) for n in primes), "every prime element has a prime number"
        assert primes == [n for n in range(2, primes[-1] + 1) if _is_prime(n)], "primes are consecutive from 2"
        cats = {c["id"]: c for c in self.data["categories"]}
        roles = {r["id"] for r in self.data["plane_roles"]}
        stages = set(self.data["development_stages"])
        for e in els:
            assert e["category"] in cats, f"{e['id']}: unknown category"
            assert e["number"] in cats[e["category"]]["elements"], f"{e['id']}: not listed in its category"
            assert e["plane_role"] in roles, f"{e['id']}: unknown plane role"
            assert e["develops"] in stages, f"{e['id']}: unknown stage"
            if e["kind"] == "prime":
                assert e["recurses"] is True, f"{e['id']}: every prime recurses (Axiom 5)"
                assert e["square"]["value"] == e["number"] ** 2, f"{e['id']}: wrong square"
            else:
                assert e["recurses"] is False, f"{e['id']}: identities do not recurse (Part XXIII)"
        listed = sorted(n for c in cats.values() for n in c["elements"])
        assert listed == sorted(nums), "every element is in exactly one category"
        for c in self.composites["composites"]:
            for path in c["paths"]:
                assert len(path) >= 2, f"{c['id']}: a composite has at least two factors"
                self.composite_id(path)


@lru_cache(maxsize=1)
def load_table() -> Table:
    data = json.loads((HERE / "elements.json").read_text())
    composites = json.loads((HERE / "composites.json").read_text())
    table = Table(data, composites)
    table.check()
    return table
