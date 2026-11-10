import json
import unittest

from . import salience as s

ALL = {p: 0.2 for p in s.PRIMES}


class SalienceTest(unittest.TestCase):
    def test_parses_and_checks_self_terms(self):
        text = "I'm six foot and everyone calls me the big one."
        ans = json.dumps({"weights": {**ALL, "self-differentiation": 0.9}, "self_terms": [
            {"quote": "the big one", "term": "big", "whose": "self", "basis": "logos"},
            {"quote": "not in text", "term": "x", "whose": "self", "basis": "logos"},
            {"quote": "six foot", "term": "tall", "whose": "self", "basis": "self-differentiation"},
        ]})
        sal, problems = s.parse(text, ans)
        self.assertEqual([t.term for t in sal.self_terms], ["big"])
        self.assertEqual(len(problems), 2)

    def test_refuses_missing_or_out_of_range_weights(self):
        w = dict(ALL); w.pop("will"); w["desire"] = 1.5
        sal, problems = s.parse("x", json.dumps({"weights": w}))
        self.assertIsNone(sal)
        self.assertEqual(len(problems), 2)
        self.assertEqual(s.parse("x", "nope"), (None, ["not JSON"]))

    def test_next_focus_is_most_prominent_unasked_body_first_on_ties(self):
        sal = s.Salience({**ALL, "emotion": 0.8, "attachment": 0.8}, [])
        self.assertEqual(s.next_focus(sal, set()), "attachment")
        self.assertEqual(s.next_focus(sal, {"attachment"}), "emotion")
        self.assertEqual(s.next_focus(s.Salience(ALL, []), set()), "embodiment")
        self.assertIsNone(s.next_focus(sal, set(s.PRIMES)))

    def test_apprehensions_are_proper_everywhere_and_mismatch_only_asks_when_asked(self):
        for d in s.DOMAINS:
            self.assertIn("aesthetic-sense", s.proper(d))
            self.assertIn("numinous-sense", s.proper(d))
        self.assertEqual(s.proper("matter_of_fact")[0], "reason")
        self.assertIsNone(s.mismatch(False, "matter_of_fact", "emotion"))
        self.assertEqual(s.mismatch(True, "matter_of_fact", "emotion"), "What tells you that?")
        self.assertIsNone(s.mismatch(True, "matter_of_fact", "numinous-sense"))

    def test_prompt_never_asks_to_advise(self):
        p = s.prompt().lower()
        self.assertIn("never advise", p)
        self.assertNotIn("you should", p)


if __name__ == "__main__":
    unittest.main()
