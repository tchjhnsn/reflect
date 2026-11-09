"""Run: python -m unittest thrivesight_core.nts.test_investigate"""
import unittest

from thrivesight_core.nts import load_table
from thrivesight_core.nts.investigate import STRUCTURE, WILL, check_question, parse_cues, will_tree


class InvestigateTest(unittest.TestCase):
    def test_lean_export_matches_table(self):
        table = {e["id"]: e["number"] for e in load_table().primes}
        self.assertEqual({p["id"]: p["number"] for p in STRUCTURE["primes"]}, table)

    def test_every_facet_has_a_question(self):
        tree = will_tree(coerced=True)
        self.assertEqual(len(tree), len(STRUCTURE["will_order"]) + len(STRUCTURE["coercion_sources"]))
        self.assertEqual(len(WILL["root"]), len(STRUCTURE["will_order"]))

    def test_stance_comes_from_lean(self):
        stances = {a.facet: a.stance for a in will_tree()}
        self.assertEqual(stances["conscience"], "normative")
        self.assertEqual(stances["desire"], "descriptive")

    def test_first_question_is_whether_it_was_chosen(self):
        self.assertEqual(will_tree()[0].facet, "voluntary_involuntary")

    def test_leading_questions_refused(self):
        for bad in ["Weren't you disappointed?", "You should rest.", "You will feel better?", "Tell me about it."]:
            with self.assertRaises(ValueError, msg=bad):
                check_question(bad)
        check_question("How did you feel when it rained?")

    def test_cues_fail_closed(self):
        entry = "I went to the park yesterday to play basketball with some friends, but it rained"
        cues, refused = parse_cues(entry, '{"cues": [{"element": "will", "quote": "I went"}, '
                                          '{"element": "emotion", "quote": "I was sad"}, {"element": "memory", "quote": "yesterday"}]}')
        self.assertEqual([c.element for c in cues], ["will"])
        self.assertEqual(len(refused), 2)


if __name__ == "__main__":
    unittest.main()
