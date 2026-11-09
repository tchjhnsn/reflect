"""Tests for the element model. Run: python -m unittest thrivesight_core.nts.test_nts"""
import unittest
from math import comb

from thrivesight_core.nts import RealityTags, ScopePath, load_table
from thrivesight_core.nts.agreement import agreement, kappa
from thrivesight_core.nts.reader import parse_answer, system_prompt


class TableTest(unittest.TestCase):
    def test_shape(self):
        t = load_table()
        self.assertEqual(len(t.elements), 18)
        self.assertEqual(len(t.primes), 16)
        self.assertEqual(len(t.data["categories"]), 9)
        self.assertTrue(all(len(c["elements"]) == 2 for c in t.data["categories"]))

    def test_counts_v7_states(self):
        # Part XXIV: 65,519 composites of 2 to 16 primes; 502 composite fields of 2 to 9 categories.
        self.assertEqual(sum(comb(16, k) for k in range(2, 17)), 65519)
        self.assertEqual(sum(comb(9, k) for k in range(2, 10)), 502)

    def test_composite_ids_match_v7(self):
        t = load_table()
        self.assertEqual(t.composite_id(["will", "emotion"]), 14)
        self.assertEqual(t.composite_id(["attachment", "emotion", "temporality"]), 6923)
        self.assertEqual(t.composite_id(["desire", "logos", "emotion"]), 777)
        self.assertEqual(t.factor(6923), ["emotion", "attachment", "temporality"])

    def test_identities_are_not_factors(self):
        t = load_table()
        with self.assertRaises(ValueError):
            t.composite_id(["memory", "reason", "intuition", "conscience"])
        with self.assertRaises(ValueError):
            t.composite_id(["image", "desire"])

    def test_product_loses_order(self):
        # Why scope paths exist: the product cannot tell Emotion[Will] from Will[Emotion].
        t = load_table()
        self.assertEqual(t.composite_id(["emotion", "will"]), t.composite_id(["will", "emotion"]))
        self.assertNotEqual(ScopePath.parse("emotion[will]"), ScopePath.parse("will[emotion]"))


class RecordsTest(unittest.TestCase):
    def test_scope_paths(self):
        self.assertEqual(str(ScopePath.parse("emotion[will][memory]")), "emotion[will][memory]")
        for bad in ["memory", "memory[emotion]", "emotion[memory][will]", "image", "emotion[image]", "courage"]:
            with self.assertRaises((ValueError, KeyError), msg=bad):
                ScopePath.parse(bad)

    def test_reality_poles(self):
        RealityTags(shared_inner="inner", present_represented="represented")
        with self.assertRaises(ValueError):
            RealityTags(shared_inner="embodied")


class ReaderTest(unittest.TestCase):
    ENTRY = "I wanted the second glass but told myself no and went for a walk."

    def test_prompt_lists_every_element(self):
        prompt = system_prompt()
        for e in load_table().elements:
            self.assertIn(f"id \"{e['id']}\"", prompt)

    def test_fails_closed(self):
        answer = """{"proposals": [
          {"scope": "desire", "amplitude": 0.6, "quote": "I wanted the second glass"},
          {"scope": "will", "amplitude": 0.7, "quote": "told myself  no"},
          {"scope": "conscience", "amplitude": 0.5, "quote": "I felt guilty"},
          {"scope": "memory", "amplitude": 0.5, "quote": "went for a walk"},
          {"scope": "desire", "amplitude": 1.5, "quote": "I wanted"}
        ], "unsure": []}"""
        accepted, refused = parse_answer(self.ENTRY, answer)
        self.assertEqual([str(p.scope) for p in accepted], ["desire", "will"])
        self.assertEqual(len(refused), 3)


class AgreementTest(unittest.TestCase):
    def test_kappa(self):
        self.assertAlmostEqual(kappa(5, 0, 0, 5), 1.0)
        self.assertIsNone(kappa(0, 0, 0, 0))
        report = agreement({"a": {"will"}, "b": {"desire"}}, {"a": {"will"}, "b": {"will"}})
        self.assertEqual(report["will"]["both"], 1)
        self.assertEqual(report["will"]["person_only"], 1)
        self.assertEqual(report["desire"]["model_only"], 1)


if __name__ == "__main__":
    unittest.main()
