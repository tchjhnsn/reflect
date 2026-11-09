import unittest
from datetime import date

from thrivesight_core.nts.onboarding import ORDER, AnswerError, Onboarding, hold_stances
from thrivesight_core.nts.person import FromJournal, Person

TODAY = date(2026, 11, 9)
AT = "2026-11-09"


def run(answers):
    ob = Onboarding(TODAY)
    for a in answers:
        ob.answer(a)
    p = Person("p1")
    ob.finish(p, AT)
    return p


class OnboardingTest(unittest.TestCase):
    def test_order_comes_from_lean(self):
        self.assertEqual([q["id"] for q in ORDER],
                         ["name", "date_of_birth", "occupation", "closest_friend", "marital_status",
                          "seeing_someone", "children", "hold_partnership", "hold_children"])

    def test_family_and_holds(self):
        p = run(["Ada", "", "", "", "married", "no", "2", "desire,duty; it matters to my family", "sacred"])
        self.assertEqual(p.current("marital_status"), "married")
        self.assertEqual(p.current("children"), "2")
        self.assertEqual([b.text for b in p.beliefs], ["desire,duty; it matters to my family", "sacred"])
        self.assertNotIn("hold_partnership", [f.key for f in p.facts])
        self.assertEqual(hold_stances("desire,duty; x"), {"desire": "descriptive", "duty": "normative"})

    def test_bad_choices_refused(self):
        for skip, bad in [(4, "engaged"), (5, "maybe"), (6, "two"), (7, "love"), (7, "unsure,desire")]:
            ob = Onboarding(TODAY)
            for _ in range(skip):
                ob.skip()
            with self.assertRaises(AnswerError):
                ob.answer(bad)

    def test_full_run_files_each_answer(self):
        p = run(["Ada", "1990-04-01", "Engineer", "Sam"])
        self.assertEqual(p.current("name"), "Ada")
        self.assertEqual(p.current("date_of_birth"), "1990-04-01")
        self.assertEqual(p.current("occupation"), "Engineer")
        self.assertEqual(p.current_relationship("closest_friend"), "Sam")
        self.assertTrue(all(f.status == "confirmed" for f in p.facts))

    def test_all_skipped_writes_nothing(self):
        p = run(["", "  ", "", ""])
        self.assertEqual((p.facts, p.relationships), ([], []))

    def test_friend_only(self):
        ob = Onboarding(TODAY)
        for _ in range(3):
            ob.skip()
        ob.answer("Sam")
        p = Person("p1")
        ob.finish(p, AT)
        self.assertEqual((len(p.facts), len(p.relationships)), (0, 1))
        self.assertEqual(ob.question, "marital_status")

    def test_back_keeps_and_replaces_answers(self):
        ob = Onboarding(TODAY)
        ob.answer("Ada")
        ob.answer("1990-04-01")
        ob.answer("Engineer")
        ob.back()
        ob.answer("Teacher")
        self.assertEqual(ob.answers["occupation"], "Teacher")
        self.assertEqual(ob.answers["name"], "Ada")

    def test_future_birth_date_refused(self):
        ob = Onboarding(TODAY)
        ob.answer("Ada")
        with self.assertRaises(AnswerError) as e:
            ob.answer("2030-01-01")
        self.assertEqual(e.exception.question, "date_of_birth")
        self.assertEqual(ob.question, "date_of_birth")

    def test_new_closest_friend_closes_old(self):
        p = Person("p1")
        p.state_relationship("Sam", "closest_friend", "2026-01-01", "2026-01-01")
        p.state_relationship("Kai", "closest_friend", AT, AT)
        self.assertEqual(p.relationships[0].valid_to, AT)
        self.assertEqual(p.current_relationship("closest_friend"), "Kai")

    def test_journal_source_refused(self):
        p = Person("p1")
        src = FromJournal("e1", "q", "g1")
        with self.assertRaises(PermissionError):
            p.state_relationship("Sam", "closest_friend", AT, AT, source=src)


if __name__ == "__main__":
    unittest.main()
