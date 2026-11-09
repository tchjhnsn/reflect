"""Run: python -m unittest thrivesight_core.nts.test_person"""
import unittest

from thrivesight_core.nts.journal import TELL_ME_MORE, Entry
from thrivesight_core.nts.person import FromJournal, Person
from thrivesight_core.nts.records import Act


class JournalTest(unittest.TestCase):
    def test_tell_me_more_until_done(self):
        e = Entry("e1", "p1", "2026-11-09T10:00:00Z")
        self.assertEqual(e.write("Went to the park."), TELL_ME_MORE)
        self.assertEqual(e.write("It rained."), TELL_ME_MORE)
        e.done()
        self.assertFalse(e.show_cues)
        with self.assertRaises(ValueError):
            e.write("more")
        self.assertEqual(e.text, "Went to the park.\n\nIt rained.")


class PersonTest(unittest.TestCase):
    def setUp(self):
        self.p = Person("p1", journal={"e1": "I started the new job at the clinic today. Honesty matters more to me than comfort."})

    def test_stated_fact_closes_old_value(self):
        self.p.state_fact("occupation", "teacher", "2020-01-01", "2026-11-09")
        self.p.state_fact("occupation", "nurse", "2026-11-09", "2026-11-09")
        self.assertEqual(self.p.current("occupation"), "nurse")
        self.assertEqual(self.p.facts[0].valid_to, "2026-11-09")

    def test_journal_needs_a_grant(self):
        src = FromJournal("e1", "started the new job at the clinic", "g1")
        with self.assertRaises(PermissionError):
            self.p.state_fact("occupation", "clinic", "2026-11-09", "2026-11-09", src)
        self.p.grant("g1", "profile", ["e1"])
        fact = self.p.state_fact("occupation", "clinic", "2026-11-09", "2026-11-09", src)
        self.assertEqual(fact.status, "proposed")
        self.assertIsNone(self.p.current("occupation"))
        self.p.confirm_fact(fact)
        self.assertEqual(self.p.current("occupation"), "clinic")

    def test_grant_is_per_store(self):
        self.p.grant("g1", "profile", ["e1"])
        with self.assertRaises(PermissionError):
            self.p.state_belief("Honesty over comfort", "2026-11-09", FromJournal("e1", "Honesty matters more to me than comfort", "g1"))

    def test_quote_must_be_in_entry(self):
        self.p.grant("g2", "beliefs", ["e1"])
        with self.assertRaises(ValueError):
            self.p.state_belief("Family first", "2026-11-09", FromJournal("e1", "family comes first", "g2"))

    def test_revoke_flags_and_blocks(self):
        self.p.grant("g2", "beliefs", ["e1"])
        b = self.p.state_belief("Honesty over comfort", "2026-11-09", FromJournal("e1", "Honesty matters more to me than comfort", "g2"))
        self.assertEqual(self.p.revoke("g2"), [b])
        with self.assertRaises(PermissionError):
            self.p.confirm_belief(b)


class ActTest(unittest.TestCase):
    def test_drivers_and_target(self):
        Act((("conscience", 6), ("attachment", 8)), ("her", "emotion"))
        Act((("numinous-sense", 9),))
        for bad in [(), (("will", 3), ("will", 4)), (("memory", 3),), (("will", 11),)]:
            with self.assertRaises((ValueError, KeyError)):
                Act(bad)


if __name__ == "__main__":
    unittest.main()
