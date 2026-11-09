import NTS.Output
/-!
# The person: journal, profile, beliefs, grants, and acts

Four stores, each with one source (Torian, 2026-11-09):

* the journal holds the person's own words, and no model reads it unless the person allows;
* the profile holds dated facts about the person (birth, family, schools, body, work);
* the beliefs store holds what the person holds ought to be (values, duties);
* reflection records hold descriptive answers (`Record` in `Output`).

Facts and beliefs are explicit, never implicit. Each has exactly one of two sources: the
person stated it, or it was taken from a journal entry under a permission the person
granted. There is no constructor for "a model inferred it", so nothing inferred can enter
either store. An item taken from a journal is only a proposal until the person confirms it.
-/
namespace NTS

/-- Which store a grant may feed. [D] -/
inductive Store where
  | profile | beliefs
  deriving DecidableEq, Repr

/-- A permission the person gives to take items from journal entries into one store.
Revocable. [D] -/
structure Grant where
  id : String
  store : Store
  entries : List String   -- the entry ids it covers
  revoked : Bool := false
  deriving DecidableEq, Repr

/-- Where a fact or belief came from. Two constructors only. [D] -/
inductive Source where
  | stated
  | fromJournal (entry : String) (quote : String) (grant : String)
  deriving DecidableEq, Repr

/-- Proposed items wait for the person; only confirmed items count. [D] -/
inductive Status where
  | proposed | confirmed
  deriving DecidableEq, Repr

/-- A fact about the person, with when it was true. A later value never overwrites an
earlier one; it closes it (`validTo`). Dates are ISO 8601 strings. [D] -/
structure Fact where
  key : String          -- e.g. "date_of_birth", "occupation", "height_cm"
  value : String
  validFrom : String
  validTo : Option String := none
  recordedAt : String
  source : Source
  status : Status
  deriving Repr

/-- A belief or value: what the person holds ought to be, in their own words. [D] -/
structure Belief where
  text : String
  recordedAt : String
  source : Source
  status : Status
  deriving Repr

/-- A source is allowed into a store when it was stated, or when a grant with that id
covers the entry, feeds that store, and has not been revoked. [D] -/
def Source.allowed (grants : List Grant) (store : Store) : Source → Bool
  | .stated => true
  | .fromJournal entry _ g =>
    grants.any fun gr => gr.id == g && gr.store == store && !gr.revoked && gr.entries.contains entry

/-- Revoking a grant removes the permission for everything taken under it. -/
theorem revoked_not_allowed (gr : Grant) (s : Store) (e q : String)
    (h : gr.revoked = true) : (Source.fromJournal e q gr.id).allowed [gr] s = false := by
  simp [Source.allowed, h]

/-- A grant for the profile does not let anything into the beliefs store. -/
theorem grant_is_per_store (gr : Grant) (e q : String) (h : gr.store = .profile) :
    (Source.fromJournal e q gr.id).allowed [gr] .beliefs = false := by
  simp [Source.allowed, h]

/-- What the person stated is always allowed. -/
theorem stated_allowed (gs : List Grant) (s : Store) : Source.stated.allowed gs s = true := rfl

/-- Whose element an act is aimed at. [D] -/
inductive Whose where
  | self
  | other (name : String)
  deriving DecidableEq, Repr

/-- An act of will, as the person describes it: what drove it, weighted by the person, and
what it was aimed at. "I had to cheer myself up" is driven by Conscience, aimed at the
person's own Emotion. "I couldn't let her sulk all day" is driven by Conscience and
Attachment, aimed at her Emotion. "I felt called" is driven by Numinous Sense. A will is an
arrangement of drivers, not a switch (Torian, 2026-11-09). [S] False if some acts turn out to
have no driver other than Will itself. -/
structure Act where
  drivers : List (Prime × Nat)   -- each driver with the person's weight, 0 to 10
  target : Option (Whose × Prime) := none
  deriving Repr

/-- An act's drivers name each prime at most once and carry weights from 0 to 10. [D] -/
def Act.wellFormed (a : Act) : Bool :=
  (a.drivers.map Prod.fst).Nodup && a.drivers.all (fun d => d.2 ≤ 10)

example : Act.wellFormed ⟨[(.conscience, 7), (.attachment, 5)], some (.other "her", .emotion)⟩ = true := by decide
example : Act.wellFormed ⟨[(.conscience, 7), (.conscience, 2)], none⟩ = false := by decide

/-- Who another person is to the person, in the person's own word. Onboarding asks for the
closest friend first. [D] -/
inductive RelationKind where
  | closestFriend
  deriving DecidableEq, Repr

/-- Another person in the person's life, named as the person writes it. Friends change, so a
relationship is dated like a fact: a new closest friend closes the old one. Onboarding only
takes what the person states. [D] -/
structure Relationship where
  name : String
  kind : RelationKind
  validFrom : String
  validTo : Option String := none
  recordedAt : String
  source : Source
  deriving Repr

/-- What kind of puzzle it is, if the person chooses to say: perennial (it cannot be stated
solved), generational (solved only over a lifetime or more), or tractable (it can become a
problem with goals). [P] -/
inductive PuzzleKind where
  | perennial | generational | tractable
  deriving DecidableEq, Repr

/-- The puzzle of the person's life, in their words. It is not a belief: it is a question the
person lives with, not a claim they hold true. [D] -/
structure Puzzle where
  text : String
  kind : Option PuzzleKind := none
  statedAt : String
  source : Source
  deriving Repr

end NTS
