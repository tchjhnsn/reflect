import NTS.Output
import NTS.Person
/-!
# The state of the elements in a moment

Torian, 2026-11-09. Every experience happens in shared reality, at a clock time and a place,
in a body in some state. The mind does not hold "I am 6'0"" as such: it holds what 6'0" is
said to be like, by comparison. Self-Differentiation runs that comparison through another
element (Logos, Emotion, Imagination...), and that element is the basis of the self-term.
Matters of fact are properly governed by Reason, and each domain has its proper governor;
the two Apprehensions can properly read anything, which is what makes poetry possible.
Numinous Sense can absorb consciousness and the drives.

This file states those as types. Nothing here is inferred by a model: a model may only
propose weights and bases (`Salience` in the TS core), and the person confirms.
-/
namespace NTS

/-! ## Shared reality -/

/-- Where a moment sits in shared reality. The clock is external and recorded by the
device; place and body state are stated by the person, or absent. [D] -/
structure Anchor where
  /-- Seconds since the epoch, UTC: external time, the same for everyone. -/
  clock : Nat
  /-- The person's local calendar date, YYYY-MM-DD. -/
  localDate : String
  place : Option String := none
  bodyState : Option String := none
  deriving Repr

/-- Temporality has two sides: the external clock it is anchored to, and lived time as the
person states it ("it dragged", "years ago"). Both are kept; neither replaces the other. [D] -/
structure TimeLink where
  anchor : Anchor
  felt : Option String := none
  deriving Repr

/-! ## Regions: how the two elements of a category are arranged -/

/-- The two elements of a category, in number order. [D] -/
def Prime.Category.pair : Prime.Category → Prime × Prime
  | .drives => (.will, .desire)
  | .generators => (.imagination, .logos)
  | .sensors => (.emotion, .intuition)
  | .regulators => (.conscience, .reason)
  | .illuminators => (.attention, .awareness)
  | .binders => (.selfDifferentiation, .attachment)
  | .matrices => (.embodiment, .temporality)
  | .apprehensions => (.aestheticSense, .numinousSense)

theorem pair_in_category (c : Prime.Category) : c.pair.1.category = c ∧ c.pair.2.category = c := by
  cases c <;> decide

/-- How a category's two elements stand to each other in a moment. [S] False if some
pair relates in a way none of these names, which Torian has said may be so ("not quite"). -/
inductive Arrangement where
  | firstLeads | secondLeads | joined | opposed
  deriving DecidableEq, Repr

/-- How strongly a category is active. There is no "off": every category is active in every
experience, even when low (Torian, 2026-11-09). `absorbed` is reserved for Numinous Sense. [D] -/
inductive Amplitude where
  | low | raised | dominant | absorbed
  deriving DecidableEq, Repr

structure Region where
  arrangement : Arrangement
  amplitude : Amplitude
  deriving DecidableEq, Repr

/-- A moment's configuration: one region per category, all at once. [D] -/
abbrev Config := Prime.Category → Region

/-- Only the Apprehensions can be absorbed, and only with Numinous Sense leading. [S] False
if Aesthetic Sense, or anything else, can consume consciousness the same way. -/
def Region.ok (c : Prime.Category) (r : Region) : Prop :=
  r.amplitude = .absorbed → c = .apprehensions ∧ r.arrangement = .secondLeads

instance (c : Prime.Category) (r : Region) : Decidable (r.ok c) := by
  unfold Region.ok; infer_instance

def Config.ok (k : Config) : Prop := ∀ c, (k c).ok c

def Config.absorbed (k : Config) : Prop := (k .apprehensions).amplitude = .absorbed

/-- Under absorption Numinous Sense is the one that leads. -/
theorem absorbed_numinous_leads (k : Config) (h : k.ok) (ha : k.absorbed) :
    (k .apprehensions).arrangement = .secondLeads ∧ Prime.Category.apprehensions.pair.2 = Prime.numinousSense :=
  ⟨((h .apprehensions) ha).2, rfl⟩

/-- No other category can be absorbed. -/
theorem only_apprehensions_absorb (k : Config) (h : k.ok) (c : Prime.Category)
    (ha : (k c).amplitude = .absorbed) : c = .apprehensions := ((h c) ha).1

/-- While absorbed, every scope kept for the moment is held inside Numinous Sense. [S]
Torian: Numinous Sense "can sometimes completely consume consciousness and drives". -/
def scopesOk (k : Config) (scopes : List Scope) : Prop :=
  k.absorbed → ∀ s ∈ scopes, s.outermost = .numinousSense

/-! ## Facts and self-terms -/

/-- A comparative term a person assigns ("tall", "quiet", "a late bloomer"). Its scope starts
at Self-Differentiation; the next element down is the basis the comparison runs through. A
grounded fact it rests on, if any, is named by key and lives in the profile. [D] -/
structure SelfTerm where
  term : String
  /-- One's own term, or one for another person (`Whose` from Person). -/
  whose : Whose
  comparedTo : Option String := none
  scope : Scope
  fact : Option String := none
  deriving Repr

/-- The element a self-term's comparison runs through: the second step of its scope. -/
def SelfTerm.basis (t : SelfTerm) : Option Prime := t.scope.path[1]?

/-- A well-formed self-term: rooted at Self-Differentiation, with a basis, within the depth
cap. -/
def SelfTerm.ok (t : SelfTerm) : Prop :=
  t.scope.outermost = .selfDifferentiation ∧ 2 ≤ t.scope.depth ∧ t.scope.depth ≤ maxDepth

/-- "I'm tall", compared through Logos and checked by Reason, against the stated height. -/
def tallExample : SelfTerm :=
  { term := "tall", whose := .self, comparedTo := some "people at work",
    scope := .inner (.inner (.root .selfDifferentiation) .logos) .reason, fact := some "height" }

example : tallExample.basis = some .logos := rfl
example : tallExample.ok := by
  refine ⟨rfl, ?_, ?_⟩ <;> decide

/-! ## Proper elements -/

/-- What an experience or claim is about. [P] The list grows as Torian names domains. -/
inductive Domain where
  | matterOfFact | rightAndWrong | beauty | sacred | howIFeel
  deriving DecidableEq, Repr

namespace Domain

def all : List Domain := [matterOfFact, rightAndWrong, beauty, sacred, howIFeel]

def id : Domain → String
  | matterOfFact => "matter_of_fact" | rightAndWrong => "right_and_wrong" | beauty => "beauty"
  | sacred => "sacred" | howIFeel => "how_i_feel"

/-- The element that properly governs a domain (Torian, 2026-11-09). [S] -/
def governor : Domain → Prime
  | matterOfFact => .reason
  | rightAndWrong => .conscience
  | beauty => .aestheticSense
  | sacred => .numinousSense
  | howIFeel => .emotion

/-- The elements that may properly read a domain: its governor and both Apprehensions,
which can properly model nearly anything. [S] False if some domain is closed to them. -/
def proper (d : Domain) : List Prime :=
  ([d.governor, .aestheticSense, .numinousSense]).eraseDups

end Domain

/-- An Apprehension is never improper, in any domain. -/
theorem apprehensions_always_proper (d : Domain) :
    Prime.aestheticSense ∈ d.proper ∧ Prime.numinousSense ∈ d.proper := by
  cases d <;> decide

theorem governor_proper (d : Domain) : d.governor ∈ d.proper := by
  cases d <;> decide

/-- A matter of fact set by feeling is a mismatch; read through beauty, it is not. -/
example : Prime.emotion ∉ Domain.matterOfFact.proper := by decide
example : Prime.aestheticSense ∈ Domain.matterOfFact.proper := by decide

/-- What the system may do with a mismatch: nothing unless the person asked, and then only
an open question. It is a descriptive observation, never a correction. [D] -/
def onMismatch (asked : Bool) (d : Domain) (basis : Prime) (s : Scope) : Option ModelOutput :=
  if asked && !(d.proper.contains basis) then
    some (.ask { scope := s, facet := .prime basis, text := "What tells you that?" })
  else none

/-- A mismatch is silent unless the person asked. -/
theorem mismatch_silent (d : Domain) (b : Prime) (s : Scope) : onMismatch false d b s = none := by
  simp [onMismatch]

/-- When it speaks, a mismatch only ever asks. -/
theorem mismatch_only_asks (a : Bool) (d : Domain) (b : Prime) (s : Scope) (o : ModelOutput)
    (h : onMismatch a d b s = some o) : ∃ q, o = .ask q := by
  unfold onMismatch at h
  split at h
  · cases h; exact ⟨_, rfl⟩
  · cases h

/-- An Apprehension basis never raises a mismatch, asked or not. -/
theorem apprehension_never_flagged (a : Bool) (d : Domain) (s : Scope) :
    onMismatch a d .aestheticSense s = none ∧ onMismatch a d .numinousSense s = none := by
  cases d <;> cases a <;> simp [onMismatch, Domain.proper, Domain.governor, List.eraseDups] <;> decide

end NTS
